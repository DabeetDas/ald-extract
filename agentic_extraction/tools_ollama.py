import json
import re
import ast
from typing import Dict, Any

# FIX: removed json5 dependency — the cleaning steps make standard json.loads
#      sufficient, and json5 is not a stdlib package.
# FIX: removed PromptTemplate import — it was only used as a plain string
#      formatter with no LangChain chaining. Plain f-strings are cleaner and
#      remove the langchain-core dependency for this module.

def _clone_default(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _looks_like_json_schema(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    schema_keys = {"type", "properties", "items", "required", "additionalProperties"}
    return bool(schema_keys.intersection(value.keys())) and (
        "properties" in value or value.get("type") in {"object", "array"}
    )


def _coerce_scalar(value: Any, template: Any) -> Any:
    if value is None:
        return None if template is None else _clone_default(template)

    # `None` in the template means "nullable scalar" in these outputs.
    if template is None:
        return value if isinstance(value, (str, int, float, bool)) else None

    if isinstance(template, str):
        return value if isinstance(value, str) else _clone_default(template)
    if isinstance(template, (int, float)):
        return value if isinstance(value, (int, float)) else _clone_default(template)
    if isinstance(template, bool):
        return value if isinstance(value, bool) else _clone_default(template)
    return value


def _coerce_to_template(value: Any, template: Any) -> Any:
    if isinstance(template, dict):
        if isinstance(value, list):
            # Salvage outputs like characterization -> ["XPS", "AFM", ...]
            list_keys = [key for key, item in template.items() if isinstance(item, list)]
            if len(list_keys) == 1 and set(template.keys()) <= {list_keys[0], "evidence"}:
                value = {list_keys[0]: value}
            else:
                raise ValueError("Model returned a list where an object was required.")

        if not isinstance(value, dict):
            raise ValueError("Model returned a non-object for a structured schema.")
        if not value:
            raise ValueError("Model returned an empty object.")
        if _looks_like_json_schema(value):
            raise ValueError("Model returned a JSON Schema definition instead of extracted data.")

        overlapping_keys = set(value.keys()).intersection(template.keys())
        if not overlapping_keys:
            raise ValueError("Model output did not contain any expected schema keys.")

        coerced: dict[str, Any] = {}
        for key, template_value in template.items():
            if key in value:
                coerced[key] = _coerce_to_template(value[key], template_value)
            else:
                coerced[key] = _clone_default(template_value)
        return coerced

    if isinstance(template, list):
        return value if isinstance(value, list) else _clone_default(template)

    return _coerce_scalar(value, template)


def robust_json_parse(text: Any, default: dict | None = None) -> dict:
    """
    Tries multiple strategies to recover valid JSON from LLM output.

    FIX: accepts a configurable `default` so callers get a schema-appropriate
         fallback instead of always {"materials": []} which was wrong for most
         agents (they expect keys like "target_material", "precursors", etc.).
    FIX: the blanket replace("'", '"') mangling has been removed — it corrupts
         chemical formulas like Si-OH*, DMAH, etc.  A targeted fix for
         Python-style None/True/False literals is applied instead.
    """
    if default is None:
        default = {}

    # Unwrap AIMessage-style objects
    if hasattr(text, "content"):
        text = text.content

    if not isinstance(text, str):
        return default

    # Strip Markdown code fences
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()

    # Extract the first complete JSON object or array
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        text = match.group(1)

    # Remove trailing commas before ] or }
    text = re.sub(r',\s*([\]}])', r'\1', text)

    # FIX: replace Python literal keywords with JSON equivalents *only* when
    #      they appear as standalone tokens, to avoid corrupting chemical names.
    text = re.sub(r'\bNone\b',  'null',  text)
    text = re.sub(r'\bTrue\b',  'true',  text)
    text = re.sub(r'\bFalse\b', 'false', text)

    parsed = None

    # Try standard JSON parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        # Try Python literal eval as last resort (handles some edge cases)
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None

    if parsed is None:
        preview = text[:300].replace("\n", " ")
        raise ValueError(f"Could not parse model output as JSON. Preview: {preview!r}")

    return _coerce_to_template(parsed, default)


# ── Shared prompt builder ─────────────────────────────────────────────────────
_SYSTEM_PREAMBLE = """\
You are a materials science expert specializing in Atomic Layer Deposition (ALD).

Your task is to extract structured scientific information from the provided research paper text.

Rules:
1. Only extract information explicitly supported by the text.
2. If information is missing, return null.
3. Do not infer or hallucinate values.
4. Include the exact evidence sentence from the paper when possible.
5. Return output strictly in JSON format.
6. Use standard scientific units where possible.
7. Do not return an all-empty placeholder template if the text explicitly contains the answer.
"""


# === Tool 0: Read fulltext ====================================================
def read_fulltext(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# === Tool 1: Summariser =======================================================
def summariser_agent(fulltext: str, llm) -> Dict[str, Any]:  # FIX: correct return type
    prompt = _SYSTEM_PREAMBLE + f"""
Task: Extract a concise summary of the Atomic Layer Deposition (ALD) process described in the text.

Focus on:
- material deposited
- main precursors
- process type (thermal ALD or PEALD)
- key temperature range
- purpose of the study

Output JSON schema:

{{
  "target_material": "",
  "process_type": "",
  "main_precursors": [],
  "temperature_range": "",
  "summary": "",
  "evidence": ""
}}

"""  + "\nText:\n```" + fulltext + "```"
    output = llm.invoke(prompt)
    # FIX: pass a schema-appropriate default so failures degrade gracefully
    return robust_json_parse(output, default={
        "target_material": None, "process_type": None,
        "main_precursors": [], "temperature_range": None,
        "summary": None, "evidence": None,
    })


# === Tool 2: Target materials =================================================
def target_materials_agent(fulltext: str, llm) -> Dict[str, Any]:  # FIX: correct return type
    prompt = _SYSTEM_PREAMBLE + f"""
Identify the primary material deposited using Atomic Layer Deposition. Ignore substrates, precursors, or reactor components. Focus ONLY on the primary target film being synthesized or studied.

Extract the following information:
- chemical_formula: The stoichiometric or chemical formula of the deposited material (e.g., "Al2O3", "TiO2"). If not explicitly stated, infer it only if the common name makes it chemically unambiguous. Otherwise, return null.
- material_name: The common English name of the material (e.g., "Aluminum oxide", "Titanium dioxide").
- material_class: The broad classification of the deposited material (e.g., "Oxide", "Nitride", "Pure Metal", "Sulfide", "2D Material"). 
- evidence: A direct, exact quote from the text that justifies your extraction.

Output JSON:

{{
  "target_material": {{
    "chemical_formula": "",
    "material_name": "",
    "material_class": ""
  }},
  "evidence": ""
}}

"""  + "\nText:\n```" + fulltext + "```"
    output = llm.invoke(prompt)
    return robust_json_parse(output, default={
        "target_material": {"chemical_formula": None, "material_name": None, "material_class": None},
        "evidence": None,
    })


# === Tool 3: Precursor / coreactant / purge ===================================
def precurosr_coreactant_purge_agent(fulltext: str, llm) -> Dict[str, Any]:  # FIX: correct return type
    prompt = _SYSTEM_PREAMBLE + f"""
Extract the specific chemical inputs used during the Atomic Layer Deposition (ALD) process. 

Follow these strict definitions for extraction:
- precursors: The primary metal, semiconductor, or organometallic sources (e.g., "TMA", "TiCl4", "Diethylzinc"). Include both the abbreviation and full name if both are present.
- coreactants: The secondary reactants used to complete the surface reaction (e.g., "H2O", "O3", "NH3 plasma"). 
- purge_gas: The inert gas used to evacuate the chamber between reactant pulses (e.g., "Ar", "N2").
- carrier_gas: The inert gas used to transport the precursor into the chamber. (Note: This is often the same as the purge gas, but only list it here if explicitly described as a carrier).

Rules:
1. If a specific input type is not explicitly mentioned in the text, return an empty list []. Do not guess or infer based on standard ALD processes.
2. If multiple precursors or co-reactants are used (e.g., in nanolaminates or doped films), list all of them.
3. Provide a single, exact quote from the text in the "evidence" field that justifies your selections.

Return the result STRICTLY as a valid JSON object matching the exact structure below. Do not include markdown formatting or conversational text.

{{
  "precursors": [],
  "coreactants": [],
  "purge_gas": [],
  "carrier_gas": [],
  "evidence": ""
}}

"""  + "\nText:\n```" + fulltext + "```"
    output = llm.invoke(prompt)
    return robust_json_parse(output, default={
        "precursors": [], "coreactants": [],
        "purge_gas": [], "carrier_gas": [], "evidence": None,
    })


# === Tool 4: Deposition conditions ============================================
def deposition_conditions_agent(fulltext: str, llm) -> Dict[str, Any]:  # FIX: correct return type
    prompt = _SYSTEM_PREAMBLE + f"""
Extract the specific Atomic Layer Deposition (ALD) process parameters from the provided text. 

Extraction Rules:
1. Numerical Values: Extract only the number for fields ending in "_s" (seconds) or "_C" (Celsius). If a range is provided (e.g., "250-300°C"), represent it as a string "250-300".
2. Timing Sequences: If the text provides a sequence like (0.1/5/0.1/5 s), map them correctly: 
   - precursor_pulse_time_s = 0.1
   - coreactant_pulse_time_s = 0.1
   - purge_time_s = 5
3. Reactor Type: Identify the specific tool or reactor geometry (e.g., "F-120", "Cambridge NanoTech Savannah", "Cross-flow", "Showerhead", "Spatial ALD").
4. Null Handling: If a parameter is not explicitly mentioned, return null for numerical fields and an empty string "" for text fields. Do not assume "standard" conditions.
5. Evidence: Provide a direct, unedited quote from the text that contains these parameters.

Return the result STRICTLY as a valid JSON object. Do not include markdown headers or conversational filler.

{{
 "deposition_temperature_C": null,
 "pressure": "",
 "precursor_pulse_time_s": null,
 "coreactant_pulse_time_s": null,
 "purge_time_s": null,
 "number_of_cycles": null,
 "reactor_type": "",
 "evidence": ""
}}

"""  + "\nText:\n```" + fulltext + "```"
    output = llm.invoke(prompt)
    return robust_json_parse(output, default={
        "deposition_temperature_C": None, "pressure": "",
        "precursor_pulse_time_s": None, "coreactant_pulse_time_s": None,
        "purge_time_s": None, "number_of_cycles": None,
        "reactor_type": "", "evidence": None,
    })


# === Tool 5: Reaction conditions ==============================================
def reaction_conditions_agent(fulltext: str, llm) -> Dict[str, Any]:  # FIX: correct return type
    prompt = _SYSTEM_PREAMBLE + f"""
Analyze the provided text to extract the Atomic Layer Deposition (ALD) surface reaction mechanism and its associated chemistry.

Extraction Guidelines:
1. reaction_equations: Extract all formal chemical equations representing the surface reactions. Use standard chemical notation (e.g., "Si-OH + Al(CH3)3 -> Si-O-Al(CH3)2 + CH4"). Represent surface species with an asterisk or "s" suffix if present in text (e.g., Al-OH*).
2. surface_mechanism_description: Provide a concise summary of the reaction steps as described by the authors (e.g., "Ligand exchange between TMA and surface hydroxyl groups followed by water pulse to regenerate OH sites").
3. intermediate_species: Identify specific transient chemical species formed on the surface during the pulse and purge cycles (e.g., "monomethyl aluminum", "surface hydroxyls", "formate species").
4. Evidence: Provide the exact excerpt from the text where these reactions or mechanisms are discussed.

Rules:
- If no formal equations are provided, return an empty list [] for "reaction_equations".
- Maintain the original chemical stoichiometry as written in the text.
- Do not include conversational filler or markdown outside the JSON block.

Return the result STRICTLY as a valid JSON object:

{{
 "reaction_equations": [],
 "surface_mechanism_description": "",
 "intermediate_species": [],
 "evidence": ""
}}

"""  + "\nText:\n```" + fulltext + "```"
    output = llm.invoke(prompt)
    return robust_json_parse(output, default={
        "reaction_equations": [], "surface_mechanism_description": "",
        "intermediate_species": [], "evidence": None,
    })


# === Tool 6: Substrate information ============================================
def substrate_information_agent(fulltext: str, llm) -> Dict[str, Any]:  # FIX: correct return type
    prompt = _SYSTEM_PREAMBLE + f"""
Analyze the provided text to extract information regarding the substrate used for the Atomic Layer Deposition (ALD) process.

Extraction Guidelines:
1. substrate_material: Identify the base material (e.g., "Silicon", "Glass", "Sapphire", "PET", "FTO-coated glass"). 
2. substrate_orientation: Extract the crystallographic orientation or Miller indices if provided (e.g., "<100>", "(111)", "c-plane"). Return an empty string if the material is amorphous or not specified.
3. pretreatment: Identify any cleaning or preparation steps performed BEFORE deposition (e.g., "RCA cleaning", "HF dip", "Acetone/IPA sonication", "Piranha etch", "Degassing at 400°C").
4. surface_functionalization: Identify any chemical treatments intended to modify surface groups for nucleation (e.g., "O2 plasma activation", "Silanization", "Hydroxylation", "Self-assembled monolayer (SAM) deposition").
5. evidence: Provide the exact, unedited quote from the text that describes the substrate and its preparation.

Rules:
- If a specific detail is not mentioned, return an empty string "".
- Do not include markdown headers or conversational text outside the JSON block.
- Focus specifically on the starting surface, not the final deposited film.

Return the result STRICTLY as a valid JSON object:

{{
 "substrate_material": "",
 "substrate_orientation": "",
 "pretreatment": "",
 "surface_functionalization": "",
 "evidence": ""
}}

"""  + "\nText:\n```" + fulltext + "```"
    output = llm.invoke(prompt)
    return robust_json_parse(output, default={
        "substrate_material": "", "substrate_orientation": "",
        "pretreatment": "", "surface_functionalization": "", "evidence": None,
    })


# === Tool 7: Film properties ==================================================
def film_properties_agent(fulltext: str, llm) -> Dict[str, Any]:  # FIX: correct return type
    prompt = _SYSTEM_PREAMBLE + f"""
Analyze the provided text to extract the physical and structural properties of the resulting ALD film.

Extraction Guidelines:
1. film_thickness_nm: Extract the measured thickness in nanometers. If multiple samples exist, prioritize the primary sample or provide a range (e.g., "10.5" or "5-50").
2. density_g_cm3: Extract the mass density in grams per cubic centimeter (g/cm³). This is often determined via XRR.
3. refractive_index: Extract the refractive index (n). Specify the wavelength if mentioned (e.g., "1.64 at 633 nm"). Return null if not mentioned.
4. surface_roughness_nm: Extract the Root Mean Square (RMS) or average roughness (Ra) in nanometers.
5. crystal_phase: Identify the crystallographic structure (e.g., "Amorphous", "Anatase", "Rutile", "Alpha-phase", "Polycrystalline").
6. evidence: Provide the exact, unedited quote from the text that mentions these characterization results.

Rules:
- If a value is not mentioned, return null for numerical fields and "" for the crystal_phase.
- Do not confuse "Surface Roughness" of the substrate with the "Surface Roughness" of the deposited film.
- Ensure refractive index is not confused with the extinction coefficient (k).

Return the result STRICTLY as a valid JSON object:

{{
 "film_thickness_nm": null,
 "density_g_cm3": null,
 "refractive_index": null,
 "surface_roughness_nm": null,
 "crystal_phase": "",
 "evidence": ""
}}

"""  + "\nText:\n```" + fulltext + "```"
    output = llm.invoke(prompt)
    return robust_json_parse(output, default={
        "film_thickness_nm": None, "density_g_cm3": None,
        "refractive_index": None, "surface_roughness_nm": None,
        "crystal_phase": "", "evidence": None,
    })


# === Tool 8: Characterization =================================================
def characterization_agent(fulltext: str, llm) -> Dict[str, Any]:  # FIX: correct return type
    prompt = _SYSTEM_PREAMBLE + f"""
Analyze the provided text to identify the characterization techniques used to analyze the ALD process or the resulting films.

Extraction Guidelines:
1. characterization_methods: Create a list of all techniques explicitly used in the study. Include both the abbreviation and the full name if provided (e.g., "X-ray Photoelectron Spectroscopy (XPS)", "Atomic Force Microscopy (AFM)"). 
2. Capture a broad range of techniques, including but not limited to:
   - Structural: XRD, TEM, SEM, SAED.
   - Chemical/Compositional: XPS, SIMS, EDX/EDS, RBS, FTIR.
   - Physical/Optical: Ellipsometry, XRR, AFM, Profilometry.
   - Electrical: Hall Effect, Four-point probe, C-V measurement.
3. Evidence: Provide the exact, unedited quote(s) from the text that list or describe the use of these characterization tools.

Rules:
- Only include techniques actually performed in the study. Ignore references to techniques used in cited works.
- If no techniques are identified, return an empty list [].
- Do not include markdown headers or conversational text outside the JSON block.

Return the result STRICTLY as a valid JSON object:

{{
 "characterization_methods": [],
 "evidence": ""
}}

"""  + "\nText:\n```" + fulltext + "```"
    output = llm.invoke(prompt)
    return robust_json_parse(output, default={
        "characterization_methods": [], "evidence": None,
    })
