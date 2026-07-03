from __future__ import annotations

import json
import re
from typing import Any, Iterable

from agentic_rag_pipeline.schemas import ChatTurn
from agentic_rag_pipeline.settings import Settings


class LLMUnavailableError(RuntimeError):
    """Raised when Gemini-backed generation is not configured."""


def _format_history(conversation: Iterable[ChatTurn]) -> str:
    turns = list(conversation)[-6:]
    if not turns:
        return "No prior conversation."
    return "\n".join(f"{turn.role.upper()}: {turn.content.strip()}" for turn in turns)


def _clean_text_response(text: str) -> str:
    return text.strip()


def _extract_json_payload(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {raw[:280]!r}") from exc


class GeminiLLM:
    def __init__(self, settings: Settings):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMUnavailableError(
                "Gemini support requires the `google-genai` package. Install dependencies from "
                "`agentic_rag_pipeline/requirements.txt`."
            ) from exc

        if not settings.gemini_api_key:
            raise LLMUnavailableError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not set, so agent planning and synthesis are disabled."
            )

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.types = types
        self.model = settings.gemini_model

    def _generate(
        self,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int,
        temperature: float = 0.2,
    ) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                thinking_config=self.types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return _clean_text_response(response.text or "")

    def _generate_json(
        self,
        system_instruction: str,
        prompt: str,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        text = self._generate(
            system_instruction=system_instruction,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            temperature=0.1,
        )
        return _extract_json_payload(text)

    def generate_hyde_document(
        self,
        query: str,
        conversation: list[ChatTurn],
        scope_paper_id: str | None,
    ) -> str:
        scope_line = scope_paper_id or "global corpus"
        return self._generate(
            system_instruction=(
                "Write a hypothetical but retrieval-optimized paragraph for an ALD literature "
                "search system. The paragraph should sound like a dense paper excerpt that would "
                "likely answer the user's question. Mention materials, precursors, process type, "
                "deposition windows, and characterization details only when relevant. Do not "
                "mention that the text is hypothetical. Do not use bullets."
            ),
            prompt=(
                f"Corpus scope: {scope_line}\n"
                f"Conversation:\n{_format_history(conversation)}\n\n"
                f"Question:\n{query}"
            ),
            max_output_tokens=250,
        )

    def generate_plan(
        self,
        query: str,
        conversation: list[ChatTurn],
        scope_paper_id: str | None,
        available_tools: list[dict[str, Any]],
        max_steps: int,
    ) -> dict[str, Any]:
        tools_json = json.dumps(available_tools, indent=2)
        return self._generate_json(
            system_instruction=(
                "You are the Strategic Planner in a multi-step agentic RAG system. Your sole job is to "
                "decompose the user's query into a precise, dependency-aware execution plan. "
                "You NEVER answer the query yourself — only plan how to answer it. "
                "Separate analysis steps (reasoning, decomposition) from tool steps (data fetching). "
                "Always prefer Pinecone-backed rag_search for domain-specific evidence. "
                "Use wikipedia_lookup ONLY for foundational definitions or background context unavailable in Pinecone. "
                "Every tool step must have a clear, falsifiable expected_output. "
                "Plans must be minimal but complete — no redundant steps, no skipped dependencies."
            ),
            prompt=(
                "Return strict JSON with this schema:\n"
                "{\n"
                '  "planner_summary": "short summary",\n'
                '  "analysis": "what must be answered and what evidence is needed",\n'
                '  "synthesis_goal": "how the synthesizer should frame the final answer",\n'
                '  "validation_focus": ["focus item"],\n'
                '  "steps": [\n'
                "    {\n"
                '      "step_id": "A1 or E1",\n'
                '      "step_type": "analysis or tool",\n'
                '      "title": "short title",\n'
                '      "objective": "what this step does",\n'
                '      "depends_on": [],\n'
                '      "tool_name": "rag_search or wikipedia_lookup or null",\n'
                '      "arguments": {"query": "string", "scope_paper_id": null},\n'
                '      "expected_output": "what the step should yield"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Rules:\n"
                f"- Include at least one analysis step and at least one tool step unless the query is invalid.\n"
                f"- Use at most {max_steps} total steps.\n"
                "- Use `rag_search` for ALD-specific evidence gathering.\n"
                "- Use `wikipedia_lookup` only for broad supporting facts or definitions.\n"
                "- When a later tool depends on an earlier tool output, reference it using `#E1.field` style placeholders.\n"
                "- Do not include final answer text in the plan.\n\n"
                f"Scope: {scope_paper_id or 'global corpus'}\n"
                f"Conversation:\n{_format_history(conversation)}\n\n"
                f"User query:\n{query}\n\n"
                f"Available tools:\n{tools_json}"
            ),
            max_output_tokens=900,
        )

    def generate_answer(
        self,
        query: str,
        conversation: list[ChatTurn],
        scope_paper_id: str | None,
        plan_summary: str,
        execution_summary: str,
        context_blocks: str,
    ) -> str:
        scope_line = scope_paper_id or "global corpus"
        return self._generate(
            system_instruction=(
                "You are the Synthesizer Agent in an ALD (Atomic Layer Deposition) retrieval pipeline. "
                "You receive raw evidence blocks retrieved from two sources: a RAG search over ALD-specific "
                "documents, and a Wikipedia lookup. Your job is to synthesize these into a single, coherent, "
                "scientifically accurate answer to the user's query.\n\n"

                "## SYNTHESIS PRINCIPLES\n\n"

                "### 1. Source Hierarchy\n"
                "- RAG evidence is primary — always prefer RAG chunks for process-specific claims "
                "(GPC, ALD window, dose times, precursor chemistry, film properties)\n"
                "- Wikipedia is supplementary — use only for general chemistry context, definitions, "
                "or background where RAG evidence is silent\n"
                "- Never let Wikipedia override a RAG-sourced numerical or process-specific claim\n"
                "- If RAG and Wikipedia contradict, state both positions explicitly — do not silently resolve\n\n"

                "### 2. Strict Evidence Boundaries\n"
                "- Every claim you make must be traceable to an evidence block\n"
                "- Do not interpolate numerics: if evidence says >150°C, do not write 150–300°C\n"
                "- Do not infer causality beyond what the evidence states\n"
                "- Do not bridge two separate evidence facts with a causal statement unless the evidence itself makes that connection\n"
                "- If evidence is insufficient to answer part of the query, explicitly say so — do not fill gaps\n\n"

                "### 3. ALD-Specific Scientific Accuracy\n"
                "- Keep the answer centered on Atomic Layer Deposition (ALD) at all times\n"
                "- Do not pivot to CVD, LPCVD, PECVD, or other non-ALD processes unless the user explicitly asks for a comparison or the evidence directly frames a comparison\n"
                "- If non-ALD evidence appears, mention it only as a contrast and immediately return the focus to the ALD evidence answering the query\n"
                "- Temperature vs GPC: in ALD, GPC typically decreases at higher temperatures due to "
                "ligand desorption or increased reaction reversibility — do not imply a CVD-like positive correlation\n"
                "- Always distinguish thermal ALD from PE-ALD (plasma-enhanced) if both appear in evidence\n"
                "- Saturation behavior: dose and purge times must be described in the context of self-limiting reactions\n"
                "- Nucleation: distinguish substrate-inhibited vs substrate-enhanced nucleation if evidence supports it\n"
                "- Phase and stoichiometry: distinguish amorphous vs crystalline, and exact phases "
                "(e.g., TiO2 anatase vs rutile, Al2O3 vs AlOx)\n"
                "- Thin-film properties (bandgap, density, refractive index) differ from bulk — "
                "do not substitute bulk values from Wikipedia when RAG provides thin-film data\n\n"

                "### 4. Handling Conflicting Evidence\n"
                "- If two RAG chunks disagree on a value (e.g., different GPC at the same temperature), "
                "report both and note the discrepancy rather than picking one\n"
                "- If Wikipedia generalizes something that RAG specifies, use the RAG-specific value\n"
                "- Never manufacture consensus where the evidence is genuinely split\n\n"

                "### 5. Completeness Without Padding\n"
                "- Include all of the following when evidence supports them:\n"
                "  * ALD window definition and bounds\n"
                "  * Saturation curves and dose/purge behavior\n"
                "  * Nucleation delay or incubation cycles\n"
                "  * Film purity and impurity levels (C, H, N from XPS)\n"
                "  * Conformality and step coverage data\n"
                "  * Characterization methods used (XPS, TEM, XRR, ellipsometry, TOF-SIMS)\n"
                "  * Comparison between PE-ALD and thermal ALD where applicable\n"
                "- Do not add filler sentences to complete a section if the evidence does not support it\n\n"

                "### 6. Terminology\n"
                "- Use GPC (Å/cycle or nm/cycle) correctly — do not conflate with growth rate (nm/min)\n"
                "- Name precursors exactly as they appear in evidence: TMA, TDMAT, TEMAH, DEZ, TDMAH\n"
                "- Do not conflate oxidants: O3, H2O, O2 plasma, and N2O have distinct surface chemistries\n"
                "- Use 'self-limiting', 'conformal', and 'pinhole-free' only when directly supported by evidence\n\n"

                "## OUTPUT FORMAT\n"
                "- Output plain text only\n"
                "- Do not use markdown or markdown-like formatting anywhere\n"
                "- Do not use `#`, `##`, `###`, `*`, `**`, `***`, backticks, bullets made with markdown symbols, or horizontal rules\n"
                "- Do not wrap words in emphasis markers and do not emit headings with markdown prefixes\n"
                "- Organize the answer as short plain-text paragraphs with simple inline labels only when helpful, for example `Answer:`, `Evidence:`, or `Limitations:`\n"
                "- Lead with the strongest ALD-supported conclusion relevant to the query\n"
                "- Where Wikipedia context is used, it should support, not lead\n"
                "- End with a brief limitations note if any part of the query could not be fully answered from evidence"
),
            prompt=(
                f"Scope: {scope_line}\n"
                f"Conversation:\n{_format_history(conversation)}\n\n"
                f"Planner summary:\n{plan_summary}\n\n"
                f"Execution summary:\n{execution_summary}\n\n"
                f"Question:\n{query}\n\n"
                f"Evidence blocks:\n{context_blocks}"
            ),
            max_output_tokens=650,
        )

    def generate_validation_queries(
        self,
        query: str,
        draft_answer: str,
        validation_focus: list[str],
        limit: int,
    ) -> list[str]:
        payload = self._generate_json(
            system_instruction=(
                "You are the Validation Agent. Produce alternate retrieval queries for cross-checking "
                "the draft answer. Queries should target factual weak points, competing explanations, "
                "or missing evidence."
            ),
            prompt=(
                "Return strict JSON with schema:\n"
                '{ "queries": ["query one", "query two"] }\n\n'
                f"Original query:\n{query}\n\n"
                f"Draft answer:\n{draft_answer}\n\n"
                f"Validation focus:\n{json.dumps(validation_focus)}\n\n"
                f"Return at most {limit} queries."
            ),
            max_output_tokens=220,
        )
        queries = payload.get("queries", [])
        if not isinstance(queries, list):
            return []
        cleaned: list[str] = []
        for item in queries:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    cleaned.append(stripped)
        return cleaned[:limit]

    def validate_answer(
        self,
        query: str,
        draft_answer: str,
        validation_focus: list[str],
        context_blocks: str,
    ) -> dict[str, Any]:
        return self._generate_json(
           system_instruction=(
                "You are the final Validation Agent in an ALD (Atomic Layer Deposition) retrieval pipeline. "
                "You receive a synthesized answer and the raw evidence blocks it was built from. "
                "Your sole job is to audit whether the synthesized answer is faithful to the evidence.\n\n"

                "## WHAT TO CHECK\n\n"

                "### Factual Grounding\n"
                "Verify every claim traces to an evidence block. Flag:\n"
                "- Numeric values absent from evidence: GPC, ALD window bounds, dose/purge times, "
                "film thickness, dielectric constant, refractive index, impurity levels\n"
                "- Precursor chemistry errors: wrong ligands, incorrect oxidant pairings\n"
                "- Material property misattributions: wrong bandgap, phase, density\n"
                "- Bulk material properties (often from general sources) substituted for thin-film ALD values\n"
                "- Temperature-GPC errors: in ALD, GPC typically decreases at higher temperatures — "
                "flag if the answer implies otherwise without evidence\n\n"

                "### Logical Consistency\n"
                "Check internal coherence across the entire answer:\n"
                "- Self-contradictions between sections\n"
                "- Saturation or self-limiting claims inconsistent with cited dose/purge behavior\n"
                "- Growth mode descriptions that contradict the stated substrate or precursor\n"
                "- Causal bridges between two evidence facts that the evidence itself never makes\n"
                "- Comparative statements (faster, thinner, more conformal) that flip direction mid-answer\n\n"

                "### Cross-Verification\n"
                "Reconcile claims across all supplied evidence blocks:\n"
                "- False consensus: answer presents agreement where evidence blocks actually disagree\n"
                "- Cherry-picking: one block used while a contradicting block is silently ignored\n"
                "- Numeric interpolation: values not in any block but derived from them (e.g., evidence "
                "says >150°C but answer states 150–300°C)\n"
                "- Characterization conclusions (XPS, TEM, XRR, ellipsometry, TOF-SIMS) that exceed "
                "what the cited data actually supports\n"
                "- PE-ALD and thermal ALD results conflated when evidence distinguishes them\n\n"
                "### Domain Focus\n"
                "Flag domain drift such as:\n"
                "- The answer shifting from ALD to CVD, LPCVD, PECVD, or general deposition discussion without explicit user intent or supporting evidence\n"
                "- CVD terminology being used where the evidence is specifically about ALD\n"
                "- Comparative discussion that stops being ALD-centered\n\n"

                "## ISSUE LABELING\n"
                "Each entry in the issues list must be prefixed with one of:\n"
                "[HALLUCINATION] — claim has no basis in any evidence block\n"
                "[FACTUAL_ERROR] — claim contradicts evidence\n"
                "[LOGICAL_ERROR] — claim is internally inconsistent\n"
                "[EVIDENCE_CONFLICT] — evidence blocks disagree and answer did not surface this\n"
                "[OMISSION] — critical information present in evidence was excluded\n"
                "[TERMINOLOGY] — incorrect or conflated domain terminology\n\n"

                "## VERDICT BEHAVIOR\n"
                "- 'pass': answer is fully faithful to evidence; set revised_answer to empty string\n"
                "- 'warning': minor unsupported claims or omissions; revised_answer patches only the "
                "flagged sections while preserving all correct content and the original plain-text format\n"
                "- 'fail': answer is materially unsupported or misleading; revised_answer is a full "
                "rewrite strictly grounded in evidence, preserving the original plain-text format\n"
                "- Any revised_answer must stay ALD-centered and must not use markdown markers such as `#`, `*`, `**`, `***`, or backticks"
            ),
            prompt=(
                "Return strict JSON with schema:\n"
                "{\n"
                '  "factual_grounding": "pass|warning|fail",\n'
                '  "logical_consistency": "pass|warning|fail",\n'
                '  "cross_verification": "pass|warning|fail",\n'
                '  "issues": ["issue"],\n'
                '  "summary": "short validation summary",\n'
                '  "verdict": "pass|warning|fail",\n'
                '  "revised_answer": "corrected answer or empty string"\n'
                "}\n\n"
                f"User query:\n{query}\n\n"
                f"Validation focus:\n{json.dumps(validation_focus)}\n\n"
                f"Draft answer:\n{draft_answer}\n\n"
                f"Evidence blocks:\n{context_blocks}"
            ),
            max_output_tokens=800,
        )
