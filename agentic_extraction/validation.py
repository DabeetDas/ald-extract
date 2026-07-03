from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


EMPTY_SCALAR = (None, "")
NUMERIC_OR_STRING = (int, float, str, type(None))
STRING_OR_NONE = (str, type(None))

AGENT_DEFAULTS: dict[str, dict[str, Any]] = {
    "summary": {
        "target_material": None,
        "process_type": None,
        "main_precursors": [],
        "temperature_range": None,
        "summary": None,
        "evidence": None,
    },
    "target_material": {
        "target_material": {
            "chemical_formula": None,
            "material_name": None,
            "material_class": None,
        },
        "evidence": None,
    },
    "precursor_coreactant": {
        "precursors": [],
        "coreactants": [],
        "purge_gas": [],
        "carrier_gas": [],
        "evidence": None,
    },
    "deposition_conditions": {
        "deposition_temperature_C": None,
        "pressure": "",
        "precursor_pulse_time_s": None,
        "coreactant_pulse_time_s": None,
        "purge_time_s": None,
        "number_of_cycles": None,
        "reactor_type": "",
        "evidence": None,
    },
    "reaction_conditions": {
        "reaction_equations": [],
        "surface_mechanism_description": "",
        "intermediate_species": [],
        "evidence": None,
    },
    "substrate_info": {
        "substrate_material": "",
        "substrate_orientation": "",
        "pretreatment": "",
        "surface_functionalization": "",
        "evidence": None,
    },
    "film_properties": {
        "film_thickness_nm": None,
        "density_g_cm3": None,
        "refractive_index": None,
        "surface_roughness_nm": None,
        "crystal_phase": "",
        "evidence": None,
    },
    "characterization": {
        "characterization_methods": [],
        "evidence": None,
    },
}

AGENT_SCHEMAS: dict[str, dict[str, Any]] = {
    "summary": {
        "target_material": STRING_OR_NONE,
        "process_type": STRING_OR_NONE,
        "main_precursors": list,
        "temperature_range": STRING_OR_NONE,
        "summary": STRING_OR_NONE,
        "evidence": STRING_OR_NONE,
    },
    "target_material": {
        "target_material": {
            "chemical_formula": STRING_OR_NONE,
            "material_name": STRING_OR_NONE,
            "material_class": STRING_OR_NONE,
        },
        "evidence": STRING_OR_NONE,
    },
    "precursor_coreactant": {
        "precursors": list,
        "coreactants": list,
        "purge_gas": list,
        "carrier_gas": list,
        "evidence": STRING_OR_NONE,
    },
    "deposition_conditions": {
        "deposition_temperature_C": NUMERIC_OR_STRING,
        "pressure": STRING_OR_NONE,
        "precursor_pulse_time_s": NUMERIC_OR_STRING,
        "coreactant_pulse_time_s": NUMERIC_OR_STRING,
        "purge_time_s": NUMERIC_OR_STRING,
        "number_of_cycles": NUMERIC_OR_STRING,
        "reactor_type": STRING_OR_NONE,
        "evidence": STRING_OR_NONE,
    },
    "reaction_conditions": {
        "reaction_equations": list,
        "surface_mechanism_description": STRING_OR_NONE,
        "intermediate_species": list,
        "evidence": STRING_OR_NONE,
    },
    "substrate_info": {
        "substrate_material": STRING_OR_NONE,
        "substrate_orientation": STRING_OR_NONE,
        "pretreatment": STRING_OR_NONE,
        "surface_functionalization": STRING_OR_NONE,
        "evidence": STRING_OR_NONE,
    },
    "film_properties": {
        "film_thickness_nm": NUMERIC_OR_STRING,
        "density_g_cm3": NUMERIC_OR_STRING,
        "refractive_index": NUMERIC_OR_STRING,
        "surface_roughness_nm": NUMERIC_OR_STRING,
        "crystal_phase": STRING_OR_NONE,
        "evidence": STRING_OR_NONE,
    },
    "characterization": {
        "characterization_methods": list,
        "evidence": STRING_OR_NONE,
    },
}

AGENT_NAMES = tuple(AGENT_DEFAULTS.keys())
HARD_ISSUE_CODES = {
    "invalid_top_level_type",
    "empty_object",
}
SOFT_ISSUE_CODES = {
    "evidence_not_found",
}


def default_output(agent_name: str) -> dict[str, Any]:
    return json.loads(json.dumps(AGENT_DEFAULTS[agent_name]))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def evidence_is_in_text(fulltext: str, evidence: str) -> bool:
    if not isinstance(evidence, str) or not evidence.strip():
        return False

    normalized_evidence = normalize_text(evidence)
    normalized_fulltext = normalize_text(fulltext)
    if normalized_evidence in normalized_fulltext:
        return True

    compact_evidence = _compact_text(evidence)
    compact_fulltext = _compact_text(fulltext)
    return len(compact_evidence) >= 24 and compact_evidence in compact_fulltext


def has_meaningful_content(value: Any, *, ignore_keys: set[str] | None = None) -> bool:
    if ignore_keys is None:
        ignore_keys = set()

    if isinstance(value, dict):
        return any(
            has_meaningful_content(child, ignore_keys=ignore_keys)
            for key, child in value.items()
            if key not in ignore_keys
        )
    if isinstance(value, list):
        return any(has_meaningful_content(item, ignore_keys=ignore_keys) for item in value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _type_name(expected: Any) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    if isinstance(expected, dict):
        return "object"
    return expected.__name__


def _matches_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(value, dict)
    if isinstance(expected, tuple):
        return isinstance(value, expected)
    return isinstance(value, expected)


def _collect_schema_issues(
    data: Any,
    schema: dict[str, Any],
    *,
    path: str,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(data, dict):
        issues.append(
            {
                "code": "invalid_top_level_type",
                "path": path,
                "message": f"Expected object, got {type(data).__name__}.",
            }
        )
        return

    if not data:
        issues.append(
            {
                "code": "empty_object",
                "path": path,
                "message": "Output is an empty object.",
            }
        )
        return

    for key, expected in schema.items():
        child_path = f"{path}.{key}"
        if key not in data:
            issues.append(
                {
                    "code": "missing_key",
                    "path": child_path,
                    "message": f"Missing required key '{key}'.",
                }
            )
            continue

        value = data[key]
        if not _matches_type(value, expected):
            issues.append(
                {
                    "code": "wrong_type",
                    "path": child_path,
                    "message": (
                        f"Expected {_type_name(expected)} for '{key}', "
                        f"got {type(value).__name__}."
                    ),
                }
            )
            continue

        if isinstance(expected, dict):
            _collect_schema_issues(value, expected, path=child_path, issues=issues)


def validate_agent_output(agent_name: str, data: Any, fulltext: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    schema = AGENT_SCHEMAS[agent_name]

    _collect_schema_issues(data, schema, path=agent_name, issues=issues)

    if not isinstance(data, dict):
        return issues

    evidence = data.get("evidence")
    extracted_content = {k: v for k, v in data.items() if k != "evidence"}
    has_content = has_meaningful_content(extracted_content)

    if has_content and not isinstance(evidence, str):
        issues.append(
            {
                "code": "missing_evidence",
                "path": f"{agent_name}.evidence",
                "message": "Non-empty extraction is missing an evidence string.",
            }
        )
    elif has_content and isinstance(evidence, str) and not evidence.strip():
        issues.append(
            {
                "code": "empty_evidence",
                "path": f"{agent_name}.evidence",
                "message": "Non-empty extraction has an empty evidence field.",
            }
        )
    elif isinstance(evidence, str) and evidence.strip() and not evidence_is_in_text(fulltext, evidence):
        issues.append(
            {
                "code": "evidence_not_found",
                "path": f"{agent_name}.evidence",
                "message": "Evidence does not appear in the source text.",
            }
        )

    return issues


def validate_paper_outputs(
    results: dict[str, Any],
    fulltext: str,
) -> dict[str, list[dict[str, Any]]]:
    validation_issues: dict[str, list[dict[str, Any]]] = {}
    for agent_name, data in results.items():
        issues = validate_agent_output(agent_name, data, fulltext)
        if issues:
            validation_issues[agent_name] = issues
    return validation_issues


def classify_validation_issues(validation_issues: dict[str, list[dict[str, Any]]]) -> str:
    if not validation_issues:
        return "clean"

    for issues in validation_issues.values():
        for issue in issues:
            if issue["code"] in HARD_ISSUE_CODES:
                return "flagged"

    return "warning"


def load_json_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
