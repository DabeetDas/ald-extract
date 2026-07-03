from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "agentic_information_extraction_ollama"))

from validation import (  # noqa: E402
    AGENT_NAMES,
    AGENT_SCHEMAS,
    evidence_is_in_text,
    has_meaningful_content,
    normalize_text,
)


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SCHEMA_ISSUE_CODES = {
    "missing_file",
    "json_parse_error",
    "invalid_top_level_type",
    "empty_object",
    "missing_key",
    "wrong_type",
}

IGNORED_HALLUCINATION_PATH_SUFFIXES = {
    ".evidence",
}


def parse_args(
    *,
    default_extracted_dir: Path = REPO_ROOT / "qwen_extracted_info",
    default_report_prefix: str = "qwen",
    default_model_label: str = "Qwen",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ALD extraction outputs for schema validity, evidence grounding, and text-support proxy metrics."
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=default_extracted_dir,
        help="Directory containing extracted paper folders.",
    )
    parser.add_argument(
        "--source-roots",
        type=Path,
        nargs="+",
        default=[REPO_ROOT / "Data", REPO_ROOT / "Data_Unpaywall_OA"],
        help="Raw paper roots containing <paper_id>/content.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "evaluation",
        help="Directory where evaluation outputs are written.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar.",
    )
    parser.add_argument(
        "--report-prefix",
        default=default_report_prefix,
        help="Filename prefix for generated evaluation outputs.",
    )
    parser.add_argument(
        "--model-label",
        default=default_model_label,
        help="Human-readable model label used in the Markdown report.",
    )
    return parser.parse_args()


def expected_type_name(expected: Any) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    if isinstance(expected, dict):
        return "object"
    return expected.__name__


def matches_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(value, dict)
    if isinstance(expected, tuple):
        return isinstance(value, expected)
    return isinstance(value, expected)


def collect_schema_issues(
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
        if not matches_type(value, expected):
            issues.append(
                {
                    "code": "wrong_type",
                    "path": child_path,
                    "message": (
                        f"Expected {expected_type_name(expected)} for '{key}', "
                        f"got {type(value).__name__}."
                    ),
                }
            )
            continue

        if isinstance(expected, dict):
            collect_schema_issues(value, expected, path=child_path, issues=issues)


def load_json(path: Path) -> tuple[Any | None, dict[str, Any] | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, {
            "code": "json_parse_error",
            "path": str(path),
            "message": str(exc),
        }


def find_content_path(paper_id: str, source_roots: list[Path]) -> Path | None:
    for root in source_roots:
        candidate = root / paper_id / "content.txt"
        if candidate.exists():
            return candidate
    return None


def is_non_empty_scalar(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float, bool)):
        return True
    return False


def flatten_values(value: Any, *, path: str) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            values.extend(flatten_values(child, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            values.extend(flatten_values(child, path=f"{path}[{index}]"))
    elif is_non_empty_scalar(value):
        values.append((path, value))
    return values


def compact_text(text: str) -> str:
    return "".join(ch for ch in normalize_text(text) if ch.isalnum())


def value_is_text_supported(value: Any, fulltext: str, evidence: str | None) -> bool:
    text_value = str(value).strip()
    if not text_value:
        return False

    normalized_value = normalize_text(text_value)
    normalized_fulltext = normalize_text(fulltext)
    if normalized_value and normalized_value in normalized_fulltext:
        return True

    compact_value = compact_text(text_value)
    compact_fulltext = compact_text(fulltext)
    if len(compact_value) >= 3 and compact_value in compact_fulltext:
        return True

    if isinstance(evidence, str) and evidence.strip():
        normalized_evidence = normalize_text(evidence)
        if normalized_value and normalized_value in normalized_evidence:
            return True

        compact_evidence = compact_text(evidence)
        if len(compact_value) >= 3 and compact_value in compact_evidence:
            return True

    return False


def percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator * 100, 4)


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def progress_iter(items: list[Path], *, enabled: bool, model_label: str):
    if enabled and tqdm is not None:
        return tqdm(items, desc=f"Evaluating {model_label} papers", unit="paper")
    return items


def main(
    *,
    default_extracted_dir: Path = REPO_ROOT / "qwen_extracted_info",
    default_report_prefix: str = "qwen",
    default_model_label: str = "Qwen",
) -> None:
    args = parse_args(
        default_extracted_dir=default_extracted_dir,
        default_report_prefix=default_report_prefix,
        default_model_label=default_model_label,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paper_dirs = sorted(
        path
        for path in args.extracted_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )

    schema_issue_counter: Counter[str] = Counter()
    schema_issue_by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    evidence_issue_counter: Counter[str] = Counter()
    hallucination_by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    per_paper_rows: list[dict[str, Any]] = []
    per_value_rows: list[dict[str, Any]] = []
    detailed_issues: dict[str, dict[str, Any]] = {}

    expected_outputs = len(paper_dirs) * len(AGENT_NAMES)
    valid_schema_outputs = 0
    parsed_outputs = 0
    raw_text_available_papers = 0
    missing_content_papers = 0

    non_empty_extractions = 0
    grounded_extractions = 0
    missing_or_empty_evidence = 0
    evidence_not_found = 0

    extracted_values = 0
    supported_values = 0
    unsupported_values = 0

    for paper_dir in progress_iter(
        paper_dirs,
        enabled=not args.no_progress,
        model_label=args.model_label,
    ):
        paper_id = paper_dir.name
        content_path = find_content_path(paper_id, args.source_roots)
        fulltext = ""
        if content_path is None:
            missing_content_papers += 1
        else:
            raw_text_available_papers += 1
            fulltext = content_path.read_text(encoding="utf-8", errors="ignore")

        paper_schema_valid = 0
        paper_schema_invalid = 0
        paper_non_empty = 0
        paper_grounded = 0
        paper_values = 0
        paper_unsupported_values = 0
        paper_issues: dict[str, Any] = {}

        for agent_name in AGENT_NAMES:
            json_path = paper_dir / f"{agent_name}.json"
            if not json_path.exists():
                issue = {
                    "code": "missing_file",
                    "path": str(json_path),
                    "message": "Expected output file is missing.",
                }
                schema_issue_counter[issue["code"]] += 1
                schema_issue_by_agent[agent_name][issue["code"]] += 1
                paper_schema_invalid += 1
                paper_issues.setdefault(agent_name, []).append(issue)
                continue

            data, parse_issue = load_json(json_path)
            if parse_issue is not None:
                schema_issue_counter[parse_issue["code"]] += 1
                schema_issue_by_agent[agent_name][parse_issue["code"]] += 1
                paper_schema_invalid += 1
                paper_issues.setdefault(agent_name, []).append(parse_issue)
                continue

            parsed_outputs += 1
            schema_issues: list[dict[str, Any]] = []
            collect_schema_issues(
                data,
                AGENT_SCHEMAS[agent_name],
                path=agent_name,
                issues=schema_issues,
            )
            if schema_issues:
                paper_schema_invalid += 1
                for issue in schema_issues:
                    schema_issue_counter[issue["code"]] += 1
                    schema_issue_by_agent[agent_name][issue["code"]] += 1
                paper_issues.setdefault(agent_name, []).extend(schema_issues)
            else:
                valid_schema_outputs += 1
                paper_schema_valid += 1

            if not isinstance(data, dict):
                continue

            extracted_content = {key: value for key, value in data.items() if key != "evidence"}
            has_content = has_meaningful_content(extracted_content)
            evidence = data.get("evidence")

            if content_path is not None and has_content:
                non_empty_extractions += 1
                paper_non_empty += 1

                if not isinstance(evidence, str) or not evidence.strip():
                    missing_or_empty_evidence += 1
                    evidence_issue_counter["missing_or_empty_evidence"] += 1
                elif evidence_is_in_text(fulltext, evidence):
                    grounded_extractions += 1
                    paper_grounded += 1
                else:
                    evidence_not_found += 1
                    evidence_issue_counter["evidence_not_found"] += 1

            if content_path is not None:
                for value_path, value in flatten_values(data, path=agent_name):
                    if any(value_path.endswith(suffix) for suffix in IGNORED_HALLUCINATION_PATH_SUFFIXES):
                        continue
                    extracted_values += 1
                    paper_values += 1
                    hallucination_by_agent[agent_name]["extracted_values"] += 1

                    is_supported = value_is_text_supported(value, fulltext, evidence)
                    if is_supported:
                        supported_values += 1
                        hallucination_by_agent[agent_name]["supported_values"] += 1
                    else:
                        unsupported_values += 1
                        paper_unsupported_values += 1
                        hallucination_by_agent[agent_name]["unsupported_values"] += 1
                        per_value_rows.append(
                            {
                                "paper_id": paper_id,
                                "agent": agent_name,
                                "path": value_path,
                                "value": value,
                                "evidence": evidence if isinstance(evidence, str) else "",
                            }
                        )

        if paper_issues:
            detailed_issues[paper_id] = paper_issues

        per_paper_rows.append(
            {
                "paper_id": paper_id,
                "content_path": str(content_path.relative_to(REPO_ROOT)) if content_path else "",
                "schema_valid_outputs": paper_schema_valid,
                "schema_invalid_outputs": paper_schema_invalid,
                "non_empty_extractions_evaluable": paper_non_empty,
                "grounded_extractions": paper_grounded,
                "extracted_values_evaluable": paper_values,
                "unsupported_values_proxy": paper_unsupported_values,
            }
        )

    schema_validity_rate = percent(valid_schema_outputs, expected_outputs)
    evidence_grounding_rate = percent(grounded_extractions, non_empty_extractions)
    hallucination_proxy_rate = percent(unsupported_values, extracted_values)

    by_agent = {}
    for agent_name in AGENT_NAMES:
        agent_counts = hallucination_by_agent[agent_name]
        by_agent[agent_name] = {
            "schema_issues": dict(schema_issue_by_agent[agent_name]),
            "extracted_values_evaluable": agent_counts["extracted_values"],
            "supported_values_proxy": agent_counts["supported_values"],
            "unsupported_values_proxy": agent_counts["unsupported_values"],
            "hallucination_proxy_rate_percent": percent(
                agent_counts["unsupported_values"],
                agent_counts["extracted_values"],
            ),
        }

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "extracted_dir": str(args.extracted_dir.relative_to(REPO_ROOT)),
            "source_roots": [
                str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
                for path in args.source_roots
            ],
        },
        "scope": {
            "paper_folders": len(paper_dirs),
            "expected_agent_outputs": expected_outputs,
            "parsed_json_outputs": parsed_outputs,
            "raw_text_available_papers": raw_text_available_papers,
            "missing_content_papers": missing_content_papers,
        },
        "schema_validity": {
            "valid_outputs": valid_schema_outputs,
            "total_outputs": expected_outputs,
            "schema_validity_rate_percent": schema_validity_rate,
            "failure_counts_by_code": dict(schema_issue_counter),
        },
        "evidence_grounding": {
            "grounded_extractions": grounded_extractions,
            "non_empty_extractions_evaluable": non_empty_extractions,
            "evidence_grounding_rate_percent": evidence_grounding_rate,
            "missing_or_empty_evidence": missing_or_empty_evidence,
            "evidence_not_found": evidence_not_found,
        },
        "hallucination_proxy": {
            "description": (
                "Automated proxy, not a true entailment score: non-evidence scalar/list values are "
                "unsupported when their normalized text does not appear in the source paper or grounded evidence."
            ),
            "unsupported_values": unsupported_values,
            "extracted_values_evaluable": extracted_values,
            "supported_values": supported_values,
            "hallucination_proxy_rate_percent": hallucination_proxy_rate,
        },
        "by_agent": by_agent,
    }

    metrics_path = args.output_dir / f"{args.report_prefix}_evaluation_metrics.json"
    issues_path = args.output_dir / f"{args.report_prefix}_schema_and_grounding_issues.json"
    per_paper_path = args.output_dir / f"{args.report_prefix}_per_paper_metrics.csv"
    unsupported_path = args.output_dir / f"{args.report_prefix}_unsupported_values_proxy.csv"
    report_path = args.output_dir / f"{args.report_prefix}_evaluation_report.md"

    write_json(metrics_path, metrics)
    write_json(issues_path, detailed_issues)

    with per_paper_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_paper_rows[0].keys()) if per_paper_rows else [])
        writer.writeheader()
        writer.writerows(per_paper_rows)

    with unsupported_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["paper_id", "agent", "path", "value", "evidence"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_value_rows)

    markdown = f"""# {args.model_label} Extraction Evaluation

Generated: {metrics["generated_at"]}

## Inputs

- Extracted outputs: `{metrics["inputs"]["extracted_dir"]}`
- Source roots: {", ".join(f"`{root}`" for root in metrics["inputs"]["source_roots"])}
- Paper folders evaluated: {len(paper_dirs):,}
- Raw text available: {raw_text_available_papers:,}
- Missing `content.txt`: {missing_content_papers:,}

## Requested Metrics

| Metric | Numerator | Denominator | Rate |
|---|---:|---:|---:|
| Schema validity rate | {valid_schema_outputs:,} | {expected_outputs:,} | {format_percent(schema_validity_rate)} |
| Evidence grounding rate | {grounded_extractions:,} | {non_empty_extractions:,} | {format_percent(evidence_grounding_rate)} |
| Hallucination proxy rate | {unsupported_values:,} | {extracted_values:,} | {format_percent(hallucination_proxy_rate)} |

## Schema Failure Counts

| Issue code | Count |
|---|---:|
"""

    for code, count in schema_issue_counter.most_common():
        markdown += f"| `{code}` | {count:,} |\n"

    markdown += f"""
## Evidence Grounding Counts

| Issue | Count |
|---|---:|
| Grounded extractions | {grounded_extractions:,} |
| Missing or empty evidence | {missing_or_empty_evidence:,} |
| Evidence not found in source text | {evidence_not_found:,} |

## Hallucination Proxy Note

The hallucination number is an automated text-support proxy, not a manual entailment score. It marks a value as unsupported when the extracted non-evidence value does not appear, after normalization, in the raw paper text or in the extraction evidence. This can over-count unsupported values for inferred labels such as material class, normalized chemical formulas, or paraphrased summaries.

## Output Files

- `{metrics_path.name}`
- `{per_paper_path.name}`
- `{unsupported_path.name}`
- `{issues_path.name}`
"""

    report_path.write_text(markdown, encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Schema validity rate: {format_percent(schema_validity_rate)}")
    print(f"Evidence grounding rate: {format_percent(evidence_grounding_rate)}")
    print(f"Hallucination proxy rate: {format_percent(hallucination_proxy_rate)}")


if __name__ == "__main__":
    main()
