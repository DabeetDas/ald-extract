from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

AGENT_NAMES = (
    "summary",
    "target_material",
    "precursor_coreactant",
    "deposition_conditions",
    "reaction_conditions",
    "substrate_info",
    "film_properties",
    "characterization",
)

DEFAULT_DATASETS = {
    "qwen": REPO_ROOT / "qwen_extracted_info",
    "llama32_3b": REPO_ROOT / "extracted_data",
}


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate substring evidence grounding scores for extracted ALD JSON outputs."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Dataset to evaluate. Can be passed multiple times. "
            "Default: qwen=qwen_extracted_info and llama32_3b=extracted_data."
        ),
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
        default=REPO_ROOT / "evaluations",
        help="Directory where output files are written.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def resolve_datasets(dataset_args: list[str]) -> dict[str, Path]:
    if not dataset_args:
        return DEFAULT_DATASETS

    datasets: dict[str, Path] = {}
    for item in dataset_args:
        if "=" not in item:
            raise SystemExit(f"Invalid --dataset value {item!r}; use LABEL=PATH.")
        label, raw_path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"Invalid --dataset value {item!r}; label is empty.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        datasets[label] = path
    return datasets


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def evidence_is_in_text(
    evidence: str,
    *,
    normalized_fulltext: str,
    compact_fulltext: str,
) -> bool:
    if not isinstance(evidence, str) or not evidence.strip():
        return False

    normalized_evidence = normalize_text(evidence)
    if normalized_evidence in normalized_fulltext:
        return True

    compact_evidence = compact_text(evidence)
    return len(compact_evidence) >= 24 and compact_evidence in compact_fulltext


def is_paraphrase_evidence(evidence: str) -> bool:
    if not isinstance(evidence, str):
        return False
    lower_ev = evidence.strip().lower()
    paraphrase_prefixes = [
        "the text mentions",
        "the paper mentions",
        "the authors mention",
        "the text describes",
        "the paper describes",
        "the authors describe",
        "the text reports",
        "the paper reports",
        "the authors report",
        "the text states",
        "the paper states",
        "the authors state",
        "this text mentions",
        "this paper mentions"
    ]
    return any(lower_ev.startswith(prefix) for prefix in paraphrase_prefixes)


def sentence_grounding_rate(
    evidence: str,
    *,
    normalized_fulltext: str,
    compact_fulltext: str,
) -> float | None:
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    sentences = re.split(r'(?<=[.!?])\s+', evidence.strip())
    evaluable_sentences = [s.strip() for s in sentences if len(s.strip()) >= 20]
    if not evaluable_sentences:
        return None
    grounded_count = 0
    for s in evaluable_sentences:
        if evidence_is_in_text(
            s,
            normalized_fulltext=normalized_fulltext,
            compact_fulltext=compact_fulltext,
        ):
            grounded_count += 1
    return grounded_count / len(evaluable_sentences)


def has_meaningful_content(value: Any) -> bool:
    if isinstance(value, dict):
        return any(has_meaningful_content(child) for child in value.values())
    if isinstance(value, list):
        return any(has_meaningful_content(item) for item in value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:
        return None, str(exc)


def find_content_path(paper_id: str, source_roots: list[Path]) -> Path | None:
    for root in source_roots:
        candidate = root / paper_id / "content.txt"
        if candidate.exists():
            return candidate
    return None


def progress(items: list[Path], *, label: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(items, desc=f"{label}: evidence grounding", unit="paper")
    return items


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def evaluate_dataset(
    *,
    label: str,
    extracted_dir: Path,
    source_roots: list[Path],
    show_progress: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not extracted_dir.exists():
        raise SystemExit(f"Dataset path does not exist for {label}: {extracted_dir}")

    paper_dirs = sorted(
        path for path in extracted_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
    )

    counts: Counter[str] = Counter()
    agent_counts: dict[str, Counter[str]] = defaultdict(Counter)
    agent_sentence_rates: dict[str, list[float]] = defaultdict(list)
    dataset_sentence_rates: list[float] = []
    per_paper_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    for paper_dir in progress(paper_dirs, label=label, enabled=show_progress):
        paper_id = paper_dir.name
        content_path = find_content_path(paper_id, source_roots)

        counts["paper_folders"] += 1
        if content_path is None:
            counts["missing_content_papers"] += 1
            normalized_fulltext = ""
            compact_fulltext = ""
        else:
            counts["raw_text_available_papers"] += 1
            fulltext = content_path.read_text(encoding="utf-8", errors="ignore")
            normalized_fulltext = normalize_text(fulltext)
            compact_fulltext = compact_text(fulltext)

        paper_counts: Counter[str] = Counter()

        for agent_name in AGENT_NAMES:
            json_path = paper_dir / f"{agent_name}.json"
            counts["expected_outputs"] += 1
            agent_counts[agent_name]["expected_outputs"] += 1

            if not json_path.exists():
                counts["missing_json_files"] += 1
                agent_counts[agent_name]["missing_json_files"] += 1
                issue_rows.append(
                    {
                        "dataset": label,
                        "paper_id": paper_id,
                        "agent": agent_name,
                        "issue_code": "missing_json_file",
                        "json_path": safe_relative(json_path),
                        "content_path": safe_relative(content_path) if content_path else "",
                        "message": "Expected extraction file is missing.",
                        "evidence_preview": "",
                    }
                )
                continue

            data, error = load_json(json_path)
            if error is not None:
                counts["json_parse_errors"] += 1
                agent_counts[agent_name]["json_parse_errors"] += 1
                issue_rows.append(
                    {
                        "dataset": label,
                        "paper_id": paper_id,
                        "agent": agent_name,
                        "issue_code": "json_parse_error",
                        "json_path": safe_relative(json_path),
                        "content_path": safe_relative(content_path) if content_path else "",
                        "message": error,
                        "evidence_preview": "",
                    }
                )
                continue

            counts["parsed_outputs"] += 1
            agent_counts[agent_name]["parsed_outputs"] += 1

            if not isinstance(data, dict):
                counts["non_object_outputs"] += 1
                agent_counts[agent_name]["non_object_outputs"] += 1
                issue_rows.append(
                    {
                        "dataset": label,
                        "paper_id": paper_id,
                        "agent": agent_name,
                        "issue_code": "non_object_output",
                        "json_path": safe_relative(json_path),
                        "content_path": safe_relative(content_path) if content_path else "",
                        "message": f"Expected JSON object, got {type(data).__name__}.",
                        "evidence_preview": "",
                    }
                )
                continue

            extracted_content = {key: value for key, value in data.items() if key != "evidence"}
            if not has_meaningful_content(extracted_content):
                counts["empty_extractions"] += 1
                agent_counts[agent_name]["empty_extractions"] += 1
                paper_counts["empty_extractions"] += 1
                continue

            counts["non_empty_extractions"] += 1
            agent_counts[agent_name]["non_empty_extractions"] += 1
            paper_counts["non_empty_extractions"] += 1

            if content_path is None:
                counts["non_empty_extractions_without_source_text"] += 1
                agent_counts[agent_name]["non_empty_extractions_without_source_text"] += 1
                paper_counts["non_empty_extractions_without_source_text"] += 1
                continue

            counts["non_empty_extractions_evaluable"] += 1
            agent_counts[agent_name]["non_empty_extractions_evaluable"] += 1
            paper_counts["non_empty_extractions_evaluable"] += 1

            evidence = data.get("evidence")
            issue_code = None
            if not isinstance(evidence, str):
                issue_code = "missing_evidence"
                message = "Non-empty extraction is missing an evidence string."
            elif not evidence.strip():
                issue_code = "empty_evidence"
                message = "Non-empty extraction has an empty evidence string."
            elif evidence_is_in_text(
                evidence,
                normalized_fulltext=normalized_fulltext,
                compact_fulltext=compact_fulltext,
            ):
                counts["grounded_extractions"] += 1
                agent_counts[agent_name]["grounded_extractions"] += 1
                paper_counts["grounded_extractions"] += 1
                
                agent_sentence_rates[agent_name].append(1.0)
                dataset_sentence_rates.append(1.0)
                continue
            elif is_paraphrase_evidence(evidence):
                issue_code = "paraphrased_evidence"
                message = "Evidence appears to be an LLM-generated paraphrase rather than a verbatim excerpt."
            else:
                issue_code = "evidence_not_found"
                message = "Evidence does not appear in the source text after normalization."

            if isinstance(evidence, str) and evidence.strip():
                sent_rate = sentence_grounding_rate(
                    evidence,
                    normalized_fulltext=normalized_fulltext,
                    compact_fulltext=compact_fulltext,
                )
                if sent_rate is not None:
                    agent_sentence_rates[agent_name].append(sent_rate)
                    dataset_sentence_rates.append(sent_rate)
                    if sent_rate >= 0.5:
                        counts["sentence_grounded_count"] += 1
                        agent_counts[agent_name]["sentence_grounded_count"] += 1
                        paper_counts["sentence_grounded_count"] += 1

            if issue_code:
                counts[issue_code] += 1
                agent_counts[agent_name][issue_code] += 1
                paper_counts[issue_code] += 1
                issue_rows.append(
                    {
                        "dataset": label,
                        "paper_id": paper_id,
                        "agent": agent_name,
                        "issue_code": issue_code,
                        "json_path": safe_relative(json_path),
                        "content_path": safe_relative(content_path) if content_path else "",
                        "message": message,
                        "evidence_preview": evidence[:240] if isinstance(evidence, str) else "",
                    }
                )

        per_paper_rows.append(
            {
                "dataset": label,
                "paper_id": paper_id,
                "content_path": safe_relative(content_path) if content_path else "",
                "non_empty_extractions": paper_counts["non_empty_extractions"],
                "non_empty_extractions_evaluable": paper_counts["non_empty_extractions_evaluable"],
                "grounded_extractions": paper_counts["grounded_extractions"],
                "sentence_grounded_count": paper_counts["sentence_grounded_count"],
                "paraphrased_evidence": paper_counts["paraphrased_evidence"],
                "missing_evidence": paper_counts["missing_evidence"],
                "empty_evidence": paper_counts["empty_evidence"],
                "evidence_not_found": paper_counts["evidence_not_found"],
                "non_empty_extractions_without_source_text": paper_counts[
                    "non_empty_extractions_without_source_text"
                ],
            }
        )

    denominator = counts["non_empty_extractions_evaluable"]
    grounding_rate = None if denominator == 0 else counts["grounded_extractions"] / denominator
    
    def avg(lst: list[float]) -> float | None:
        return sum(lst) / len(lst) if lst else None
        
    dataset_sent_rate = avg(dataset_sentence_rates)

    summary = {
        "dataset": label,
        "extracted_dir": safe_relative(extracted_dir),
        "paper_folders": counts["paper_folders"],
        "raw_text_available_papers": counts["raw_text_available_papers"],
        "missing_content_papers": counts["missing_content_papers"],
        "expected_outputs": counts["expected_outputs"],
        "parsed_outputs": counts["parsed_outputs"],
        "missing_json_files": counts["missing_json_files"],
        "json_parse_errors": counts["json_parse_errors"],
        "non_object_outputs": counts["non_object_outputs"],
        "empty_extractions": counts["empty_extractions"],
        "non_empty_extractions": counts["non_empty_extractions"],
        "non_empty_extractions_evaluable": denominator,
        "grounded_extractions": counts["grounded_extractions"],
        "substring_evidence_grounding_rate": grounding_rate,
        "substring_evidence_grounding_rate_percent": None
        if grounding_rate is None
        else round(grounding_rate * 100, 4),
        "sentence_grounding_rate_percent": None if dataset_sent_rate is None else round(dataset_sent_rate * 100, 4),
        "paraphrased_evidence": counts["paraphrased_evidence"],
        "missing_evidence": counts["missing_evidence"],
        "empty_evidence": counts["empty_evidence"],
        "evidence_not_found": counts["evidence_not_found"],
        "non_empty_extractions_without_source_text": counts[
            "non_empty_extractions_without_source_text"
        ],
        "by_agent": {
            agent_name: {
                "expected_outputs": agent_counts[agent_name]["expected_outputs"],
                "parsed_outputs": agent_counts[agent_name]["parsed_outputs"],
                "non_empty_extractions_evaluable": agent_counts[agent_name][
                    "non_empty_extractions_evaluable"
                ],
                "grounded_extractions": agent_counts[agent_name]["grounded_extractions"],
                "substring_evidence_grounding_rate_percent": None
                if agent_counts[agent_name]["non_empty_extractions_evaluable"] == 0
                else round(
                    agent_counts[agent_name]["grounded_extractions"]
                    / agent_counts[agent_name]["non_empty_extractions_evaluable"]
                    * 100,
                    4,
                ),
                "sentence_grounding_rate_percent": None if not agent_sentence_rates[agent_name] else round(avg(agent_sentence_rates[agent_name]) * 100, 4),
                "paraphrased_evidence": agent_counts[agent_name]["paraphrased_evidence"],
                "missing_evidence": agent_counts[agent_name]["missing_evidence"],
                "empty_evidence": agent_counts[agent_name]["empty_evidence"],
                "evidence_not_found": agent_counts[agent_name]["evidence_not_found"],
                "non_empty_extractions_without_source_text": agent_counts[agent_name][
                    "non_empty_extractions_without_source_text"
                ],
            }
            for agent_name in AGENT_NAMES
        },
    }

    return summary, per_paper_rows, issue_rows


def write_outputs(
    *,
    output_dir: Path,
    summaries: list[dict[str, Any]],
    per_paper_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
    source_roots: list[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_payload = {
        "generated_at": generated_at,
        "metric": "substring_evidence_grounding_rate",
        "definition": (
            "grounded_extractions / non_empty_extractions_evaluable, where an extraction is grounded "
            "if its evidence string appears as a normalized substring of the matching content.txt."
        ),
        "source_roots": [safe_relative(path) for path in source_roots],
        "datasets": summaries,
    }

    with (output_dir / "evidence_grounding_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with (output_dir / "evidence_grounding_per_paper.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "dataset",
            "paper_id",
            "content_path",
            "non_empty_extractions",
            "non_empty_extractions_evaluable",
            "grounded_extractions",
            "sentence_grounded_count",
            "paraphrased_evidence",
            "missing_evidence",
            "empty_evidence",
            "evidence_not_found",
            "non_empty_extractions_without_source_text",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_paper_rows)

    with (output_dir / "evidence_grounding_issues.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "dataset",
            "paper_id",
            "agent",
            "issue_code",
            "json_path",
            "content_path",
            "message",
            "evidence_preview",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(issue_rows)

    report = [
        "# Substring Evidence Grounding Scores",
        "",
        f"Generated: {generated_at}",
        "",
        "Metric: `grounded_extractions / non_empty_extractions_evaluable`",
        "Sentence-level metric: average fraction of sentences per evidence string found in text",
        "",
        "| Dataset | Grounded | Evaluable non-empty | Subscore | Sent Score | Paraphrased | Missing ev | Empty ev | Match failed | Missing src |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for summary in summaries:
        rate = summary["substring_evidence_grounding_rate_percent"]
        rate_text = "n/a" if rate is None else f"{rate:.2f}%"
        sent_rate = summary.get("sentence_grounding_rate_percent")
        sent_rate_text = "n/a" if sent_rate is None else f"{sent_rate:.2f}%"
        report.append(
            "| {dataset} | {grounded:,} | {denom:,} | {rate} | {sent_rate} | {paraphrased:,} | {missing:,} | {empty:,} | {not_found:,} | {no_source:,} |".format(
                dataset=summary["dataset"],
                grounded=summary["grounded_extractions"],
                denom=summary["non_empty_extractions_evaluable"],
                rate=rate_text,
                sent_rate=sent_rate_text,
                paraphrased=summary.get("paraphrased_evidence", 0),
                missing=summary["missing_evidence"],
                empty=summary["empty_evidence"],
                not_found=summary["evidence_not_found"],
                no_source=summary["non_empty_extractions_without_source_text"],
            )
        )

    report.extend(
        [
            "",
            "Outputs:",
            "",
            "- `evidence_grounding_scores.json`",
            "- `evidence_grounding_per_paper.csv`",
            "- `evidence_grounding_issues.csv`",
            "",
            "This is a deterministic substring-grounding metric. It does not run an LLM and does not prove the extracted value is entailed by the evidence.",
        ]
    )

    (output_dir / "evidence_grounding_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    datasets = resolve_datasets(args.dataset)

    summaries: list[dict[str, Any]] = []
    per_paper_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    for label, extracted_dir in datasets.items():
        summary, paper_rows, dataset_issue_rows = evaluate_dataset(
            label=label,
            extracted_dir=extracted_dir,
            source_roots=args.source_roots,
            show_progress=not args.no_progress,
        )
        summaries.append(summary)
        per_paper_rows.extend(paper_rows)
        issue_rows.extend(dataset_issue_rows)

    write_outputs(
        output_dir=args.output_dir,
        summaries=summaries,
        per_paper_rows=per_paper_rows,
        issue_rows=issue_rows,
        source_roots=args.source_roots,
    )

    print(f"Wrote outputs to {args.output_dir}")
    for summary in summaries:
        rate = summary["substring_evidence_grounding_rate_percent"]
        rate_text = "n/a" if rate is None else f"{rate:.2f}%"
        sent_rate = summary.get("sentence_grounding_rate_percent")
        sent_rate_text = "n/a" if sent_rate is None else f"{sent_rate:.2f}%"
        print(
            f"{summary['dataset']}: {rate_text} substring grounding "
            f"({summary['grounded_extractions']:,}/"
            f"{summary['non_empty_extractions_evaluable']:,}) | {sent_rate_text} sentence grounding"
        )


if __name__ == "__main__":
    main()
