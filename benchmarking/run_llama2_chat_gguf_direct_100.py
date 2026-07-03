from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import sys
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXTRACTION_DIR = REPO_ROOT / "agentic_information_extraction_ollama"
sys.path.insert(0, str(EXTRACTION_DIR))

from tools import robust_json_parse  # noqa: E402
from validation import (  # noqa: E402
    AGENT_DEFAULTS,
    AGENT_NAMES,
    classify_validation_issues,
    validate_paper_outputs,
)


DEFAULT_MODEL_PATH = os.getenv("LLAMA2_MODEL_PATH", "/scratch/work/dabeetkd24/models/llamat-2-chat-q4_k_m.gguf")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarking" / "llama2_chat_gguf_direct_outputs"
DEFAULT_MODEL_CONTEXT_TOKENS = int(os.getenv("LLAMA2_MODEL_CONTEXT_TOKENS", "2048"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("LLAMA2_DIRECT_MAX_NEW_TOKENS", "1024"))
DEFAULT_CONTEXT_SAFETY_TOKENS = int(os.getenv("LLAMA2_CONTEXT_SAFETY_TOKENS", "256"))
DEFAULT_NUM_WORKERS = int(os.getenv("LLAMA2_DIRECT_WORKERS", "2"))
DEFAULT_GPUS = os.getenv("LLAMA2_DIRECT_GPUS", "0,1")
DEFAULT_RANDOM_STATE = int(os.getenv("LLAMA2_DIRECT_RANDOM_STATE", "42"))
DEFAULT_N_GPU_LAYERS = int(os.getenv("LLAMA2_N_GPU_LAYERS", "-1"))


# ---------------------------------------------------------------------------
# Structured-text prompt
# ---------------------------------------------------------------------------

# Each label maps to a (section, field_key) in the final JSON schema.
# Order matters: the model fills top-to-bottom.
TEMPLATE_LABELS: list[tuple[str, str, str]] = [
    # (label_in_prompt, section, field_key)
    ("Summary",                 "summary",              "summary"),
    ("Target Material",         "target_material",      "material_name"),
    ("Chemical Formula",        "target_material",      "chemical_formula"),
    ("Precursor Name",          "precursor_coreactant", "precursor_name"),
    ("Precursor Formula",       "precursor_coreactant", "precursor_formula"),
    ("Coreactant Name",         "precursor_coreactant", "coreactant_name"),
    ("Coreactant Formula",      "precursor_coreactant", "coreactant_formula"),
    ("Deposition Temperature",  "deposition_conditions","temperature"),
    ("Deposition Pressure",     "deposition_conditions","pressure"),
    ("Number of Cycles",        "deposition_conditions","num_cycles"),
    ("Pulse Time Precursor",    "reaction_conditions",  "pulse_time_precursor"),
    ("Purge Time Precursor",    "reaction_conditions",  "purge_time_precursor"),
    ("Pulse Time Coreactant",   "reaction_conditions",  "pulse_time_coreactant"),
    ("Purge Time Coreactant",   "reaction_conditions",  "purge_time_coreactant"),
    ("Growth Per Cycle",        "reaction_conditions",  "growth_per_cycle"),
    ("Substrate Material",      "substrate_info",       "substrate_material"),
    ("Substrate Preparation",   "substrate_info",       "substrate_preparation"),
    ("Film Thickness",          "film_properties",      "thickness"),
    ("Film Density",            "film_properties",      "density"),
    ("Film Resistivity",        "film_properties",      "resistivity"),
    ("Film Roughness",          "film_properties",      "roughness"),
    ("Refractive Index",        "film_properties",      "refractive_index"),
    ("Bandgap",                 "film_properties",      "bandgap"),
    ("Characterization Methods","characterization",     "methods"),
    ("Characterization Details","characterization",     "details"),
]

# Build a flat list of labels (used in the prompt and the parser)
_LABELS: list[str] = [label for label, _, _ in TEMPLATE_LABELS]

# Sentinel used in the prompt for empty fields
_NONE_VALUE = "N/A"


def build_structured_prompt(text: str) -> str:
    """
    Return a prompt that asks the model to fill a labelled key:value template.
    Keeping the schema as plain text (not JSON) dramatically improves reliability
    on smaller instruction-tuned models like LLaMaT-2-Chat.
    """
    label_lines = "\n".join(f"{label}: " for label in _LABELS)
    return (
        f"Extract ALD (Atomic Layer Deposition) information from the paper below.\n"
        f"Fill in every field using ONLY information found in the paper.\n"
        f"Write {_NONE_VALUE!r} when a field is not mentioned.\n"
        f"Do NOT add extra fields, explanations, or JSON.\n\n"
        f"Paper:\n{text}\n\n"
        f"Fill in this template:\n\n"
        f"{label_lines}"
    )


def _build_empty_prompt(text: str = "") -> str:
    """Used only for token-counting; text is intentionally empty."""
    return build_structured_prompt(text)


# ---------------------------------------------------------------------------
# Template parser → flat dict
# ---------------------------------------------------------------------------

def _normalise_label(raw: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", raw.strip().lower())


# Pre-build normalised lookup once.
_NORM_TO_LABEL: dict[str, str] = {_normalise_label(lbl): lbl for lbl in _LABELS}


def parse_template_output(raw: str) -> dict[str, str | None]:
    """
    Parse the model's filled-in template into a flat dict keyed by label.

    Handles:
    - Extra whitespace / line endings
    - Labels with or without trailing colon
    - Values that span multiple lines (until the next label line)
    - Model echoing the prompt header before the template
    """
    result: dict[str, str | None] = {lbl: None for lbl in _LABELS}

    # Find the first occurrence of any known label so we skip any preamble.
    lines = raw.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        candidate = re.split(r":", line, maxsplit=1)[0]
        if _normalise_label(candidate) in _NORM_TO_LABEL:
            start_idx = i
            break

    # Walk lines; accumulate multi-line values.
    current_label: str | None = None
    value_parts: list[str] = []

    def _flush() -> None:
        nonlocal current_label, value_parts
        if current_label is None:
            return
        raw_val = " ".join(value_parts).strip()
        # Treat sentinel values / empty / "none" / "unknown" as missing
        if not raw_val or raw_val.upper() in (_NONE_VALUE, "NONE", "UNKNOWN", "N/A", "-", "NULL"):
            result[current_label] = None
        else:
            result[current_label] = raw_val
        current_label = None
        value_parts = []

    for line in lines[start_idx:]:
        # Does the line start with a known label?
        colon_pos = line.find(":")
        if colon_pos != -1:
            candidate_raw = line[:colon_pos]
            norm = _normalise_label(candidate_raw)
            if norm in _NORM_TO_LABEL:
                _flush()
                current_label = _NORM_TO_LABEL[norm]
                remainder = line[colon_pos + 1:].strip()
                if remainder:
                    value_parts.append(remainder)
                continue
        # Continuation of the current value
        if current_label is not None:
            stripped = line.strip()
            if stripped:
                value_parts.append(stripped)

    _flush()
    return result


# ---------------------------------------------------------------------------
# Flat dict → nested JSON schema
# ---------------------------------------------------------------------------

def _val(flat: dict[str, str | None], label: str) -> str | None:
    return flat.get(label)


def flat_to_schema(flat: dict[str, str | None]) -> dict[str, Any]:
    """
    Map the parsed flat dict to the 8-section nested schema expected by the
    rest of the pipeline (AGENT_NAMES / AGENT_DEFAULTS).

    Unknown / missing fields are set to None so the downstream validators and
    the existing schema merging logic work unchanged.
    """
    return {
        "summary": {
            "summary": _val(flat, "Summary"),
        },
        "target_material": {
            "material_name":    _val(flat, "Target Material"),
            "chemical_formula": _val(flat, "Chemical Formula"),
        },
        "precursor_coreactant": {
            "precursor_name":     _val(flat, "Precursor Name"),
            "precursor_formula":  _val(flat, "Precursor Formula"),
            "coreactant_name":    _val(flat, "Coreactant Name"),
            "coreactant_formula": _val(flat, "Coreactant Formula"),
        },
        "deposition_conditions": {
            "temperature": _val(flat, "Deposition Temperature"),
            "pressure":    _val(flat, "Deposition Pressure"),
            "num_cycles":  _val(flat, "Number of Cycles"),
        },
        "reaction_conditions": {
            "pulse_time_precursor":   _val(flat, "Pulse Time Precursor"),
            "purge_time_precursor":   _val(flat, "Purge Time Precursor"),
            "pulse_time_coreactant":  _val(flat, "Pulse Time Coreactant"),
            "purge_time_coreactant":  _val(flat, "Purge Time Coreactant"),
            "growth_per_cycle":       _val(flat, "Growth Per Cycle"),
        },
        "substrate_info": {
            "substrate_material":    _val(flat, "Substrate Material"),
            "substrate_preparation": _val(flat, "Substrate Preparation"),
        },
        "film_properties": {
            "thickness":       _val(flat, "Film Thickness"),
            "density":         _val(flat, "Film Density"),
            "resistivity":     _val(flat, "Film Resistivity"),
            "roughness":       _val(flat, "Film Roughness"),
            "refractive_index":_val(flat, "Refractive Index"),
            "bandgap":         _val(flat, "Bandgap"),
        },
        "characterization": {
            "methods": _val(flat, "Characterization Methods"),
            "details": _val(flat, "Characterization Details"),
        },
    }


def combined_default() -> dict[str, Any]:
    """Return a deep copy of the pipeline default values for all 8 agents."""
    return json.loads(json.dumps({name: AGENT_DEFAULTS[name] for name in AGENT_NAMES}))


def merge_with_defaults(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-merge extracted values over the pipeline defaults.
    Extracted non-None values win; defaults fill any gaps not covered by
    flat_to_schema (e.g. fields added later to AGENT_DEFAULTS).
    """
    defaults = combined_default()
    for section, fields in schema.items():
        if section not in defaults:
            defaults[section] = {}
        if isinstance(fields, dict):
            for k, v in fields.items():
                if v is not None:
                    defaults[section][k] = v
        # If fields is a scalar (e.g. "summary" section is a dict with one key
        # but AGENT_DEFAULTS may store it differently), keep default behaviour.
    return defaults


# ---------------------------------------------------------------------------
# Response cleaning (strip chat template artifacts)
# ---------------------------------------------------------------------------

def strip_response(raw: str) -> str:
    """Remove chat-template leakage and return the model's fill-in section."""
    content = raw.strip()

    stop_markers = [r"<s>", r"</s>", r"\[INST\]", r"\[/INST\]", r"<<SYS>>", r"<</SYS>>"]
    pattern = "|".join(stop_markers)
    parts = re.split(pattern, content, maxsplit=1)
    content = parts[0].strip()

    return content


# ---------------------------------------------------------------------------
# Context-window truncation (unchanged logic, updated for new prompt builder)
# ---------------------------------------------------------------------------

def truncate_for_context(
    fulltext: str,
    llm: Any,
    *,
    model_context_tokens: int,
    max_new_tokens: int,
    safety_tokens: int,
) -> tuple[str, dict[str, Any]]:
    prompt_budget = model_context_tokens - max_new_tokens - safety_tokens
    if prompt_budget <= 512:
        raise ValueError(
            "Token budget is too small. Increase --model-context-tokens or reduce "
            "--max-new-tokens / --context-safety-tokens."
        )

    # Measure prompt overhead with an empty paper body.
    empty_prompt_formatted = (
        "[INST] <<SYS>>\nYou extract ALD data from scientific papers. "
        f"Fill in every field. Write {_NONE_VALUE!r} when not found.\n<</SYS>>\n\n"
        f"{_build_empty_prompt()}\n\n[/INST]"
    )
    overhead_tokens = len(llm.tokenize(empty_prompt_formatted.encode("utf-8")))
    text_budget = prompt_budget - overhead_tokens
    if text_budget <= 128:
        raise ValueError(
            f"Prompt overhead leaves only {text_budget} text tokens. "
            "Reduce prompt size or increase context budget."
        )

    text_tokens = llm.tokenize(fulltext.encode("utf-8"))
    original_text_tokens = len(text_tokens)
    was_truncated = original_text_tokens > text_budget
    if not was_truncated:
        return fulltext, {
            "was_truncated": False,
            "original_text_tokens": original_text_tokens,
            "used_text_tokens": original_text_tokens,
            "text_budget": text_budget,
            "prompt_budget": prompt_budget,
            "overhead_tokens": overhead_tokens,
        }

    head_tokens = int(text_budget * 0.78)
    tail_tokens = text_budget - head_tokens
    selected = text_tokens[:head_tokens] + text_tokens[-tail_tokens:]
    truncated = llm.detokenize(selected).decode("utf-8", errors="ignore")
    return truncated, {
        "was_truncated": True,
        "original_text_tokens": original_text_tokens,
        "used_text_tokens": len(selected),
        "text_budget": text_budget,
        "prompt_budget": prompt_budget,
        "overhead_tokens": overhead_tokens,
        "head_tokens": head_tokens,
        "tail_tokens": tail_tokens,
    }


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class DirectLlama2Extractor:
    def __init__(
        self,
        *,
        model_path: str,
        model_context_tokens: int,
        max_new_tokens: int,
        context_safety_tokens: int,
        n_gpu_layers: int,
    ):
        self.model_path = model_path
        self.model_context_tokens = model_context_tokens
        self.max_new_tokens = max_new_tokens
        self.context_safety_tokens = context_safety_tokens
        self.n_gpu_layers = n_gpu_layers
        self.llm = None

    def load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'llama_cpp'. Please run `pip install llama-cpp-python`."
            ) from exc

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.model_context_tokens,
            n_gpu_layers=self.n_gpu_layers,
            chat_format="llama-2",
            verbose=False,
        )

    def extract(
        self, fulltext: str
    ) -> tuple[dict[str, Any], str, str, dict[str, Any], dict[str, str | None]]:
        """
        Returns
        -------
        combined : dict
            Merged schema dict (8 sections), defaults filled in.
        raw : str
            Raw model output string.
        cleaned : str
            Output after stripping chat-template artifacts.
        token_report : dict
            Truncation / token-budget metadata.
        flat_parsed : dict
            The intermediate flat label→value dict (saved for debugging).
        """
        assert self.llm is not None
        truncated_text, token_report = truncate_for_context(
            fulltext,
            self.llm,
            model_context_tokens=self.model_context_tokens,
            max_new_tokens=self.max_new_tokens,
            safety_tokens=self.context_safety_tokens,
        )

        prompt = build_structured_prompt(truncated_text)

        full_prompt = f"""
        [INST]
        Text:

        Titanium phosphate thin films were deposited by a new plasma-enhanced atomic layer deposition
        process. The process consisted of sequential exposures to trimethyl phosphate (TMP, Me3 PO4 )
        plasma, O2 plasma and titanium isopropoxide (TTIP, Ti(OCH(CH3 )2 )4 ) vapor, and it was charac-
        terized by in-situ spectroscopic ellipsometry and ex-situ X-ray reflectometry. The growth linearity,
        growth per cycle (GPC), and density of the resulting thin films was investigated as a function of the
        pulse times and the substrate temperature. The conformality of the process was characterized
        by deposition on micropillars. At a substrate temperature of 300 ◦ C and using saturated pulse
        times, linear growth with a GPC of 0.66 nm/cycle and without nucleation delay was achieved.
        As-deposited films were amorphous, while crystalline TiP2 O7 was formed upon annealing in air
        or helium atmospheres. In lithium-ion test cells, the as-deposited films showed insertion and ex-
        traction of Li+ around a potential of 2.7 V vs. Li/Li+ . Charge/discharge measurements revealed
        a volumetric capacity of 330 mAh/cm3 , together with a good rate capability and minimal capacity
        fading.

        Answer exactly in this format:

        Material: <answer>
        Precursor: <answer>
        Temperature: <answer>
        GPC: <answer>
        Characterization: <answer>
        [/INST]
        """

        response = self.llm(
            full_prompt,
            max_tokens = self.max_new_tokens,
            temperature = 0.0
        )
        
        raw = response["choices"][0]["text"]

        print(raw)

        raw: str = response["choices"][0]["message"]["content"] or ""
        cleaned: str = strip_response(raw)

        flat_parsed: dict[str, str | None] = parse_template_output(cleaned)
        schema: dict[str, Any] = flat_to_schema(flat_parsed)
        combined: dict[str, Any] = merge_with_defaults(schema)

        return combined, raw, cleaned, token_report, flat_parsed


# ---------------------------------------------------------------------------
# File I/O helpers (unchanged)
# ---------------------------------------------------------------------------

def read_fulltext(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Per-paper processing
# ---------------------------------------------------------------------------

def process_paper(
    folder: Path, output_dir: Path, extractor: DirectLlama2Extractor
) -> dict[str, Any]:
    txt_path = folder / "content.txt"
    paper_out = output_dir / folder.name
    paper_out.mkdir(parents=True, exist_ok=True)

    if not txt_path.exists():
        return {
            "paper": folder.name,
            "status": "failed",
            "errors": {"paper": "Missing content.txt"},
        }

    fulltext = read_fulltext(txt_path)
    if not fulltext.strip():
        return {
            "paper": folder.name,
            "status": "failed",
            "errors": {"paper": "Empty content.txt"},
        }

    raw = ""
    cleaned = ""
    try:
        combined, raw, cleaned, token_report, flat_parsed = extractor.extract(fulltext)

        # Write per-agent JSON files (unchanged contract with rest of pipeline)
        results = {}
        for agent_name in AGENT_NAMES:
            results[agent_name] = combined.get(agent_name) or AGENT_DEFAULTS[agent_name]
            write_json(paper_out / f"{agent_name}.json", results[agent_name])

        # Debugging artefacts
        raw_out = paper_out / "_raw_model_outputs"
        raw_out.mkdir(exist_ok=True)
        (raw_out / "combined.txt").write_text(raw, encoding="utf-8")

        cleaned_out = paper_out / "_cleaned_model_outputs"
        cleaned_out.mkdir(exist_ok=True)
        (cleaned_out / "combined.template.txt").write_text(cleaned, encoding="utf-8")

        # Save flat parsed dict for transparency / debugging
        write_json(paper_out / "_flat_parsed.json", flat_parsed)

        write_json(paper_out / "token_report.json", token_report)

        # Validation (unchanged)
        validation_issues = validate_paper_outputs(results, fulltext)
        if validation_issues:
            write_json(paper_out / "validation_issues.json", validation_issues)
        else:
            validation_path = paper_out / "validation_issues.json"
            if validation_path.exists():
                validation_path.unlink()

        validation_status = classify_validation_issues(validation_issues)
        status = "completed"
        if validation_status == "flagged":
            status = "flagged"
        elif validation_status == "warning":
            status = "warning"

        return {
            "paper": folder.name,
            "status": status,
            "token_report": token_report,
            "validation_issue_count": sum(len(items) for items in validation_issues.values()),
        }

    except Exception as exc:
        error = {
            "paper": folder.name,
            "status": "failed",
            "errors": {"combined_extraction": f"{type(exc).__name__}: {exc}"},
            "traceback": traceback.format_exc(),
        }
        write_json(paper_out / "agent_errors.json", error["errors"])
        if raw:
            raw_out = paper_out / "_raw_model_outputs"
            raw_out.mkdir(exist_ok=True)
            (raw_out / "combined.txt").write_text(raw, encoding="utf-8")
        if cleaned:
            cleaned_out = paper_out / "_cleaned_model_outputs"
            cleaned_out.mkdir(exist_ok=True)
            (cleaned_out / "combined.template.txt").write_text(cleaned, encoding="utf-8")
        return error


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def worker_main(
    *,
    rank: int,
    gpu_id: str,
    folders: list[str],
    args_dict: dict[str, Any],
    result_queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    output_dir = Path(args_dict["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        extractor = DirectLlama2Extractor(
            model_path=args_dict["model_path"],
            model_context_tokens=args_dict["model_context_tokens"],
            max_new_tokens=args_dict["max_new_tokens"],
            context_safety_tokens=args_dict["context_safety_tokens"],
            n_gpu_layers=args_dict["n_gpu_layers"],
        )
        print(
            f"[worker {rank}] loading model from {args_dict['model_path']} "
            f"on CUDA_VISIBLE_DEVICES={gpu_id}",
            flush=True,
        )
        extractor.load()
        print(f"[worker {rank}] model loaded; processing {len(folders)} papers", flush=True)
    except Exception as exc:
        result_queue.put(
            {
                "worker": rank,
                "status": "worker_failed",
                "errors": {"model_load": f"{type(exc).__name__}: {exc}"},
                "traceback": traceback.format_exc(),
            }
        )
        return

    for folder_str in folders:
        folder = Path(folder_str)
        outcome = process_paper(folder, output_dir, extractor)
        outcome["worker"] = rank
        result_queue.put(outcome)

    result_queue.put({"worker": rank, "status": "worker_done"})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark direct Llama-2-chat via GGUF extraction using a structured text "
            "template instead of JSON prompting. Defaults to 100 papers."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=REPO_ROOT / "Data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--folders-file", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--gpus", type=str, default=DEFAULT_GPUS)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--model-context-tokens",
        type=int,
        default=DEFAULT_MODEL_CONTEXT_TOKENS,
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--context-safety-tokens", type=int, default=DEFAULT_CONTEXT_SAFETY_TOKENS)
    parser.add_argument("--n-gpu-layers", type=int, default=DEFAULT_N_GPU_LAYERS)
    return parser.parse_args()


def validate_data_dir(base_dir: Path) -> Path:
    resolved = base_dir.expanduser().resolve()
    if resolved.name != "Data":
        raise SystemExit(
            f"This benchmark is restricted to a folder named Data/. Got: {resolved}"
        )
    if not resolved.exists() or not resolved.is_dir():
        raise SystemExit(f"Data directory does not exist: {resolved}")
    return resolved


def resolve_folders(base_dir: Path, args: argparse.Namespace) -> list[Path]:
    base_dir = validate_data_dir(base_dir)
    if args.folders_file:
        names = [
            line.strip()
            for line in args.folders_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(
            Path(name).is_absolute() or "/" in name or "\\" in name for name in names
        ):
            raise SystemExit(
                "--folders-file must contain folder names from Data/ only, not paths."
            )
        folders = [base_dir / name for name in names]
    else:
        folders = sorted(
            (p for p in base_dir.iterdir() if p.is_dir()), key=lambda p: p.name
        )
        folders = folders[args.start : args.stop]

    random.Random(args.random_state).shuffle(folders)
    if args.max_papers is not None:
        folders = folders[: args.max_papers]
    return folders


def shard_folders(folders: list[Path], num_shards: int) -> list[list[Path]]:
    shards: list[list[Path]] = [[] for _ in range(num_shards)]
    for index, folder in enumerate(folders):
        shards[index % num_shards].append(folder)
    return shards


def write_run_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for result in results:
        status = result.get("status", "unknown")
        if status == "worker_done":
            continue
        counts[status] = counts.get(status, 0) + 1
    write_json(
        output_dir / "run_summary.json",
        {"counts": counts, "results": results},
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = resolve_folders(args.base_dir.expanduser(), args)
    if len(folders) > 100 and args.max_papers is None:
        raise SystemExit(
            "Refusing to process more than 100 papers unless --max-papers is set."
        )

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise SystemExit("No GPU ids were provided. Use --gpus 0,1 for two A40s.")

    num_workers = min(args.num_workers, len(gpu_ids), len(folders))
    if num_workers < 1:
        print("No folders selected.")
        return

    selected_gpus = gpu_ids[:num_workers]
    shards = shard_folders(folders, num_workers)
    args_dict = {
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "model_context_tokens": args.model_context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "context_safety_tokens": args.context_safety_tokens,
        "n_gpu_layers": args.n_gpu_layers,
    }

    write_json(
        output_dir / "run_config.json",
        {
            "base_dir": str(args.base_dir),
            "model_path": args.model_path,
            "max_papers": args.max_papers,
            "random_state": args.random_state,
            "selected_papers": [folder.name for folder in folders],
            "gpus": selected_gpus,
            "num_workers": num_workers,
            "prompting_strategy": "structured_text_template",
            **args_dict,
        },
    )

    print(
        f"Processing {len(folders)} papers with {num_workers} workers on GPUs {selected_gpus}. "
        f"Output: {output_dir}"
    )
    print(f"Random paper selection seed: {args.random_state}")
    print(
        f"Context guard: context={args.model_context_tokens}, "
        f"max_new={args.max_new_tokens}, safety={args.context_safety_tokens}"
    )
    print("Prompting strategy: structured text template (no JSON in prompt)")

    ctx = get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for rank, (gpu_id, shard) in enumerate(zip(selected_gpus, shards)):
        process = ctx.Process(
            target=worker_main,
            kwargs={
                "rank": rank,
                "gpu_id": gpu_id,
                "folders": [str(folder) for folder in shard],
                "args_dict": args_dict,
                "result_queue": result_queue,
            },
        )
        process.start()
        processes.append(process)

    results: list[dict[str, Any]] = []
    done_workers = 0
    with tqdm(total=len(folders), desc="direct llama-2 papers", unit="paper") as progress:
        while done_workers < len(processes):
            try:
                result = result_queue.get(timeout=10)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes):
                    break
                continue
            results.append(result)
            if result.get("status") == "worker_done":
                done_workers += 1
            elif result.get("status") == "worker_failed":
                done_workers += 1
                progress.write(f"Worker {result.get('worker')} failed.")
            else:
                progress.update(1)
                progress.set_postfix_str(
                    f"{result.get('paper')}:{result.get('status')}", refresh=False
                )
            write_run_summary(output_dir, results)

    for process in processes:
        process.join()

    write_run_summary(output_dir, results)
    print(f"Finished. Summary: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()