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


DEFAULT_MODEL_ID = os.getenv("TRANSFORMERS_MODEL_PATH") or os.getenv(
    "TRANSFORMERS_MODEL_ID",
    "Qwen/Qwen3-8B",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarking" / "qwen3_8b_direct_outputs"
DEFAULT_MODEL_CONTEXT_TOKENS = int(os.getenv("QWEN3_MODEL_CONTEXT_TOKENS", "32768"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("QWEN3_DIRECT_MAX_NEW_TOKENS", "6144"))
DEFAULT_CONTEXT_SAFETY_TOKENS = int(os.getenv("QWEN3_CONTEXT_SAFETY_TOKENS", "1024"))
DEFAULT_NUM_WORKERS = int(os.getenv("QWEN3_DIRECT_WORKERS", "2"))
DEFAULT_GPUS = os.getenv("QWEN3_DIRECT_GPUS", "0,1")
DEFAULT_RANDOM_STATE = int(os.getenv("QWEN3_DIRECT_RANDOM_STATE", "42"))


torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
BitsAndBytesConfig = None


def ensure_transformers_imports() -> None:
    global torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if torch is not None:
        return
    try:
        import torch as torch_module
        from transformers import (
            AutoModelForCausalLM as auto_model_for_causal_lm,
            AutoTokenizer as auto_tokenizer,
            BitsAndBytesConfig as bits_and_bytes_config,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        missing = exc.name or "required package"
        raise ModuleNotFoundError(
            f"Missing dependency '{missing}'. Install torch, transformers, and tqdm first."
        ) from exc

    torch = torch_module
    AutoModelForCausalLM = auto_model_for_causal_lm
    AutoTokenizer = auto_tokenizer
    BitsAndBytesConfig = bits_and_bytes_config


def parse_torch_dtype(dtype_name: str | None):
    if not dtype_name or dtype_name == "auto":
        return None
    ensure_transformers_imports()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype_map[dtype_name]


def build_quantization_config(load_in_4bit: bool, load_in_8bit: bool):
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one of --load-in-4bit or --load-in-8bit.")
    if not load_in_4bit and not load_in_8bit:
        return None

    ensure_transformers_imports()
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def normalize_tokenizer_output(tokenized) -> dict[str, Any]:
    if hasattr(tokenized, "keys"):
        return {key: tokenized[key] for key in tokenized.keys()}
    return {"input_ids": tokenized}


def strip_qwen_response(raw: str) -> str:
    content = raw.strip()
    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    else:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    if "```json" in content:
        content = content.split("```json", 1)[1]
        content = content.split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1]
        content = content.split("```", 1)[0]

    json_start_positions = [
        position for position in (content.find("{"), content.find("[")) if position != -1
    ]
    if json_start_positions:
        content = content[min(json_start_positions) :]

    content = re.sub(r"<\|[^|]*\|>", "", content).strip()
    return content


def combined_default() -> dict[str, Any]:
    return json.loads(json.dumps({name: AGENT_DEFAULTS[name] for name in AGENT_NAMES}))


def build_combined_prompt(text: str) -> str:
    schema = combined_default()
    return f"""\
You are a materials science expert specializing in Atomic Layer Deposition (ALD).

Extract all requested structured information from the paper text.

Rules:
1. Use only information explicitly supported by the provided text.
2. Do not hallucinate values.
3. If information is absent, use null for nullable numeric/scalar fields, "" for text fields whose schema default is "", and [] for list fields.
4. Include exact evidence sentence(s) from the paper for every non-empty section.
5. Return exactly one valid JSON object and nothing else.
6. The JSON must have exactly these top-level keys:
   {", ".join(AGENT_NAMES)}

Output schema and defaults:
```json
{json.dumps(schema, indent=2, ensure_ascii=False)}
```

Field guidance:
- summary: deposited material, process type, main precursors, temperature range, concise study summary.
- target_material: only the primary deposited film, not substrates or precursors.
- precursor_coreactant: precursors, coreactants, purge gas, carrier gas explicitly mentioned.
- deposition_conditions: temperature, pressure, pulse/purge times, cycles, reactor type.
- reaction_conditions: formal equations, surface mechanism, intermediate species.
- substrate_info: substrate material/orientation, pretreatment, surface functionalization.
- film_properties: film thickness, density, refractive index, roughness, crystal phase.
- characterization: techniques actually used in the study.

Paper text:
```
{text}
```
"""


def truncate_for_context(
    fulltext: str,
    tokenizer,
    *,
    model_context_tokens: int,
    max_new_tokens: int,
    safety_tokens: int,
) -> tuple[str, dict[str, Any]]:
    prompt_budget = model_context_tokens - max_new_tokens - safety_tokens
    if prompt_budget <= 2048:
        raise ValueError(
            "Token budget is too small. Increase --model-context-tokens or reduce "
            "--max-new-tokens / --context-safety-tokens."
        )

    empty_prompt = build_combined_prompt("")
    overhead_tokens = len(tokenizer.encode(empty_prompt, add_special_tokens=False))
    text_budget = prompt_budget - overhead_tokens
    if text_budget <= 256:
        raise ValueError(
            f"Prompt overhead leaves only {text_budget} text tokens. "
            "Reduce schema/prompt size or increase context budget."
        )

    text_tokens = tokenizer.encode(fulltext, add_special_tokens=False)
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
    truncated = tokenizer.decode(selected, skip_special_tokens=True)
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


class DirectQwenExtractor:
    def __init__(
        self,
        *,
        model_name: str,
        model_context_tokens: int,
        max_new_tokens: int,
        context_safety_tokens: int,
        torch_dtype: str,
        local_files_only: bool,
        trust_remote_code: bool,
        load_in_4bit: bool,
        load_in_8bit: bool,
        enable_thinking: bool,
    ):
        self.model_name = model_name
        self.model_context_tokens = model_context_tokens
        self.max_new_tokens = max_new_tokens
        self.context_safety_tokens = context_safety_tokens
        self.torch_dtype_name = torch_dtype
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.enable_thinking = enable_thinking
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        ensure_transformers_imports()
        tokenizer_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        dtype = parse_torch_dtype(self.torch_dtype_name)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        quantization_config = build_quantization_config(self.load_in_4bit, self.load_in_8bit)
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    def apply_chat_template(self, prompt: str):
        assert self.tokenizer is not None
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract ALD data from scientific papers. "
                    "Return one valid JSON object only."
                ),
            },
            {
                "role": "user",
                "content": prompt + "\n\nReturn only valid JSON. Do not include markdown.",
            },
        ]
        if getattr(self.tokenizer, "chat_template", None):
            template_kwargs = {
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "truncation": True,
                "max_length": self.model_context_tokens - self.max_new_tokens,
            }
            if not self.enable_thinking:
                try:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        enable_thinking=False,
                        **template_kwargs,
                    )
                except Exception:
                    pass
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)

        plain = "\n\n".join(
            [
                f"{message['role'].upper()}:\n{message['content']}"
                for message in messages
            ]
            + ["ASSISTANT:\n"]
        )
        return self.tokenizer(
            plain,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_context_tokens - self.max_new_tokens,
        )

    def extract(self, fulltext: str) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        assert self.tokenizer is not None
        assert self.model is not None
        truncated_text, token_report = truncate_for_context(
            fulltext,
            self.tokenizer,
            model_context_tokens=self.model_context_tokens,
            max_new_tokens=self.max_new_tokens,
            safety_tokens=self.context_safety_tokens,
        )
        prompt = build_combined_prompt(truncated_text)
        tokenized = self.apply_chat_template(prompt)
        inputs = normalize_tokenizer_output(tokenized)
        device = model_input_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_tokens = inputs["input_ids"].shape[-1]
        raw = self.tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=False)
        cleaned = strip_qwen_response(raw)
        parsed = robust_json_parse(cleaned, default=combined_default())
        return parsed, raw, cleaned, token_report


def read_fulltext(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def process_paper(folder: Path, output_dir: Path, extractor: DirectQwenExtractor) -> dict[str, Any]:
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
        combined, raw, cleaned, token_report = extractor.extract(fulltext)
        results = {}
        for agent_name in AGENT_NAMES:
            results[agent_name] = combined.get(agent_name) or AGENT_DEFAULTS[agent_name]
            write_json(paper_out / f"{agent_name}.json", results[agent_name])

        raw_out = paper_out / "_raw_model_outputs"
        raw_out.mkdir(exist_ok=True)
        (raw_out / "combined.txt").write_text(raw, encoding="utf-8")
        cleaned_out = paper_out / "_cleaned_model_outputs"
        cleaned_out.mkdir(exist_ok=True)
        (cleaned_out / "combined.jsonish.txt").write_text(cleaned, encoding="utf-8")
        write_json(paper_out / "token_report.json", token_report)

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
            (cleaned_out / "combined.jsonish.txt").write_text(cleaned, encoding="utf-8")
        return error


def worker_main(
    *,
    rank: int,
    gpu_id: str,
    folders: list[str],
    args_dict: dict[str, Any],
    result_queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    output_dir = Path(args_dict["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        extractor = DirectQwenExtractor(
            model_name=args_dict["model"],
            model_context_tokens=args_dict["model_context_tokens"],
            max_new_tokens=args_dict["max_new_tokens"],
            context_safety_tokens=args_dict["context_safety_tokens"],
            torch_dtype=args_dict["torch_dtype"],
            local_files_only=args_dict["local_files_only"],
            trust_remote_code=args_dict["trust_remote_code"],
            load_in_4bit=args_dict["load_in_4bit"],
            load_in_8bit=args_dict["load_in_8bit"],
            enable_thinking=args_dict["enable_thinking"],
        )
        print(f"[worker {rank}] loading model on CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark direct Qwen3-8B extraction: one combined schema call per paper, "
            "split into the same JSON files as the agentic pipeline. Defaults to 100 papers."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=REPO_ROOT / "Data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--folders-file", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=100)
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Seed used to randomly select papers from Data/. Default: 42.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=DEFAULT_GPUS,
        help="Comma-separated physical GPU ids. Default: 0,1 for two A40s.",
    )
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--model-context-tokens",
        type=int,
        default=DEFAULT_MODEL_CONTEXT_TOKENS,
        help="Qwen3 context window. Input budget is context minus max_new_tokens and safety tokens.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--context-safety-tokens",
        type=int,
        default=DEFAULT_CONTEXT_SAFETY_TOKENS,
        help="Extra context margin to avoid position-limit overshoot.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=os.getenv("TRANSFORMERS_TORCH_DTYPE", "bfloat16"),
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=os.getenv("TRANSFORMERS_LOCAL_FILES_ONLY", "0").lower()
        in {"1", "true", "yes"},
        help="Require a local model cache/path.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=os.getenv("TRANSFORMERS_TRUST_REMOTE_CODE", "0").lower()
        in {"1", "true", "yes"},
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Keep Qwen3 thinking mode enabled. Default disables it where chat template supports it.",
    )
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
        if any(Path(name).is_absolute() or "/" in name or "\\" in name for name in names):
            raise SystemExit(
                "--folders-file must contain folder names from Data/ only, not paths."
            )
        folders = [base_dir / name for name in names]
    else:
        folders = sorted((path for path in base_dir.iterdir() if path.is_dir()), key=lambda p: p.name)
        folders = folders[args.start : args.stop]

    random.Random(args.random_state).shuffle(folders)
    if args.max_papers is not None:
        folders = folders[: args.max_papers]
    return folders


def shard_folders(folders: list[Path], num_shards: int) -> list[list[Path]]:
    shards = [[] for _ in range(num_shards)]
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
        {
            "counts": counts,
            "results": results,
        },
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    folders = resolve_folders(args.base_dir.expanduser(), args)
    if len(folders) > 100 and args.max_papers is None:
        raise SystemExit("Refusing to process more than 100 papers unless --max-papers is set.")

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
        "model": args.model,
        "model_context_tokens": args.model_context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "context_safety_tokens": args.context_safety_tokens,
        "torch_dtype": args.torch_dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
        "enable_thinking": args.enable_thinking,
    }

    write_json(
        output_dir / "run_config.json",
        {
            "base_dir": str(args.base_dir),
            "model": args.model,
            "max_papers": args.max_papers,
            "random_state": args.random_state,
            "selected_papers": [folder.name for folder in folders],
            "gpus": selected_gpus,
            "num_workers": num_workers,
            **args_dict,
        },
    )

    print(
        f"Processing {len(folders)} papers with {num_workers} workers on GPUs {selected_gpus}. "
        f"Output: {output_dir}"
    )
    print(f"Random paper selection seed: {args.random_state}")
    print(
        "Context guard: "
        f"context={args.model_context_tokens}, max_new={args.max_new_tokens}, "
        f"safety={args.context_safety_tokens}"
    )

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
    with tqdm(total=len(folders), desc="direct qwen3 papers", unit="paper") as progress:
        while done_workers < len(processes):
            try:
                result = result_queue.get(timeout=10)
            except queue.Empty:
                if any(process.exitcode not in (None, 0) for process in processes):
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
                    f"{result.get('paper')}:{result.get('status')}",
                    refresh=False,
                )
            write_run_summary(output_dir, results)

    for process in processes:
        process.join()

    write_run_summary(output_dir, results)
    print(f"Finished. Summary: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
