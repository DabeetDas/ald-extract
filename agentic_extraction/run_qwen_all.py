import argparse
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

from tqdm import tqdm

from tools import (
    characterization_agent,
    deposition_conditions_agent,
    film_properties_agent,
    precurosr_coreactant_purge_agent,
    reaction_conditions_agent,
    read_fulltext,
    substrate_information_agent,
    summariser_agent,
    target_materials_agent,
)
from validation import (
    classify_validation_issues,
    default_output,
    has_meaningful_content,
    validate_paper_outputs,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "qwen_extracted_info"

DEFAULT_MODEL_ID = (
    os.getenv("TRANSFORMERS_MODEL_PATH")
    or os.getenv("TRANSFORMERS_MODEL_ID")
    or "Qwen/Qwen3-8B"
)
DEFAULT_MAX_INPUT_TOKENS = int(os.getenv("TRANSFORMERS_MAX_INPUT_TOKENS", "32768"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("TRANSFORMERS_MAX_NEW_TOKENS", "4096"))
DEFAULT_DEVICE_MAP = os.getenv("TRANSFORMERS_DEVICE_MAP", "auto")
DEFAULT_TORCH_DTYPE = os.getenv("TRANSFORMERS_TORCH_DTYPE", "auto")
DEFAULT_LOCAL_FILES_ONLY = os.getenv("TRANSFORMERS_LOCAL_FILES_ONLY", "1").lower() in {
    "1",
    "true",
    "yes",
}
DEFAULT_TRUST_REMOTE_CODE = os.getenv("TRANSFORMERS_TRUST_REMOTE_CODE", "0").lower() in {
    "1",
    "true",
    "yes",
}
DEFAULT_LOAD_IN_4BIT = os.getenv("TRANSFORMERS_LOAD_IN_4BIT", "0").lower() in {
    "1",
    "true",
    "yes",
}
DEFAULT_LOAD_IN_8BIT = os.getenv("TRANSFORMERS_LOAD_IN_8BIT", "0").lower() in {
    "1",
    "true",
    "yes",
}
DEFAULT_ENABLE_THINKING = os.getenv("TRANSFORMERS_ENABLE_THINKING", "0").lower() in {
    "1",
    "true",
    "yes",
}

torch = None
AutoModelForCausalLM = None
AutoTokenizer = None
BitsAndBytesConfig = None


AGENTS = {
    "summary": summariser_agent,
    "target_material": target_materials_agent,
    "precursor_coreactant": precurosr_coreactant_purge_agent,
    "deposition_conditions": deposition_conditions_agent,
    "reaction_conditions": reaction_conditions_agent,
    "substrate_info": substrate_information_agent,
    "film_properties": film_properties_agent,
    "characterization": characterization_agent,
}

_model_lock = threading.Lock()
_agent_context = threading.local()


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
            f"Missing dependency '{missing}'. Install torch and transformers first."
        ) from exc

    torch = torch_module
    AutoModelForCausalLM = auto_model_for_causal_lm
    AutoTokenizer = auto_tokenizer
    BitsAndBytesConfig = bits_and_bytes_config


def is_gguf_file(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def resolve_model_source(model_ref: str, local_files_only: bool) -> str:
    path = Path(model_ref).expanduser()
    if path.exists():
        resolved_path = path.resolve()
        if resolved_path.is_file():
            if is_gguf_file(resolved_path):
                raise ValueError(
                    f"{resolved_path} is a GGUF file. Pass a Transformers checkpoint "
                    "directory containing config.json and tokenizer/model files."
                )
            raise ValueError(f"{resolved_path} is a file. Pass the model directory.")
        return str(resolved_path)

    if local_files_only:
        raise FileNotFoundError(
            f"Model path not found: {path}. Pass your local qwen3-8b-4bit directory "
            "with --model, or set TRANSFORMERS_MODEL_PATH."
        )

    return model_ref


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


def apply_chat_template(tokenizer, messages, max_input_tokens: int, enable_thinking: bool):
    template_kwargs = {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "truncation": True,
        "max_length": max_input_tokens,
    }
    if enable_thinking:
        return tokenizer.apply_chat_template(messages, **template_kwargs)

    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
    except Exception:
        return tokenizer.apply_chat_template(messages, **template_kwargs)


def build_plain_prompt(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message["role"].upper()
        parts.append(f"{role}:\n{message['content']}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def normalize_tokenizer_output(tokenized) -> dict:
    if hasattr(tokenized, "keys"):
        return {key: tokenized[key] for key in tokenized.keys()}
    return {"input_ids": tokenized}


class QwenTransformersLLM:
    def __init__(
        self,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        device_map: str,
        torch_dtype: str,
        trust_remote_code: bool,
        local_files_only: bool,
        load_in_4bit: bool,
        load_in_8bit: bool,
        enable_thinking: bool,
    ):
        self.requested_model = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.torch_dtype = parse_torch_dtype(torch_dtype)
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.enable_thinking = enable_thinking
        self.model_source: str | None = None
        self.tokenizer = None
        self.model = None
        self.raw_outputs: dict[str, str] = {}
        self.cleaned_outputs: dict[str, str] = {}

    def load(self) -> None:
        if self.model is not None:
            return

        ensure_transformers_imports()
        self.model_source = resolve_model_source(self.requested_model, self.local_files_only)
        tokenizer_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        model_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        quantization_config = build_quantization_config(self.load_in_4bit, self.load_in_8bit)
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        print(
            "Loading Qwen with transformers: "
            f"{self.model_source} (max_input_tokens={self.max_input_tokens}, "
            f"max_new_tokens={self.max_new_tokens})"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_source, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_source, **model_kwargs)
        self.model.eval()
        print("Model loaded.\n")

    def invoke(self, prompt: str):
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        messages = [
            {
                "role": "system",
                "content": (
                    "You extract Atomic Layer Deposition data from scientific papers. "
                    "Return one valid JSON object only. Use null only when the requested "
                    "field is truly absent from the provided text."
                ),
            },
            {
                "role": "user",
                "content": prompt + "\n\nReturn only valid JSON. Do not include markdown.",
            }
        ]
        if getattr(self.tokenizer, "chat_template", None):
            try:
                tokenized = apply_chat_template(
                    self.tokenizer,
                    messages,
                    self.max_input_tokens,
                    self.enable_thinking,
                )
                inputs = normalize_tokenizer_output(tokenized)
            except Exception:
                prompt_text = build_plain_prompt(messages)
                inputs = normalize_tokenizer_output(
                    self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_input_tokens,
                    )
                )
        else:
            prompt_text = build_plain_prompt(messages)
            inputs = normalize_tokenizer_output(
                self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_tokens,
                )
            )

        input_device = model_input_device(self.model)
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_tokens = generated[0][prompt_tokens:]
        raw = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        agent_name = getattr(_agent_context, "agent_name", "unknown")
        self.raw_outputs[agent_name] = raw

        content = strip_qwen_response(raw)
        self.cleaned_outputs[agent_name] = content
        return type("AIMessage", (), {"content": content})()


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


def run_single_agent(agent_name: str, agent_fn, fulltext: str, llm):
    try:
        with _model_lock:
            _agent_context.agent_name = agent_name
            result = agent_fn(fulltext, llm)
            if not has_meaningful_content(result, ignore_keys={"evidence"}):
                result["_warning"] = (
                    "Parsed successfully, but no extracted content was found. "
                    "Check _raw_model_outputs for this paper."
                )
                error = "Parsed successfully, but extracted content is empty."
            else:
                error = None
            _agent_context.agent_name = None
        return agent_name, result, error
    except Exception as exc:
        _agent_context.agent_name = None
        error_output = default_output(agent_name)
        error_message = f"{type(exc).__name__}: {str(exc) or repr(exc)}"
        error_output["_error"] = error_message
        error_output["_traceback"] = traceback.format_exc()
        return agent_name, error_output, error_message


def process_paper(folder: Path, output_dir: Path, llm) -> dict:
    txt_path = folder / "content.txt"
    if not txt_path.exists():
        print(f"Skipping {folder.name}: no content.txt")
        return {
            "status": "failed",
            "results": {},
            "errors": {"paper": "Missing content.txt"},
            "validation_issues": {},
        }

    fulltext = read_fulltext(str(txt_path))
    if not fulltext or not fulltext.strip():
        print(f"Skipping {folder.name}: content.txt is empty")
        return {
            "status": "failed",
            "results": {},
            "errors": {"paper": "Empty content.txt"},
            "validation_issues": {},
        }

    print(f"Processing {folder.name} ({len(fulltext):,} chars) ...")

    results: dict[str, object] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(AGENTS)) as pool:
        futures = {
            pool.submit(run_single_agent, name, fn, fulltext, llm): name
            for name, fn in AGENTS.items()
        }
        for future in as_completed(futures):
            name, result, error = future.result()
            results[name] = result
            marker_error = None
            if isinstance(result, dict):
                marker_error = result.get("_error") or result.get("_warning")
            if error or marker_error:
                errors[name] = error or marker_error
                print(f"   {name} failed: {errors[name]}")
            else:
                print(f"   {name}")

    paper_out = output_dir / folder.name
    paper_out.mkdir(parents=True, exist_ok=True)

    for agent_name, data in results.items():
        out_path = paper_out / f"{agent_name}.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    errors_path = paper_out / "agent_errors.json"
    if errors:
        with open(errors_path, "w", encoding="utf-8") as handle:
            json.dump(errors, handle, indent=2, ensure_ascii=False)
    elif errors_path.exists():
        errors_path.unlink()

    raw_out = paper_out / "_raw_model_outputs"
    raw_out.mkdir(exist_ok=True)
    for agent_name in AGENTS:
        raw_text = llm.raw_outputs.get(agent_name)
        if raw_text is None:
            continue
        with open(raw_out / f"{agent_name}.txt", "w", encoding="utf-8") as handle:
            handle.write(raw_text)

    cleaned_out = paper_out / "_cleaned_model_outputs"
    cleaned_out.mkdir(exist_ok=True)
    for agent_name in AGENTS:
        cleaned_text = llm.cleaned_outputs.get(agent_name)
        if cleaned_text is None:
            continue
        with open(cleaned_out / f"{agent_name}.txt", "w", encoding="utf-8") as handle:
            handle.write(cleaned_text)

    llm.raw_outputs.clear()
    llm.cleaned_outputs.clear()

    validation_issues = validate_paper_outputs(results, fulltext)

    validation_path = paper_out / "validation_issues.json"
    if validation_issues:
        with open(validation_path, "w", encoding="utf-8") as handle:
            json.dump(validation_issues, handle, indent=2, ensure_ascii=False)
    elif validation_path.exists():
        validation_path.unlink()

    status = "completed"
    if errors:
        status = "failed"
    else:
        validation_status = classify_validation_issues(validation_issues)
        if validation_status == "flagged":
            status = "flagged"
        elif validation_status == "warning":
            status = "warning"

    issue_count = sum(len(items) for items in validation_issues.values())
    print(
        f"Done: {folder.name} - {len(results)} outputs, {len(errors)} errors, "
        f"{issue_count} validation issues -> {status}\n"
    )
    return {
        "status": status,
        "results": results,
        "errors": errors,
        "validation_issues": validation_issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every selected ALD paper through a local Qwen Transformers model. "
            "No resume logs or skip logic are used."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT / "Data",
        help="Directory containing paper folders with content.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where fresh Qwen extraction JSON files are written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=(
            "Local Transformers checkpoint directory, such as your qwen3-8b-4bit "
            "folder. You can also set TRANSFORMERS_MODEL_PATH."
        ),
    )
    parser.add_argument(
        "--folders-file",
        type=Path,
        default=None,
        help="Optional newline-delimited list of folder names to process.",
    )
    parser.add_argument("--start", type=int, default=None, help="Optional start index.")
    parser.add_argument("--stop", type=int, default=None, help="Optional stop index.")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional cap for testing. Defaults to all selected folders.",
    )
    parser.add_argument(
        "--num-ctx",
        "--max-input-tokens",
        dest="max_input_tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="Maximum prompt tokens passed to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens per agent.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=DEFAULT_DEVICE_MAP,
        help="Transformers device_map.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=DEFAULT_TORCH_DTYPE,
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--allow-remote-model",
        dest="local_files_only",
        action="store_false",
        default=DEFAULT_LOCAL_FILES_ONLY,
        help="Allow transformers to download a repo id instead of requiring local files.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=DEFAULT_TRUST_REMOTE_CODE,
        help="Pass trust_remote_code=True to tokenizer/model loading.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=DEFAULT_LOAD_IN_4BIT,
        help="Quantize an unquantized checkpoint with bitsandbytes at load time.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        default=DEFAULT_LOAD_IN_8BIT,
        help="Quantize an unquantized checkpoint with bitsandbytes at load time.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help="Keep Qwen3 thinking mode enabled. Defaults to disabled for JSON extraction.",
    )
    return parser.parse_args()


def resolve_folders(base_dir: Path, args: argparse.Namespace) -> list[Path]:
    if args.folders_file:
        folder_names = [
            line.strip()
            for line in args.folders_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        folders = [base_dir / folder_name for folder_name in folder_names]
    else:
        folders = sorted(
            (path for path in base_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )

    start = args.start
    stop = args.stop
    if start is not None or stop is not None:
        folders = folders[start:stop]
    if args.max_papers is not None:
        folders = folders[: args.max_papers]
    return folders


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = QwenTransformersLLM(
        model_name=args.model,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        enable_thinking=args.enable_thinking,
    )

    try:
        llm.load()
    except Exception as exc:
        raise SystemExit(f"Model load failed: {exc}") from exc

    folders = resolve_folders(args.base_dir.expanduser(), args)
    print(f"Processing {len(folders)} folders. Output: {output_dir}\n")

    counts = {"completed": 0, "warning": 0, "flagged": 0, "failed": 0}
    for folder in tqdm(folders):
        try:
            outcome = process_paper(folder, output_dir, llm)
            status = outcome["status"]
            counts[status] = counts.get(status, 0) + 1
        except Exception as exc:
            counts["failed"] += 1
            print(f"Failed on {folder.name}: {exc}")
            traceback.print_exc()

    print(
        "\nFinished fresh Qwen extraction. "
        f"completed={counts['completed']}, warning={counts['warning']}, "
        f"flagged={counts['flagged']}, failed={counts['failed']}"
    )


if __name__ == "__main__":
    main()
