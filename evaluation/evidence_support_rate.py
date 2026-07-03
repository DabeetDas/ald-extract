import argparse
import asyncio
import copy
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import dotenv
dotenv.load_dotenv()

try:
    from groq import AsyncGroq
except ImportError:
    sys.exit("Error: The 'groq' python package is required. Install via: pip install groq")

try:
    from scipy.stats import chi2_contingency
except ImportError:
    chi2_contingency = None
    print("Warning: 'scipy' is not installed. P-values will not be generated. Install via: pip install scipy")

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

SYS_PROMPT = """You are an expert materials scientist evaluating an information extraction system. 
You are given a short "Source Text" quote from a scientific paper, and a set of "Extracted Claims" (represented as a JSON object).

Your task is to determine if *ALL* of the positively stated Extracted Claims can be fully supported and directly inferred from the Source Text alone, without relying on outside knowledge.

* CRITICAL EVALUATION RULES:
1. IGNORE any JSON fields that have empty strings "", "unknown", "none", "not mentioned", "n/a", or null. Do not penalize the extraction for having empty/missing fields.
2. Only evaluate fields that contain actual extracted information (e.g., numbers, materials, conditions).
3. If all the populated fields are supported by the Source Text, output exactly "1".
4. If ANY populated field contains hallucinations, contradictory information, or info drawn from outside the Source Text, output exactly "0".

Output precisely "1" if fully supported, or "0" if unsupported. Do not output any additional text, explanation, or markdown formatting.
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Evidence Support Rate using LLM-as-a-judge.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Dataset to evaluate. Default: qwen=qwen_extracted_info and llama32_3b=extracted_data.",
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=500, 
        help="Number of random samples to evaluate per agent per dataset."
    )
    parser.add_argument(
        "--concurrency", 
        type=int, 
        default=15, 
        help="Max concurrent tasks for the API."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "evaluations",
        help="Directory where output files are written.",
    )
    return parser.parse_args()


def resolve_datasets(dataset_args: list[str]) -> dict[str, Path]:
    if not dataset_args:
        return DEFAULT_DATASETS
    datasets: dict[str, Path] = {}
    for item in dataset_args:
        label, raw_path = item.split("=", 1)
        path = Path(raw_path.strip()).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        datasets[label.strip()] = path
    return datasets


def has_meaningful_content(value: Any) -> bool:
    if isinstance(value, dict):
        return any(has_meaningful_content(child) for child in value.values())
    if isinstance(value, list):
        return any(has_meaningful_content(item) for item in value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def load_valid_extractions(dataset_dir: Path, agent_name: str) -> List[Dict[str, Any]]:
    valid_items = []
    if not dataset_dir.exists():
        return valid_items
        
    for paper_dir in dataset_dir.iterdir():
        if not paper_dir.is_dir() or paper_dir.name.startswith("."):
            continue
            
        json_path = paper_dir / f"{agent_name}.json"
        if not json_path.exists():
            continue
            
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
            
        if not isinstance(data, dict):
            continue
            
        evidence = data.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            continue
            
        claims = {k: v for k, v in data.items() if k != "evidence"}
        if has_meaningful_content(claims):
            valid_items.append({
                "paper_id": paper_dir.name,
                "evidence": evidence.strip(),
                "claims": claims
            })
            
    return valid_items


async def evaluate_single_extraction(
    client: AsyncGroq, 
    semaphore: asyncio.Semaphore, 
    item: Dict[str, Any], 
) -> int:
    evidence = item["evidence"]
    claims_str = json.dumps(item["claims"], indent=2)
    
    user_prompt = f"Source Text:\n{evidence}\n\nExtracted Claims:\n{claims_str}"
    
    max_retries = 5
    base_delay = 2
    
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_completion_tokens=5,
                )
                
                content = response.choices[0].message.content.strip()
                if content == "1":
                    return 1
                elif content == "0":
                    return 0
                else:
                    # Model failed to follow format -> assume 0 (unsupported)
                    return 0
            except Exception as e:
                err_msg = str(e).lower()
                if "rate limit" in err_msg or "429" in err_msg:
                    await asyncio.sleep(base_delay * (2 ** attempt) + random.uniform(0, 1))
                else:
                    return 0
        return 0


async def evaluate_agent_dataset(
    client: AsyncGroq, 
    semaphore: asyncio.Semaphore, 
    dataset_label: str, 
    agent_name: str, 
    items: List[Dict[str, Any]]
) -> Tuple[int, int]:
    
    tasks = [evaluate_single_extraction(client, semaphore, item) for item in items]
    results = await asyncio.gather(*tasks)
    
    supported_count = sum(results)
    return supported_count, len(items)


def calculate_p_value(count1: int, nobs1: int, count2: int, nobs2: int) -> float | None:
    if chi2_contingency is None:
        return None
    if nobs1 == 0 or nobs2 == 0:
        return None
        
    table = [
        [count1, nobs1 - count1],
        [count2, nobs2 - count2]
    ]
    try:
        _, p_val, _, _ = chi2_contingency(table)
        return p_val
    except Exception:
        return None


async def main_async(args: argparse.Namespace):
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("Error: GROQ_API_KEY environment variable is missing.")
        
    datasets = resolve_datasets(args.dataset)
    client = AsyncGroq()
    semaphore = asyncio.Semaphore(args.concurrency)
    
    all_results = {}
    
    print(f"Loading extractions and sampling {args.sample_size} per agent...")
    samples_to_run = []
    
    for label, extracted_dir in datasets.items():
        all_results[label] = {}
        for agent_name in AGENT_NAMES:
            valid_items = load_valid_extractions(extracted_dir, agent_name)
            
            if len(valid_items) > args.sample_size:
                # Use a specific seed for reproducibility across runs
                sampled = random.Random(42).sample(valid_items, args.sample_size)
            else:
                sampled = valid_items
                
            samples_to_run.append((label, agent_name, sampled))
            all_results[label][agent_name] = {
                "supported": 0,
                "total": 0,
                "support_rate": None
            }
            
    print(f"Total API calls to make: {sum(len(items) for _, _, items in samples_to_run)}")
    
    # Run evaluation
    for label, agent_name, items in samples_to_run:
        print(f"Evaluating {label} - {agent_name} ({len(items)} samples)...")
        if not items:
            continue
            
        supported, total = await evaluate_agent_dataset(client, semaphore, label, agent_name, items)
        
        all_results[label][agent_name]["supported"] = supported
        all_results[label][agent_name]["total"] = total
        all_results[label][agent_name]["support_rate"] = round((supported / total) * 100, 2) if total > 0 else 0.0

    print("\n--- RESULTS ---\n")
    report = ["# Evidence Support Rate Evaluation\n", "LLM Judge: llama-3.3-70b-versatile via Groq\n"]
    
    dataset_keys = list(all_results.keys())
    
    if len(dataset_keys) == 2:
        d1, d2 = dataset_keys[0], dataset_keys[1]
        report.append(f"| Agent | {d1} Support Rate | {d2} Support Rate | p-Value |")
        report.append("|---|---|---|---|")
        
        for agent_name in AGENT_NAMES:
            res1 = all_results[d1][agent_name]
            res2 = all_results[d2][agent_name]
            
            rate1_str = f"{res1['support_rate']}% ({res1['supported']}/{res1['total']})" if res1['total'] > 0 else "N/A"
            rate2_str = f"{res2['support_rate']}% ({res2['supported']}/{res2['total']})" if res2['total'] > 0 else "N/A"
            
            p_val = calculate_p_value(res1['supported'], res1['total'], res2['supported'], res2['total'])
            if p_val is not None:
                p_val_str = f"{p_val:.4f}"
                if p_val < 0.05:
                    p_val_str += " **(Sig)**"
            else:
                p_val_str = "N/A"
                
            row = f"| {agent_name} | {rate1_str} | {rate2_str} | {p_val_str} |"
            print(row)
            report.append(row)
            
    else:
        for label, agents in all_results.items():
            report.append(f"## {label}")
            for agent, stats in agents.items():
                res_str = f"{agent}: {stats['support_rate']}% ({stats['supported']}/{stats['total']})"
                print(res_str)
                report.append(f"- {res_str}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "evidence_support_rate.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"\nSaved report to {report_path}")

def main():
    args = parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
