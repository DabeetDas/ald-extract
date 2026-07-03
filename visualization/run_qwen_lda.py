#!/usr/bin/env python3
"""Run exploratory LDA topic modeling on Qwen extracted ALD information."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "qwen_extracted_info"
DEFAULT_OUTPUT_DIR = ROOT / "qwen_lda_metrics"
DEFAULT_CLEANED_OUTPUT_DIR = ROOT / "qwen_lda_cleaned_metrics"
SECTION_FILES = [
    "summary.json",
    "target_material.json",
    "substrate_info.json",
    "deposition_conditions.json",
    "precursor_coreactant.json",
    "reaction_conditions.json",
    "film_properties.json",
    "characterization.json",
]
CLEANED_SECTION_FILES = [
    "summary.txt",
    "target_material.txt",
    "substrate_info.txt",
    "deposition_conditions.txt",
    "precursor_coreactant.txt",
    "reaction_conditions.txt",
    "film_properties.txt",
    "characterization.txt",
]
EXTRA_STOPWORDS = {
    "ald",
    "atomic",
    "layer",
    "deposition",
    "film",
    "films",
    "thin",
    "paper",
    "reported",
    "evidence",
    "summary",
    "temperature",
    "temperatures",
    "process",
    "material",
    "materials",
    "substrate",
    "substrates",
    "using",
    "used",
    "null",
    "none",
    "unknown",
    "n",
    "na",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(flatten_text(item))
        return parts
    if isinstance(value, dict):
        parts = []
        for item in value.values():
            parts.extend(flatten_text(item))
        return parts
    return []


def tokenize(text: str) -> list[str]:
    normalized = (
        text.lower()
        .replace("₂", "2")
        .replace("₃", "3")
        .replace("₄", "4")
        .replace("₅", "5")
        .replace("₆", "6")
        .replace("₇", "7")
        .replace("₈", "8")
        .replace("₉", "9")
        .replace("₀", "0")
    )
    tokens = re.findall(r"[a-z][a-z0-9]{2,}", normalized)
    return [
        token
        for token in tokens
        if token not in STOPWORDS
        and not token.isdigit()
        and not re.fullmatch(r"c\d+", token)
    ]


def load_extracted_documents(input_dir: Path) -> list[dict[str, Any]]:
    documents = []
    for paper_dir in sorted(input_dir.iterdir(), key=lambda item: item.name):
        if not paper_dir.is_dir():
            continue
        section_text: list[str] = []
        for filename in SECTION_FILES:
            section_text.extend(flatten_text(read_json(paper_dir / filename)))
        text = " ".join(section_text)
        tokens = tokenize(text)
        if tokens:
            documents.append({"id": paper_dir.name, "text": text, "tokens": tokens})
    return documents


def cleaned_file_text(path: Path) -> list[str]:
    text = read_text(path)
    if not text.strip():
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    return flatten_text(parsed)


def load_cleaned_documents(input_dir: Path) -> list[dict[str, Any]]:
    documents = []
    for paper_dir in sorted(input_dir.iterdir(), key=lambda item: item.name):
        if not paper_dir.is_dir():
            continue
        cleaned_dir = paper_dir / "_cleaned_model_outputs"
        if not cleaned_dir.is_dir():
            continue
        section_text: list[str] = []
        for filename in CLEANED_SECTION_FILES:
            section_text.extend(cleaned_file_text(cleaned_dir / filename))
        text = " ".join(section_text)
        tokens = tokenize(text)
        if tokens:
            documents.append({"id": paper_dir.name, "text": text, "tokens": tokens})
    return documents


def load_documents(input_dir: Path, source: str) -> list[dict[str, Any]]:
    if source == "cleaned":
        return load_cleaned_documents(input_dir)
    return load_extracted_documents(input_dir)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_tokens(documents: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for document in documents:
        counts.update(document["tokens"])
    return [{"token": token, "count": count} for token, count in counts.most_common(limit)]


def train_model(dictionary: corpora.Dictionary, corpus: list[list[tuple[int, int]]], num_topics: int, passes: int, iterations: int) -> LdaModel:
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        chunksize=400,
        passes=passes,
        iterations=iterations,
        alpha="auto",
        eta="auto",
        eval_every=None,
    )


def evaluate_topic_counts(
    dictionary: corpora.Dictionary,
    corpus: list[list[tuple[int, int]]],
    tokenized_docs: list[list[str]],
    topic_counts: list[int],
    passes: int,
    iterations: int,
) -> tuple[list[dict[str, Any]], dict[int, LdaModel]]:
    rows = []
    models: dict[int, LdaModel] = {}
    for num_topics in topic_counts:
        model = train_model(dictionary, corpus, num_topics, passes, iterations)
        models[num_topics] = model
        coherence_cv = CoherenceModel(
            model=model,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_v",
            processes=1,
        ).get_coherence()
        coherence_umass = CoherenceModel(
            model=model,
            corpus=corpus,
            dictionary=dictionary,
            coherence="u_mass",
            processes=1,
        ).get_coherence()
        rows.append(
            {
                "num_topics": num_topics,
                "coherence_cv": round(coherence_cv, 6),
                "coherence_umass": round(coherence_umass, 6),
                "log_perplexity": round(model.log_perplexity(corpus), 6),
            }
        )
        print(
            f"k={num_topics}: c_v={coherence_cv:.4f}, "
            f"u_mass={coherence_umass:.4f}, log_perplexity={model.log_perplexity(corpus):.4f}",
            flush=True,
        )
    return rows, models


def choose_best_topic_count(metrics: list[dict[str, Any]]) -> int:
    return int(max(metrics, key=lambda row: row["coherence_cv"])["num_topics"])


def model_topics(model: LdaModel, topn: int) -> list[dict[str, Any]]:
    rows = []
    for topic_id in range(model.num_topics):
        terms = model.show_topic(topic_id, topn=topn)
        rows.append(
            {
                "topic_id": topic_id,
                "top_terms": ", ".join(term for term, _ in terms),
                "weighted_terms": ", ".join(f"{term}:{weight:.4f}" for term, weight in terms),
            }
        )
    return rows


def document_topics(model: LdaModel, corpus: list[list[tuple[int, int]]], documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for document, bow in zip(documents, corpus):
        topics = model.get_document_topics(bow, minimum_probability=0)
        dominant_topic, dominant_probability = max(topics, key=lambda item: item[1])
        topic_distribution = {
            str(int(topic)): round(float(probability), 6)
            for topic, probability in topics
        }
        rows.append(
            {
                "document_id": document["id"],
                "dominant_topic": int(dominant_topic),
                "dominant_probability": round(float(dominant_probability), 6),
                "topic_distribution": json.dumps(topic_distribution),
                "token_count": len(document["tokens"]),
            }
        )
    return rows


def parse_topic_counts(value: str) -> list[int]:
    counts: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(piece.strip()) for piece in part.split("-", 1)]
            counts.update(range(start, end + 1))
        else:
            counts.add(int(part))
    return sorted(count for count in counts if count > 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LDA on Qwen extracted ALD data and save metrics.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source", choices=["extracted", "cleaned"], default="extracted")
    parser.add_argument("--topics", default="4-16", help="Comma-separated topic counts or ranges, e.g. 4-16 or 4,6,8")
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--topn", type=int, default=15)
    args = parser.parse_args()

    output_dir = args.output_dir or (DEFAULT_CLEANED_OUTPUT_DIR if args.source == "cleaned" else DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    documents = load_documents(args.input_dir, args.source)
    tokenized_docs = [document["tokens"] for document in documents]
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=6000)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]
    filtered = [(document, bow) for document, bow in zip(documents, corpus) if bow]
    documents = [document for document, _ in filtered]
    corpus = [bow for _, bow in filtered]
    tokenized_docs = [document["tokens"] for document in documents]

    topic_counts = parse_topic_counts(args.topics)
    metrics, models = evaluate_topic_counts(dictionary, corpus, tokenized_docs, topic_counts, args.passes, args.iterations)
    best_k = choose_best_topic_count(metrics)
    best_model = models[best_k]
    topics = model_topics(best_model, args.topn)
    doc_topics = document_topics(best_model, corpus, documents)

    write_csv(output_dir / "lda_metrics.csv", metrics, ["num_topics", "coherence_cv", "coherence_umass", "log_perplexity"])
    write_csv(output_dir / "lda_topics.csv", topics, ["topic_id", "top_terms", "weighted_terms"])
    write_csv(
        output_dir / "document_topics.csv",
        doc_topics,
        ["document_id", "dominant_topic", "dominant_probability", "topic_distribution", "token_count"],
    )
    write_csv(output_dir / "top_tokens.csv", top_tokens(documents), ["token", "count"])

    summary = {
        "input_dir": str(args.input_dir),
        "source": args.source,
        "document_count": len(documents),
        "vocabulary_size": len(dictionary),
        "topic_counts_tested": topic_counts,
        "selected_num_topics": best_k,
        "selection_rule": "highest c_v coherence",
        "passes": args.passes,
        "iterations": args.iterations,
        "files": {
            "metrics": "lda_metrics.csv",
            "topics": "lda_topics.csv",
            "document_topics": "document_topics.csv",
            "top_tokens": "top_tokens.csv",
        },
    }
    (output_dir / "lda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    best_model.save(str(output_dir / "lda_best_model.gensim"))
    dictionary.save(str(output_dir / "lda_dictionary.gensim"))
    print(f"Selected k={best_k} by highest c_v coherence.")
    print(f"Saved LDA outputs to {output_dir}")


if __name__ == "__main__":
    main()
