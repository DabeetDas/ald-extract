# ALD-Extract: Multi-Agent Extraction and Database Creation for Atomic Layer Deposition

This repository contains the core programmatic modules used for the compilation, extraction, and validation of thin-film ALD (Atomic Layer Deposition) properties from academic literature using Large Language Models (LLMs). This minimal codebase represents the methodology discussed in our study on accelerating ALD research with Generative AI.


## Codebase Organization

The framework pipeline consists of 6 primary stages, sequentially organized:

```text
ald-extract/
├── README.md                      # Overview of the framework
├── agentic_extraction/            # Core LLM data extraction workflow scripts
├── agentic_rag/                   # LLM/Backend architecture for querying extracted data
├── benchmarking/                  # Model benchmarking and performance scripts
├── data_collection/               # Publication download and metadata crawling tools
├── evaluation/                    # Post-extraction metrics, grounding, and validation
├── preprocessing/                 # Text normalization, chunking, and layout processing
└── visualization/                 # ALD property analysis and graph generation
```

## Module Functionality 

### 1. Data Collection (`data_collection/`)
Scripts used to gather publication metadata and acquire literature source files.
- `scrape.py`: Harvests basic bibliographic data (e.g. from Elsevier) based on query parameters.
- `doi2pdf.py` & `scihub.py`: Utilitarian scripts for securely mapping DOIs to PDF literature via API wrappers/mirrors.

### 2. Preprocessing (`preprocessing/`)
Translates unstructured PDFs into structured inputs for the LLM. 
- `chunker/`: Contains tools (`img_to_txt.py`) for optical character recognition (OCR) parsing where needed, and structured semantic chunking (`generate_schema.py`).
- `regex_functions.py` & `prep_data.py`: Sanitize and clean raw text (normalizing superscripts/subscripts, standardizing chemical formulae, etc.). 
- `dataset_sanity_check.py`: Rejects corrupt/unreadable files prior to inference.

### 3. Agentic Extraction (`agentic_extraction/`)
The LLM inference engine that reads ALD papers and outputs standardized JSON representations of the materials science properties. 
- `run_agent.py` & `run_qwen_all.py`: The main orchestration loops. They load the documents, query the model (such as LLaMA or Qwen), and capture structured ALD metrics.
- `tools.py` & `tools_ollama.py`: Define the agentic tool-using functions for targeted searching over the ALD taxonomy parameters (precursors, temperatures, GPC, substrate, etc.). 
- `validation.py`: Immediately verifies extracted schema adherence, prompting the LLM to auto-correct upon error.

### 4. Evaluation (`evaluation/`)
Post-processing tools for quantifying model accuracy across a subset of truth-annotated data.
- `evaluate_qwen_extractions.py`: Grades LLM extractions against human annotations or benchmark truth tables.
- `evidence_grounding_score.py` & `evidence_support_rate.py`: Uses separate validation algorithms to trace any extracted material parameter back to the specific context in the text to prevent hallucinations.

### 5. RAG Pipeline (`agentic_rag/`)
The analytical backend querying apparatus that allows natural language questioning over the synthesized dataset. 
- Implements a FastAPI application wrapped around LangChain and vector stores, providing endpoints for searching precursor compatibility grids or ALD parameters via `main.py` and `service.py`.

### 6. Benchmarking & Visualization (`benchmarking/`, `visualization/`)
- **Benchmarking**: `run_qwen3_8b_direct_100.py` and `run_llama2_chat_gguf_direct_100.py` represent isolated stress tests across model checkpoints.
- **Visualization**: `export_raw_graphs.py` synthesizes ALD chemistry flows and crystal phase transitions into visualizations, mapping relationships like Precursor $\to$ Co-reactant $\to$ Crystal Phase distributions. `run_qwen_lda.py` clusters semantic topic analysis over the corpus. 

## Requirements
To execute this supplementary code, refer to the environments and library versions listed in `/ald-extract/agentic_rag/requirements.txt`. Execution requires PyTorch, Transformers, LangChain, and other dependent ML/NLP tooling. Appropriate model weights (e.g. Qwen2.5-VL/Qwen3 or Llama-3.1) must be loaded locally or via an endpoint (such as Ollama or HuggingFace) according to the configured variables.
