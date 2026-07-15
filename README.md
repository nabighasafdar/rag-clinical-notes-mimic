# Clinical RAG System

A Retrieval-Augmented Generation (RAG) pipeline for clinical reasoning. Ask a medical question; the system retrieves relevant passages from structured clinical notes and disease knowledge graphs, then generates a grounded diagnostic explanation with a local LLM.

> **Disclaimer:** This project is for research and educational purposes only. It is **not** a medical device and must not be used for clinical decision-making or patient care.

---

## Features

- **Structured data ingestion** — parses clinical notes and diagnosis flowchart / knowledge-graph JSON into searchable text chunks with metadata
- **Semantic retrieval** — embeds chunks with [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) and indexes them in FAISS (`IndexFlatL2`)
- **Local generation** — answers with [Nous-Hermes-2-Mistral-7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO) in 4-bit quantization (bitsandbytes)
- **Interactive UI** — Gradio demo for querying the knowledge base and inspecting retrieved context
- **Retrieval evaluation** — Precision@K and Recall@K against disease-aware ground truth

---

## Architecture

```text
Clinical Notes (JSON)  ──┐
                         ├──► Chunking ──► BGE Embeddings ──► FAISS Index
Diagnosis KG (JSON)    ──┘                                      │
                                                                ▼
User Query ──► Embed ──► Top-K Retrieval ──► Prompt + Context ──► Hermes-2-Mistral ──► Answer
```

1. **Ingest** — walk note trees and knowledge graphs into word-level chunks (size 180, overlap 40)
2. **Index** — embed all chunks and persist `faiss_index.index` + `metadata.pkl`
3. **Retrieve** — embed the query and return the top-K nearest chunks
4. **Generate** — build an instruction prompt with retrieved clinical context and sample from the local LLM
5. **Serve** — optional Gradio interface for interactive use

---

## Tech Stack

| Component        | Choice                                      |
|------------------|---------------------------------------------|
| Embeddings       | `BAAI/bge-base-en-v1.5` (LangChain HF)       |
| Vector store     | FAISS (`faiss-cpu`)                         |
| LLM              | Nous-Hermes-2-Mistral-7B-DPO (4-bit NF4)    |
| Quantization     | bitsandbytes                                |
| UI               | Gradio                                      |
| Runtime          | Google Colab (T4 GPU recommended)           |

---

## Project Layout

```text
clinical-rag-system/
├── RAG_Project.ipynb   # End-to-end notebook (ingest → retrieve → generate → UI → eval)
└── README.md
```

---

## Data

The notebook expects two zip archives uploaded to Colab (`/content/`):

| Archive                   | Extracted path              | Role                                      |
|---------------------------|-----------------------------|-------------------------------------------|
| `Diagnosis_flowchart.zip` | `/content/Diagnosis_flowchart` | Disease knowledge / diagnostic flowcharts |
| `Finished.zip`            | `/content/Finished`            | Structured clinical note JSONs            |

Parsers extract:

- **Notes** — `input*` sections plus walked diagnostic tree nodes (with path metadata)
- **Knowledge graphs** — flattened disease knowledge strings tagged by disease name and path

---

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPU recommended (for the 7B LLM in 4-bit)
- Hugging Face access for model downloads

### Run in Google Colab

1. Open `RAG_Project.ipynb` in [Google Colab](https://colab.research.google.com/) (GPU runtime: T4 or better).
2. Upload `Diagnosis_flowchart.zip` and `Finished.zip` to `/content/`.
3. Run cells in order:
   - Install dependencies
   - Extract and parse JSON data
   - Build (or load) the FAISS index
   - Load the 4-bit generation model
   - Run example queries or launch Gradio

### Core dependencies

```bash
pip install transformers accelerate sentencepiece bitsandbytes \
  langchain langchain_huggingface faiss-cpu huggingface_hub gradio
```

---

## Usage Examples

Example queries exercised in the notebook:

- What are the symptoms of Type I Diabetes?
- Which patients have hypertension as a symptom?
- Describe the diagnostic process for stroke.
- How is Type I Diabetes diagnosed?
- What are the risk factors and symptoms of Alzheimer?

Each query returns:

1. **Retrieved chunks** — top-K passages with L2 distance scores  
2. **Generated answer** — diagnostic explanation conditioned on that context  

---

## Configuration

Key settings (defined in the notebook):

| Parameter          | Default                                  | Description                |
|--------------------|------------------------------------------|----------------------------|
| `EMBED_MODEL_NAME` | `BAAI/bge-base-en-v1.5`                   | Embedding model            |
| `MODEL_ID`         | `NousResearch/Nous-Hermes-2-Mistral-7B-DPO` | Generation model        |
| `CHUNK_SIZE`       | `180`                                    | Words per chunk            |
| `CHUNK_OVERLAP`    | `40`                                     | Overlap between chunks     |
| `TOP_K`            | `5`                                      | Chunks retrieved per query |
| `FAISS_INDEX_PATH` | `faiss_index.index`                      | Saved vector index         |
| `METADATA_PATH`    | `metadata.pkl`                           | Saved chunk metadata       |

---

## Evaluation

The notebook includes a lightweight retrieval eval:

- Builds ground-truth indices by matching query disease terms to chunk metadata
- Reports **Precision@K** and **Recall@K** per query, plus averages

This measures how well the retriever surfaces disease-relevant chunks—not clinical correctness of generated text.

---

## Limitations

- Paths are Colab-oriented (`/content/...`); adjust for local runs
- IndexFlatL2 is exact but may become slow at very large corpus sizes
- Generation quality depends on retrieved context and prompt design
- Outputs are **not** clinically validated

---

## License

This repository is provided as-is for learning and research. Ensure you have the right to use any clinical datasets and model weights you load.
