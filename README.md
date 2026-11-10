# GroundForge Retrieval Engine

A document retrieval engine prototype built to parse structured PDFs, run hybrid search, and surface accurate context for AI applications.

Documents are parsed using Docling to capture text and structural layout, chunked with contextual overlap, and stored in PostgreSQL with both vector and keyword indexes. Retrieval runs semantic similarity and exact keyword search in parallel, blends candidate results using Reciprocal Rank Fusion (RRF), and scores the top results with a Cross-Encoder reranker.

> **Looking for the lightweight flagship?** GroundForge serves as the deep-parsing prototype designed to evaluate layout-aware document ingestion. If you need an ultra-lightweight, zero-Torch retrieval sidecar optimized for autonomous AI agents (~300 MB RAM and sub-50ms latency), check out **[DocPilot](https://github.com/imLeo007/docpilot)**.

---

## Skill Stack

<p align="left">
<img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL" />
<img src="https://img.shields.io/badge/SQLAlchemy-D71F00?logo=sqlalchemy&logoColor=white" alt="SQLAlchemy" />
<img src="https://img.shields.io/badge/Alembic-6BA81E?logoColor=white" alt="Alembic" />
<img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker" />
<img src="https://img.shields.io/badge/Docling-1F6FEB?logoColor=white" alt="Docling" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
<img src="https://img.shields.io/badge/SentenceTransformers-FFD21E?logo=huggingface&logoColor=black" alt="Sentence Transformers" />
<img src="https://img.shields.io/badge/DeepEval-10B981?logoColor=white" alt="DeepEval" />
</p>

---

## How It Works

GroundForge processes uploaded documents and delivers relevant context through a five-stage retrieval pipeline:

```mermaid
flowchart LR
    subgraph Ingestion ["1. Document Ingestion"]
        A[PDF Upload] --> B[Docling Parser]
        B --> C[Sliding Window Chunking]
        C --> D[Vector Embeddings]
        D --> E[(PostgreSQL: HNSW + GIN Indexes)]
    end
```

```mermaid
flowchart LR
    subgraph Retrieval ["2. Hybrid Search & Reranking"]
        Q[Search Query] --> F[Vector Search: pgvector HNSW]
        Q --> G[Keyword Search: Postgres GIN]
        F --> H[Reciprocal Rank Fusion - RRF]
        G --> H
        H --> I[Cross-Encoder Reranker]
        I --> J[Top-K Ranked Chunks]
    end
```

### 1. Layout-Aware Ingestion
Uploaded documents are processed through Docling to extract clean body text and recognize document structure instead of performing basic character dumps.

### 2. Context-Preserving Chunking
Extracted text is split into uniform passages using sliding window overlap. This ensures key sentences are not cut in half across chunk boundaries and preserves surrounding context.

### 3. Dual-Path Hybrid Search
Every query triggers two independent searches in parallel:
* **Semantic Search:** Uses `pgvector` with HNSW indexing to find passages based on conceptual meaning.
* **Lexical Search:** Uses PostgreSQL full-text search (`tsvector` + GIN indexing) to lock onto exact numbers, acronyms, and proper nouns.

### 4. Reciprocal Rank Fusion (RRF)
The top candidates from both search paths are combined based on their relative ranking positions rather than raw scores. This balances vector and keyword relevance without score calibration issues.

### 5. Precision Reranking
The fused candidate pool is passed through a Cross-Encoder reranker. The model inspects the question and candidate text together, and the top-ranked passages are returned as clean, ready-to-use context.

---

### Workflow Architecture
![DocPilot Pipeline](screenshots/workflow.png)
*The end-to-end flow from document upload to dual-index storage, hybrid retrieval, and final reranking.*

### Interactive API Preview
![Swagger UI](screenshots/preview.png)
*FastAPI interactive documentation interface for uploading documents and running searches.*

### Benchmark Evaluation
![Evaluation Results](screenshots/benchmark.png)
*Automated evaluation run using DeepEval measuring contextual retrieval performance.*

---

## Performance & Evaluation

GroundForge was evaluated using an automated benchmark suite powered by **DeepEval** against a **100-question dataset** built from standard public curriculum textbooks.

An independent LLM judge evaluated the factual completeness and ranking order of retrieved chunks against reference answers:

### Benchmark Results (100 Questions)

| Metric | Average Score | Pass Rate | Status | Total Evaluated |
| :--- | :---: | :---: | :---: | :---: |
| **Contextual Recall** | **0.98** | **96.00%** | 96 passed / 4 failed | 100 test cases |
| **Contextual Precision** | **0.90** | **92.00%** | 92 passed / 8 failed | 100 test cases |

* **Contextual Recall (96%):** Confirms that in 96 out of 100 test queries, the engine successfully surfaced all necessary factual information in the retrieved chunks.
* **Contextual Precision (92%):** Confirms that high-relevance chunks were prioritized at the top of the context list.

## Architectural Comparison: GroundForge vs. DocPilot

| Aspect | GroundForge (This Project) | DocPilot (Production Flagship) |
| :--- | :--- | :--- |
| **Primary Goal** | Deep document parsing prototype | Ultra-lightweight agent retrieval sidecar |
| **PDF Extraction** | Docling (layout & structure aware) | PyMuPDF (compiled C-speed extraction) |
| **Model Runtime** | Native PyTorch (CPU) | Pure ONNX Runtime (Zero-Torch dependency) |
| **Memory Footprint** | ~1.8 GB – 2.4 GB RAM | **~280 MB – 320 MB RAM** |
| **Retrieval Speed** | Standard (~150ms – 250ms) | **Sub-50ms** |
| **Recommended For** | Layout-heavy documents, deep parsing research | Fast, low-memory tool for Agno & LangGraph |

---

## Project Structure

```
groundforge/
├── app/
│   ├── ai/
│   │   ├── embedding.py       # SentenceTransformer vector embeddings
│   │   ├── reranker.py        # CrossEncoder scoring model
│   │   └── retriever.py       # Hybrid search (pgvector + tsvector) & RRF
│   ├── router/
│   │   ├── document.py        # Document upload and parsing endpoints
│   │   └── question.py        # Question retrieval endpoints
│   ├── core/
│   │   ├── config.py          # Environment settings
│   │   └── database.py        # Database initialization
│   ├── models/
│   │   ├── chunks.py          # Dual-indexed chunk table (HNSW + GIN)
│   │   └── document.py        # Document metadata table
│   ├── schemas/
│   │   ├── document.py        # Document pydantic Schemas
│   │   └── retriever.py       # Retriever pydantic schemas
│   ├── services/
│   │   ├── chunker.py         # Chunk page text
│   │   ├── document.py        # Document fetching services
│   │   └── pdf.py             # Docling parser integration
│   └── main.py                # FastAPI entrypoint
├── evaluation/
│   ├── dataset.json           # 100-question ground truth dataset
│   ├── evaluator.py           # DeepEval precision and recall runner
│   └── judge.py               # Custom structured LLM judge
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/imLeo007/groundforge-rag
cd groundforge-rag
```

### 2. Configure environment variables

Your `.env` file should include the following variables (referenced from `.env.example`):
```env
# Database Settings
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/docpilot_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=docpilot_db
POSTGRES_PORT=5432

# Model Configurations
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Evaluation / Judge API Settings (Optional for evaluation runs)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 3. Build and start services

```bash
docker compose up --build -d
```

### 4. Run database migrations

```bash
docker compose exec api alembic upgrade head
```

### 5. Access the API

Open your browser to:
```
http://localhost:8000/docs
```

* **Upload a Document:** Use `POST /document/upload` to parse and index your PDF.
* **Search Context:** Use `POST /question/search` to retrieve ranked chunks using hybrid search and reranking.

---

## Conclusion

GroundForge was developed as a hands-on retrieval prototype to test deep document parsing, hybrid indexing, and multi-stage reranking on complex files. 

While GroundForge handles structural document analysis reliably, the insights gained from its compute footprint directly inspired **DocPilot**—our streamlined, sub-300MB, zero-Torch retrieval engine engineered specifically for fast AI agent workflows.