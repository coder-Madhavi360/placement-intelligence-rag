# Placement Intelligence Assistant

> An enterprise-style Placement Intelligence Assistant built using Hybrid Retrieval, Safety-Aware Generation, Tool-Augmented Reasoning, Vision Processing, and Automated Evaluation.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama--3.1-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green?style=flat-square)
![BM25](https://img.shields.io/badge/Sparse-BM25-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Project Overview

Placement Intelligence Assistant is a multimodal RAG system that helps students explore placement information using natural language queries.

The system combines:

- Hybrid Search (FAISS + BM25)
- CrossEncoder Reranking
- Vision-Based Chart Understanding
- Tool-Augmented Query Routing
- Multi-Hop Reasoning
- Hallucination Prevention
- Automated Evaluation Framework

It can answer questions related to eligibility, hiring trends, package analysis, interview experiences, company comparisons, and placement statistics.

---

## Why This Project?

Placement reports contain:

| Content Type | Challenge |
|-------------|------------|
| Tables | Difficult to retrieve accurately |
| Charts | Invisible to traditional RAG systems |
| Interview Experiences | Long unstructured text |
| Trend Data | Requires year-wise reasoning |
| Conflicting Records | Multiple sources may disagree |

To solve these challenges, this project uses:

- Content-aware chunking
- Hybrid retrieval
- CrossEncoder reranking
- Vision-powered chart extraction
- Conflict detection
- Tool routing for external queries
- Hallucination guards

The result is a reliable placement intelligence assistant capable of answering both simple and complex placement-related questions.

---

## Features

- **Hybrid Retrieval Engine** — Combines FAISS semantic search and BM25 keyword search for better retrieval accuracy.

- **Query Rewriting** — Expands user queries into multiple search variations to improve recall.

- **CrossEncoder Reranking** — Reorders retrieved chunks to surface the most relevant information.

- **Multimodal Chart Understanding** — Converts placement charts and graphs into searchable text using vision models.

- **Content-Aware Chunking** — Applies different chunking strategies for tables, narratives, trends, and chart data.

- **Context Compression** — Removes redundant context before generation to improve answer quality.

- **Conflict Detection System** — Identifies contradictory information from multiple data sources.

- **Hallucination Prevention Layer** — Uses multiple safeguards to reduce unsupported responses.

- **Tool Routing Framework** — Automatically routes real-time or external queries to specialized tools.

- **Eligibility Analysis Tool** — Assists with CGPA-based eligibility and placement filtering.

- **Temporal Reasoning Support** — Enables year-wise trend and growth analysis.

- **Multi-Hop Question Answering** — Combines information from multiple document sections.

- **Answer Verification Pipeline** — Validates generated responses before presenting them.

- **Persistent Chat History** — Maintains conversation history across application sessions.

- **Evaluation Dashboard** — Measures retrieval and generation quality using benchmark queries.

- **Performance Monitoring** — Tracks retrieval quality, latency, and system behavior.

- **Modular SOLID Architecture** — Each component is independent, reusable, and easy to extend.

---

## Technologies Used

| Layer | Technology | Purpose |
|---------|------------|----------|
| Document Parsing | Docling + pdfplumber | PDF extraction and table processing |
| Embeddings | all-MiniLM-L6-v2 | Dense vector generation |
| Vector Search | FAISS | Semantic retrieval |
| Sparse Search | BM25 | Keyword retrieval |
| Fusion | Reciprocal Rank Fusion | Hybrid search ranking |
| Reranking | CrossEncoder | Retrieval refinement |
| Vision Model | Groq Llama-4-Scout | Chart understanding |
| LLM | Groq Llama-3.1 | Answer generation |
| Web Search | DuckDuckGo | External information retrieval |
| UI | Streamlit | Interactive interface |
| Language | Python | Core implementation |

---

## Project Structure

```
placement-rag/
│
├── README.md                        # This file
├── requirements.txt                 # All Python dependencies
├── .env.example                     # Environment variable template
├── .gitignore                       # Excludes venv, indexes, secrets
├── version.py                       # Project metadata and banner
├── app.py                           # Streamlit UI — main entry point
│
├── core/                            # SOLID abstract interfaces
│   ├── interfaces.py                # ABC classes for every pipeline stage
│   └── pipeline.py                  # 6-stage RAG orchestrator + tool router
│
├── ingestion/                       # Stage 0: Parse → Chunk → Embed
│   ├── parser.py                    # Docling + pdfplumber + Groq vision
│   ├── chunker.py                   # Section-10 chunking strategy
│   ├── deduplicator.py              # TF-IDF cosine deduplication
│   └── embedder.py                  # FAISS + BM25 hybrid index builder
│
├── retrieval/                       # Stages 1–3
│   ├── rewriter.py                  # Query expansion (2–4 variants)
│   ├── retriever.py                 # Hybrid RRF + metadata boosting
│   └── reranker.py                  # CrossEncoder reranking
│
├── generation/                      # Stages 4–6
│   ├── refiner.py                   # Context pruning wrapper
│   ├── prompt_builder.py            # Grounded prompt with query instructions
│   └── generator.py                 # Groq Llama-3.1 with self-consistency
│
├── safety/                          # Hallucination prevention
│   ├── conflict_detector.py         # Official vs portal conflict detection
│   ├── fallback_guard.py            # Out-of-corpus detection
│   └── overshadow_limiter.py        # AIMD binary-search context cap
│
├── tools/                           # Tool-augmented agent
│   ├── base_tool.py                 # Abstract tool interface
│   ├── tool_router.py               # Query classifier and dispatcher
│   ├── web_search_tool.py           # DuckDuckGo search
│   ├── calculator_tool.py           # Eligibility filtering and ratios
│   ├── date_tool.py                 # Current date queries
│   └── opinion_guard_tool.py        # Objective comparison for subjective queries
│
├── evaluation/                      # 34-query evaluation pipeline
│   ├── queries.py                   # 30 official + 4 multi-hop queries
│   ├── metrics.py                   # Retrieval quality metrics
│   └── evaluator.py                 # Automated scorer with visual summary
│
├── feedback/                        # Background AIMD feedback loop
│   └── loop.py                      # Feedback controller (silent)
│
├── scripts/
│   ├── ingest.py                    # One-time ingestion pipeline
│   └── evaluate.py                  # Run full 34-query evaluation
│
└── data/
    ├── README.md                    # Data folder documentation
    └── Placement_RAG_Dataset_Enhanced.pdf   ← place here
```

---

## Quick Start

### Prerequisites

| Requirement | Version |
|------------|-----------|
| Python | 3.11+ |
| Git | Latest |
| Groq API Key | Required |

---

### 1. Clone Repository

```bash
git clone https://github.com/madhavi-b24/placement-intelligence-rag.git
cd placement-intelligence-rag
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
GROQ_MODEL=llama-3.1-8b-instant
```

---

### 5. Add Dataset

Place the dataset PDF inside:

```text
data/
└── Placement_RAG_Dataset_Enhanced.pdf
```

---

### 6. Build Retrieval Index

```bash
python scripts/ingest.py
```

This step:

- Parses PDF content
- Extracts tables
- Processes chart images
- Creates chunks
- Generates embeddings
- Builds BM25 + FAISS indexes

---

### 7. Launch Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

### 8. Run Evaluation Suite

```bash
python scripts/evaluate.py
```

Generated files:

```text
data/eval_results.csv
data/multihop_results.csv
```

---

## How to Use the Project

### Asking questions

Type any placement-related question in the input box and click **Ask**. The system retrieves relevant chunks, reranks them, and generates a grounded answer.

### Quick test queries

The sidebar contains pre-loaded queries covering every difficulty level. Click any of them to instantly populate the input box and run the query.

### Reading the answer

| Box colour | Meaning |
|---|---|
| Green | Standard grounded answer from documents |
| Red | Conflicting data detected between official and portal sources |
| Yellow | Out-of-corpus — information not available in the documents |

### Diagnostics

Expand the **Retrieval Diagnostics** section under any answer to see retrieval quality score, context tokens, overshadow risk, and self-consistency score.

### Chat history

All previous questions appear below the current answer, newest first. The sidebar also shows a clickable history list. Reloading the page preserves history — it is saved automatically to `data/chat_history.json`.

### Response time

Each answer shows its response time. Repeated questions show near-zero response time because they are served from the 1-hour cache — no extra API call is made.

### Running the evaluation suite

```bash
python scripts/evaluate.py
```

Runs all 34 queries and saves results to:

```
data/eval_results.csv
data/multihop_results.csv
```

---

## RAG Pipeline — 6 Stages

| Stage | Name | What it does |
|---|---|---|
| 1 | **Rewrite** | Expands query into 2–4 variants for better recall |
| 2 | **Retrieve** | Hybrid BM25 + FAISS with RRF fusion and metadata boosting |
| 3 | **Rerank** | CrossEncoder scores every (query, chunk) pair for precision |
| 4 | **Refine** | AIMD overshadow limiter caps context to 500–1500 token sweet spot |
| 5 | **Insert** | Grounded prompt built with query-type specific instructions |
| 6 | **Generate** | Groq Llama-3.1 generates answer with self-consistency check |

Before Stage 1, a **Tool Router** classifies the query. Out-of-corpus queries are sent directly to tools instead of entering the RAG pipeline.

---

## Content Processing Strategy

The placement dataset contains multiple content formats including tables, interview narratives, hiring statistics, trend records, conflict entries, and chart images.

To maximize retrieval accuracy, each content type is processed using a dedicated chunking strategy.

| Content Source | Processing Method | Purpose |
|----------------|-------------------|----------|
| Eligibility Data | One company per chunk | Fast eligibility filtering |
| Interview Experiences | Round-wise paragraph chunking | Interview preparation queries |
| Hiring Statistics | Company-level chunks | Hiring trend analysis |
| Package Trends | Year-specific chunks | Temporal reasoning |
| Conflict Records | Dual-source preservation | Conflict detection |
| Chart Images | Vision-to-text conversion | Multimodal retrieval |
| Evaluation Queries | Excluded from indexing | Benchmark testing only |

### Metadata Attached To Chunks

| Metadata | Usage |
|-----------|---------|
| Company Name | Company-specific retrieval |
| Section Type | Context filtering |
| Year | Temporal analysis |
| Source | Conflict verification |
| Round Number | Interview reasoning |
| Chart Type | Chart-specific retrieval |

### Corpus Optimization

| Stage | Result |
|---------|---------|
| Raw Document Extraction | 400+ chunks |
| Content-Aware Chunking | Reduced redundancy |
| TF-IDF Deduplication | Removed duplicate content |
| Final Search Corpus | Optimized retrieval dataset |

This strategy improves retrieval precision, reduces context noise, and enables accurate multi-hop reasoning across different sections of the placement dataset.
---

## Tool-Augmented Intelligence Layer

To handle questions that cannot be answered directly from the placement dataset, the system routes queries through specialized tools before generating a final response.

| Tool | Function | Example Query |
|--------|----------|---------------|
| `web_search_tool` | Retrieves real-time information beyond the dataset | "When will TCS visit SVECW?" |
| `calculator_tool` | Computes ratios, eligibility, and placement statistics | "What is Google's package-to-CGPA ratio?" |
| `date_tool` | Handles current date and time calculations | "How many days until 31 Dec 2026?" |
| `opinion_guard_tool` | Provides objective comparisons for subjective queries | "Should I join Google or Microsoft?" |
| `answer_verifier_tool` | Validates generated answers against retrieved evidence | Internal verification |
| `context_compression_tool` | Reduces irrelevant context before generation | Internal optimization |
| `hallucination_scorer_tool` | Measures answer reliability and hallucination risk | Internal scoring |
| `tool_router` | Selects the appropriate tool based on query intent | Query classification |

---

## Evaluation Framework

The system is evaluated using a structured benchmark covering retrieval, reasoning, aggregation, and multi-hop questions.

| Category | Focus Area |
|------------|------------|
| Easy | Direct fact retrieval |
| Medium | Filtering and comparison queries |
| Hard | Aggregation and synthesis |
| Expert | Temporal and reasoning-intensive queries |
| Multi-Hop | Cross-section reasoning and joins |
| Out-of-Corpus | Tool routing and fallback handling |

### Evaluation Capabilities

- Direct Retrieval
- Eligibility Filtering
- Package Comparison
- Temporal Reasoning
- Aggregation Queries
- Multi-Hop Reasoning
- Conflict Detection
- Tool Invocation
- Hallucination Prevention

---



## Future Improvements

The current system provides a strong foundation for placement intelligence, but several enhancements can further improve its capabilities.

| Improvement | Benefit |
|------------|----------|
| Conversational Memory | Better multi-turn question answering |
| Advanced Agentic Workflows | Dynamic planning and tool orchestration |
| Real-Time Placement Updates | Automatic synchronization with latest placement information |
| Knowledge Graph Integration | Improved relationship-based reasoning |
| Multilingual Support | Support for regional languages and English |
| Voice-Based Interaction | Hands-free question answering experience |
| Advanced Analytics Dashboard | Visual insights into placement trends and statistics |
| Fine-Tuned Domain Model | Better understanding of placement-specific terminology |
| Feedback-Driven Learning | Continuous improvement using user feedback |
| Cloud Deployment | Scalable access for students and placement coordinators |
| Role-Based Access Control | Separate views for students, faculty, and administrators |
| Enhanced Evaluation Suite | Larger benchmark set with additional metrics |

### Long-Term Vision

- Build a complete AI-powered Placement Intelligence Platform.
- Support multiple colleges and placement datasets.
- Enable institution-wide analytics and reporting.
- Provide personalized placement guidance for students.
- Combine RAG, agents, tools, and analytics into a single ecosystem.
---


## Credits

Built by students of SVECW — Department of Information Technology for RAG-ATHON 24.

Dataset provided by the RAG-ATHON 24 organizing committee.

Libraries and tools that made this possible:

- [Docling](https://github.com/DS4SD/docling) — layout-aware PDF parsing
- [Groq](https://console.groq.com) — fast LLM and vision inference
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search by Meta
- [Sentence Transformers](https://www.sbert.net/) — embedding and cross-encoder models
- [Streamlit](https://streamlit.io) — UI framework
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF table extraction
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 sparse retrieval

---

## License

```
MIT License

Copyright (c) 2026 SVECW — Department of Information Technology

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.
```

---

> Built with care for RAG-ATHON 24 · SVECW · Department of Information Technology