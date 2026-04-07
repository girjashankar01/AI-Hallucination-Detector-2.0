# 🕵️‍♂️ Hallucination Detector 2.0

## 📋 Project Submission

**Project Name**: Hallucination Detector 2.0  
**Team Members**: Althea  
**Track**: AI Verification & Fact-Checking  

---

## 🚀 Project Overview

**Hallucination Detector 2.0** is a domain-aware AI hallucination detection system that verifies LLM-generated text against authoritative sources using 4 independent signals. It solves the critical problem of fact-checking AI output by intelligently routing claims to the most appropriate knowledge base and evaluating them dynamically.

Instead of relying on a single general-purpose database, the system classifies the domain of the input (Medical, Legal, Financial, or General) and grounds the facts against specialized sources like PubMed, CourtListener, and Wikipedia. 

---

## 🏗️ Architecture & Pipelines

**Core Pipeline:** 
`Input Text → Domain Classifier → Topic Inference → Fact Fetch → Embed → Vector Store (ChromaDB) → Score → API Response → Frontend`

### 1. Domain Routing & Models Used
The pipeline classifies input text using zero-shot NLI and keyword checks before fetching facts. This determines both the fact source and the embedding model.

| Domain | Classifier Trigger | Fact Source | Embedding Model | Dimensions |
|---|---|---|---|---|
| **Medical** | Clinical vocabulary (patient, drug, infection, etc.) | PubMed E-utilities | `NeuML/pubmedbert-base-embeddings` | 768 |
| **Legal** | Legal vocabulary (court, defendant, statute, etc.) | CourtListener | `law-ai/InLegalBert` | 768 |
| **Financial** | Financial vocabulary (stock, revenue, EBITDA, etc.) | Wikipedia | `ProsusAI/finbert` | 768 |
| **General** | Fallback (no strong domain signal) | Wikipedia | Gemini `gemini-embedding-001` | 3072 |

*(Note: Domain-specific local models download automatically on first run, totaling ~1.3GB, and cache to `~/.cache/huggingface/hub/`.)*

### 2. Scoring System Engine
Each sentence in the input text receives four sub-scores, combined into a final weighted score. 
*If the final score < 0.45, the sentence is flagged as "hallucinated".*

- **NLI (40%)**: Gemini strictly classifies each retrieved source fact as `ENTAILMENT`, `NEUTRAL`, or `CONTRADICTION` against the original claim. (Hard cap: `CONTRADICTION` → final score ≤ 0.25).
- **Self-Consistency (35%)**: Three independent models vote `TRUE`, `FALSE`, or `UNCERTAIN`: Gemini Standard (40%), Gemini Adversarial (25%), BART-MNLI (35%). It averages the `TRUE` confidence scores. (Hard cap: All-`FALSE` → final score ≤ 0.30).
- **Grounding (15%)**: Computed as `1 / (1 + ChromaDB_Vector_Distance)`. Measures how close the nearest fetched source fact is in the vector space.
- **Embedding Similarity (10%)**: Cosine similarity between the claim vector and the closest fact vector, normalized.

---

## 📂 Project Structure

```text
hallucination-detector/
├── .env                        ← API keys — never commit
├── .env.example                ← Template (commit this)
├── requirements.txt            ← Project dependencies
├── Procfile                    ← Railway deployment config
├── runtime.txt                 ← Python version (3.11)
├── api.py                      ← FastAPI backend entrypoint
├── frontend/                   ← Served static files
│   ├── index.html              ← Main UI markup
│   ├── style.css               ← Dark mode & responsive styling
│   └── script.js               ← Logic, API fetching, real-time UI updates
└── core/
    ├── __init__.py
    ├── scorer.py               ← Orchestration layer integrating all modules
    ├── embedder.py             ← embed(), embed_batch(), cosine_similarity()
    ├── domain_embedder.py      ← Domain-specific model routing + batch support
    ├── domain_classifier.py    ← BART-MNLI zero-shot + Gemini fallback
    ├── fetcher.py              ← infer_topic() + fetch_facts() w/ fallbacks
    ├── vector_store.py         ← ChromaDB build_collection() & retrieval
    ├── consistency.py          ← Multi-model consensus voting (Gemini ×2 + BART)
    ├── nli.py                  ← NLI classification + batch processing
    └── sources/                ← Domain fact adapters
        ├── wikipedia_source.py 
        ├── pubmed_source.py    
        └── legal_source.py     
```

---

## 🛠️ Python Modules & Dependencies

Key libraries utilized in this project, detailed in `requirements.txt`:

- **`fastapi`** + **`uvicorn`**: High-performance REST API backend.
- **`google-genai`**: Interacting with the Gemini API for NLI, consistency, topic inference, and general embeddings.
- **`transformers`** + **`torch`**: Local BERT-family models for domain-specific inference. Uses MPS on Apple Silicon, CUDA on NVIDIA, or CPU fallback.
- **`chromadb`**: In-memory vector database for extremely fast fact retrieval.
- **`wikipedia-api`**: Wikipedia article fetching.
- **`requests`**: HTTP client for external sources (PubMed, CourtListener) and HuggingFace API.
- **`python-dotenv`**: Secure environment variable loading.
- **`numpy`**: Advanced tensor handling and cosine similarity computation.

---

## 💻 Replicating & Running Locally

Follow these precise steps to set up and run the application on your local machine using a Python Virtual Environment.

### 1. Prerequisites
Ensure you have Python 3.11+ installed.

### 2. Clone and Setup Virtual Environment
It is highly recommended to run this inside an isolated virtual environment to prevent dependency conflicts.
```bash
# Clone the repository
git clone <repository_url>
cd hallucination-detector

to create the virtual environment in python version-3.11.9
~/.pyenv/versions/3.11.9/bin/python -m venv .venv

# Create a virtual environment named '.venv'
python3 -m venv .venv

# Activate the virtual environment:
# On macOS and Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Verify you are in the correct isolated environment
which python
# Should output: /path/to/project/.venv/bin/python
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
You must provide required API keys. Copy the example file and edit it.
```bash
cp .env.example .env
```
Inside `.env`, configure the following:
- `GEMINI_API_KEY` (Required): Gemini API used for core logic.
- `HF_TOKEN` (Optional): HuggingFace token — increases BART-MNLI rate limits.
- `COURTLISTENER_TOKEN` (Optional): CourtListener token — increases rate limits for legal domain fact-finding.

### 5. Running the Application
**Pre-download Models (Optional but recommended):**
To avoid cold-start delays on the first API request, manually trigger the download of the local BERT models (~1.3GB):
```bash
python3 -m core.domain_embedder
```

**Start the Server:**
Use Uvicorn to run the FastAPI backend.
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
The application will be accessible at: `http://localhost:8000`

> **Note for VS Code Users:** Use the integrated terminal to run the commands. Ensure your Python Interpreter is set to the `.venv` path (`Cmd/Ctrl + Shift + P` -> `Python: Select Interpreter`).

---

## ✅ Pre-Submission Checklist

- [x] **Code Runs**: API and backend run locally executing without errors.
- [x] **Dependencies**: All external libraries are rigidly frozen in `requirements.txt`.
- [x] **Environment**: Provided a `.env.example` mapping out necessary API keys.
- [x] **Screenshots**: High-quality dark-mode UI screenshots captured for verification.
- [x] **Demo Instructions**: Complete README setup detailing the deployment on any local machine.

---

## 🌐 Deployment Details
- **Backend**: Pre-configured for deployment on **Railway** using the included `Procfile` (`web: uvicorn api:app --host 0.0.0.0 --port $PORT`).
- **Frontend**: The `frontend/` folder acts as a static SPA and can be deployed directly via **Vercel** or **Netlify**. Ensure `script.js` API endpoints are configured optimally to point to your live backend domain.
