# Hallucination Detector 2.0

Detects and flags hallucinated claims in LLM-generated text by grounding output against authoritative sources using four independent signals — NLI classification, self-consistency, vector similarity, and embedding similarity.

Domain-aware: automatically routes medical claims to PubMed, legal claims to CourtListener, and general claims to Wikipedia. Each domain uses a specialized embedding model for higher precision.

---

## Output verification (5/5 tests passing)

| Test | Expected domain | Got | Expected label | Got |
|------|----------------|-----|----------------|-----|
| Einstein mixed (1 hallucinated + 1 true) | general | general ✓ | mixed | mixed — 1/2 hallucinated ✓ |
| Einstein fully grounded | general | general ✓ | grounded | mostly grounded ✓ |
| Satya Nadella won Nobel Prize | general | general ✓ | hallucinated | likely hallucinated ✓ |
| Amoxicillin antibiotic claim | medical | medical ✓ | grounded | mostly grounded ✓ |
| Fourth Amendment | legal | legal ✓ | grounded | mostly grounded ✓ |

Source URLs are surfaced per sentence for manual verification:
- Wikipedia: `https://en.wikipedia.org/wiki/Albert_Einstein`
- PubMed: `https://pubmed.ncbi.nlm.nih.gov/31811919/`
- CourtListener: `https://www.courtlistener.com/opinion/2550/...`

---

## Architecture

```
INPUT TEXT
    │
    ▼
┌─────────────────────────┐
│   Domain Classifier     │  BART-MNLI zero-shot → medical / legal / financial / general
│   (domain_classifier)   │  Keyword sanity check prevents entity-association misfires
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Topic Inference       │  Gemini extracts the best Wikipedia/source article title
│   (fetcher)             │  3-tier fallback: exact → search → short query
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Domain Fact Source    │  Routes to authoritative source for the domain:
│   (sources/)            │  medical → PubMed E-utilities (peer-reviewed abstracts)
│                         │  legal   → CourtListener (actual court opinions)
│                         │  general/financial → Wikipedia
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Domain Embedder       │  Selects the right embedding model:
│   (domain_embedder)     │  medical   → NeuML/pubmedbert-base-embeddings (768-dim, local)
│                         │  legal     → law-ai/InLegalBert (768-dim, local)
│                         │  financial → ProsusAI/finbert (768-dim, local)
│                         │  general   → Gemini gemini-embedding-001 (3072-dim, API)
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   ChromaDB Collection   │  Facts embedded in 1 batch call and stored in-memory
│   (vector_store)        │  Namespaced by domain to prevent dimension collisions
└────────┬────────────────┘
         │
         ├──────────────── per sentence ────────────────────┐
         ▼                                                  │
┌────────────────────────────────────────────────────────┐  │
│                    SCORING ENGINE                      │  │
│                                                        │  │
│  NLI (0.40)          Consistency (0.35)                │  │
│  ─────────────       ────────────────────              │  │
│  Gemini labels       Gemini Standard + Adversarial     │  │
│  each source fact    + BART-MNLI weighted vote         │  │
│  as ENTAILMENT /     All 3 independently assess        │  │
│  NEUTRAL /           TRUE / FALSE / UNCERTAIN          │  │
│  CONTRADICTION       → consistency_score               │  │
│  → nli_score                                           │  │
│                                                        │  │
│  Grounding (0.15)    Embedding (0.10)                  │  │
│  ─────────────       ──────────────                    │  │
│  1/(1+distance)      cosine_similarity                 │  │
│  from ChromaDB       (claim_vec, fact_vec)             │  │
│  → grounding_score   → embedding_score                 │  │
│                                                        │  │
│  final = nli*0.40 + consistency*0.35                   │  │
│        + grounding*0.15 + embedding*0.10               │  │
│                                                        │  │
│  Hard caps:                                            │  │
│    CONTRADICTION → cap at 0.25                         │  │
│    All-FALSE consistency → cap at 0.30                 │  │
│                                                        │  │
│  label = "hallucinated" if final < 0.45                │  │
└────────────────────────────────────────────────────────┘  │
         └──────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   FastAPI Backend       │  POST /analyze → JSON
│   (api.py)              │  Sentence-level scores + source URLs
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Frontend              │  Sentences highlighted red/green
│   (frontend/)           │  Click → evidence + source URL + consistency trace
└─────────────────────────┘
```

---

## Project structure

```
hallucination-detector/
├── .env                        ← API keys — never commit
├── .env.example                ← template (commit this)
├── .gitignore
├── requirements.txt
├── Procfile                    ← Railway: web: uvicorn api:app --host 0.0.0.0 --port $PORT
├── runtime.txt                 ← python-3.11
├── api.py                      ← FastAPI backend
├── core/
│   ├── __init__.py
│   ├── embedder.py             ← embed() + embed_batch() + cosine_similarity()
│   ├── domain_embedder.py      ← domain-specific model routing + batch support
│   ├── domain_classifier.py    ← BART-MNLI zero-shot + Gemini fallback
│   ├── fetcher.py              ← infer_topic() + fetch_facts() with 3-tier fallback
│   ├── vector_store.py         ← build_collection() + retrieve_closest()
│   ├── consistency.py          ← multi-model consensus (Gemini ×2 + BART-MNLI)
│   ├── nli.py                  ← NLI classification + batch with retry
│   ├── scorer.py               ← orchestration layer
│   └── sources/
│       ├── __init__.py         ← domain router
│       ├── wikipedia_source.py ← Wikipedia adapter
│       ├── pubmed_source.py    ← PubMed E-utilities
│       └── legal_source.py     ← CourtListener opinions
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Verify you're in the right environment
which python
# → /path/to/project/.venv/bin/python

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env with your API keys
echo "GEMINI_API_KEY=your_key_here" >> .env
echo "HF_TOKEN=your_token_here" >> .env        # optional — higher HF rate limits

# 4. Run a test
python3 -m core.scorer
```

> **VS Code tip:** Use the terminal directly rather than the Run button. If modules aren't detected, open Command Palette → `Python: Select Interpreter` → enter the full path to `.venv/bin/python`. If imports still fail, try `Restart Language Server` or `Reload Window`.

---

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Gemini API (embeddings, NLI, consistency, topic inference) |
| `HF_TOKEN` | No | HuggingFace token — increases BART-MNLI rate limits |
| `COURTLISTENER_TOKEN` | No | CourtListener token — higher rate limits for legal domain |

---

## Domain routing

The pipeline classifies input text into one of four domains before fetching facts. This determines both the fact source and the embedding model.

| Domain | Classifier trigger | Fact source | Embedding model | Dim |
|---|---|---|---|---|
| medical | Clinical vocabulary (patient, drug, infection, mg...) | PubMed abstracts | NeuML/pubmedbert-base-embeddings | 768 |
| legal | Legal vocabulary (court, defendant, amendment, statute...) | CourtListener opinions | law-ai/InLegalBert | 768 |
| financial | Financial vocabulary (stock, revenue, EBITDA, basis points...) | Wikipedia | ProsusAI/finbert | 768 |
| general | Fallback — no strong domain signal | Wikipedia | Gemini gemini-embedding-001 | 3072 |

Domain-specific local models download automatically on first run (~440MB each) and cache to `~/.cache/huggingface/hub/`.

Pre-warm before a demo to avoid cold-start delay:
```bash
python3 -m core.domain_embedder
```

---

## Scoring

Each sentence in the input text receives four sub-scores, then a weighted final score:

| Signal | Weight | How it works |
|---|---|---|
| NLI | 0.40 | Gemini classifies each source fact as ENTAILMENT / NEUTRAL / CONTRADICTION against the claim. Hard cap: CONTRADICTION → final ≤ 0.25 |
| Self-consistency | 0.35 | Three independent models vote TRUE/FALSE/UNCERTAIN: Gemini Standard (0.40), Gemini Adversarial (0.25), BART-MNLI (0.35). Weighted average of TRUE confidence scores |
| Grounding | 0.15 | `1 / (1 + ChromaDB_distance)` — how close the nearest source fact is in vector space |
| Embedding similarity | 0.10 | Cosine similarity between claim vector and closest fact vector, normalized to [0,1] |

`final < 0.45` → `hallucinated`. Otherwise → `grounded`.

Overall label logic:
- All sentences hallucinated → `"likely hallucinated"`
- Some hallucinated → `"mixed — N/M sentence(s) hallucinated"`
- None hallucinated → `"mostly grounded"`

---

## API reference

**`POST /analyze`**

Request body:
```json
{ "text": "Einstein won the Nobel Prize for the theory of relativity." }
```

Response:
```json
{
  "topic": "Albert Einstein",
  "domain": "general",
  "overall_score": 0.56,
  "overall_label": "mixed — 1/2 sentence(s) hallucinated",
  "sentence_count": 2,
  "hallucinated_count": 1,
  "grounded_count": 1,
  "results": [
    {
      "sentence": "Einstein won the Nobel Prize for the theory of relativity.",
      "final_score": 0.1839,
      "label": "hallucinated",
      "nli_verdict": "CONTRADICTION",
      "evidence": "Einstein was awarded the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect.",
      "evidence_source": "Wikipedia",
      "evidence_url": "https://en.wikipedia.org/wiki/Albert_Einstein",
      "consistency_score": 0.003,
      "grounding_score": 0.42,
      "embedding_score": 0.71,
      "contradicting_fact": "Einstein was awarded the 1921 Nobel Prize..."
    }
  ]
}
```

---

## Dependencies

```
google-genai           Gemini API (embeddings, NLI, consistency, topic inference)
transformers           Local BERT-family models for domain-specific embeddings
torch                  Model inference (MPS on Apple Silicon, CUDA on NVIDIA, CPU fallback)
chromadb               In-memory vector database for fact retrieval
wikipedia-api          Wikipedia article fetching
requests               HTTP client (PubMed, CourtListener, BART-MNLI)
fastapi + uvicorn      REST API backend
python-dotenv          .env file loading
numpy                  Cosine similarity computation
```

Install all:
```bash
pip install -r requirements.txt
```

---

## Known limitations

- **Free-tier Gemini quota (15 RPM):** Running all 5 test cases consecutively hits the quota around test 4–5. The pipeline handles this with exponential backoff and fallback weight redistribution — it degrades gracefully but slows down. Upgrade to a paid tier or add `time.sleep()` between tests.
- **ChromaDB is in-memory:** Collections reset every run. This is intentional — stale cached facts would cause incorrect grounding. Repeated scoring of the same topic within one session reuses the collection efficiently.
- **Domain models are large:** The three local BERT models total ~1.3GB on first download. Subsequent runs load from disk cache in 2–5 seconds each.
- **CourtListener snippets are sometimes procedural:** The legal source filters case metadata noise, but very niche legal topics may still return few substantive sentences. The pipeline falls back to Wikipedia automatically if fewer than 2 facts are extracted.
- **NLI NEUTRAL on off-topic facts:** When the domain classifier routes correctly but the fetched facts don't directly address the specific claim (e.g. a claim about a specific Nobel Prize fact, where Wikipedia returns biographical facts), NLI returns NEUTRAL. The consistency signal carries the hallucination detection in these cases.

---

## Deployment

**Backend — Railway:**
```
Procfile:    web: uvicorn api:app --host 0.0.0.0 --port $PORT
runtime.txt: python-3.11
```
Set `GEMINI_API_KEY` in Railway environment variables.

**Frontend — Vercel:**
Deploy the `frontend/` folder. Update the API base URL in `script.js` to your Railway deployment URL.
