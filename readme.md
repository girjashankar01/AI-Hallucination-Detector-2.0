
## Basic Setup
- virtual python environment : `python3 -m venv .venv`
- wait for around 30 seconds then, `source .venv/bin/activate`
- verify via which python i.e in which locn your virtual env. is present; so you can use pip install in same locn without issues.
- virtual environment in python to ensure portability and that all the pip modules and python are in same path so they could work together and also prevent global contamination by installing several pip moudles
- it would return sth like : `/Users/althea/Developer/Projects/Hallucination-Detector/.venv/bin/python`
- first create a .env file and place your api key in that and put that file in gitignore;
- you should never hardcode API keys in source code
- Methods to use your api key without putting in src code
    1. `export GEMINI_API_KEY=xyz`
    python core/embedder.py  # os.getenv() works 
    but this is temporary, you have to export for each session
    2. using dotenv moudle: .env file can't be read directly 
    from dotenv import load_dotenv
    load_dotenv()  # reads .env, injects into os.environ
    os.getenv("GEMINI_API_KEY")  # → "xyz"
    Bash command `echo "GEMINI_API_KEY=your_key_here" >> .env`

## Module 1: embedder
- using gemini embedding model 001, it compares meaning of diff. words/sentences; 
- each words or sentence is split it into diff. tokens then each of them give diff. embedding(vector) in certain diamension like this one give vector of 3072 dimension; and then at end each combines (sth like weighted avg. of all token vectors) to give a final vector (embedding) that signifies the meaning of that sentence.
- using cosine similarity it determines, how close (angular seperation) are 2 sentences or words
- we compare the angle b/w them and return a value b/w [-1,1] for similarity score.

## Module 2: fetcher
- using wikipedia api, for fetching facts (summary) then splitting it into parts using (re)
- basic text split using (.) would split incorrectly so using re (regex split)
    import re
    re.split(r'(?<=[.!?]) +', text)
    This says: split on one or more spaces, but only when preceded by . ! or ?
    its not perfect but this will give about 90% accuracy, using another NLP for it would increase the accuracy but along with that complexity too.
- Token inference : to determine which wikipedia article to fetch.
- to fetch: wiki.page("article name e.g Albert Einstein"); the article name to fetch would be figured out by token inference
- Named Entity Recognition (NER) — identifying the main subject of text. Here using Gemini as a lazy NER system instead of a dedicated NLP model.

## Module 3 : vector_store.py
- What it does
    Stores Wikipedia facts as embeddings in ChromaDB (in-memory)
    Retrieves the N closest facts to a given claim using vector similarity
    Install
    pip install chromadb
- if you directly do for(i in sentences): cosine_similarity(embed(fact),embed(claim)), its naive but works for less no. of sentences but it has very high no. of api calls(=no. of sentences); it doesn't involve any semantic serach, its just brute force linear scan.
- A vector store solves both:
    You embed the facts once when building the collection
    You embed the claim once at query time
    The store finds the closest facts in one operation instead of looping
- For 10 facts this difference is small. For 200 facts (full article instead of summary), or when you're scoring 10 sentences in one request, this compounds fast.
- a vector database stores embeddings, and retrieves them by similarity; under the hood it performs nearest neighbour serach in high dimensional space.
- it can be used further for semantic serach, RAG, Recommendation system, Duplicate/near Duplicate detection, image search by visual similarity.
- we are using simplified RAG, retrieve the relevant facts from wikipedia and feed them as context to the scorer.

### Chroma DB:
- ChromaDB is an open-source vector database. It's the simplest production-grade option — no Docker, no server process, no setup. You pip install chromadb and it works.
- two modes:
    Mode 1: In-memory — data dies when process ends: client = chromadb.Client()
    Mode 2: Persistent — data saved to disk across runs: client = chromadb.PersistentClient(path="./chroma_data")
    We are using Mode 1. Why: each API request builds a fresh collection from fresh Wikipedia data. Persisting stale data would cause bugs. In-memory is correct here.

-The correct command to run this is:
    python3 -m core.vector_store ; no need to use extension (.py)
    The -m flag tells Python "run this as a module, not a script." This sets the path correctly so from core.embedder import embed resolves.
- bcz we are using other scripts as well in core dirctory e.g. embed fn in embedder.py of core.
- running this module alone will give v.small diff. e.g. 2-4% but we take final weigted avg. after multiple factors, that predicts hallucination reliably.




### Note:
    - Bash:
    > (Overwrite): Truncates the target file to 0 bytes before writing standard output. Destructive to existing data.
    >> (Append): Writes standard output to the end of the file (EOF). Preserves existing data.
    - in module python-dotevn ; python is included in the name of that module
    - cmd pallete > restart language server; if modules are note detected
                  > Reload window
    - if you see .venv in right of python base(3.13.5) that just indicates you are in that virtual envorment not that your interpretor is that venv
    - to change that go to select interpretor and then manually enter the path if you don't see it in the list, append /bin/python to the absolute path of .venv directory
    e.g. `/Users/althea/Developer/Projects/Hallucination-Detector/.venv/bin/python`
    - avoid using run button of vs code; as that might select a different interpretor; directly run using terminal and entering the path of file manually.
    - add multiple sleep timer by importing time; so you don't hit api rate limit
    - RPD(requests per day); RPM (Requests per minute)


## Project Structure

```
hallucination-detector/
├── .env                  ← API keys (never commit)
├── .env.example          ← template (commit this)
├── .gitignore
├── requirements.txt
├── api.py                ← FastAPI backend
├── core/
│   ├── __init__.py
│   ├── embedder.py       ← Module 1
│   ├── fetcher.py        ← Module 2
│   ├── vector_store.py   ← Module 3
│   ├── consistency.py    ← Module 4
│   ├── nli.py            ← Module 5
│   └── scorer.py         ← Module 6
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```
## Architecture (Simple)

```
Frontend (HTML/CSS/JS)  →  Vercel (free)
        ↓  fetch() POST
Backend (FastAPI)       →  Railway (free)
        ↓
    3 Detection Modules
    ├── Wikipedia Grounding
    ├── Self-Consistency
    └── Embedding Similarity
        ↓
    ChromaDB (in-memory, no setup)
    Wikipedia API (free, no key)
    Gemini API (you have key)
```

## Modules used

1. pip install google-generativeai python-dotenv numpy

    - google-generativeai: The official gRPC/REST Python wrapper for the Gemini API. It abstracts network authentication, payload serialization, and local session state for text, multimodal, and vector embedding models; 
    this module is no longer supported for our working of project the newer one is google-genai
    pip uninstall google-generativeai -y
    pip install google-genai

    - python-dotenv: A configuration parser adhering to the 12-Factor App methodology. It reads local .env files and injects the secrets directly into Python's os.environ, preventing API keys from being hardcoded into version-controlled source files.

    - numpy: A highly optimized, C-backed scientific computing library. It utilizes the ndarray (a homogeneous, contiguous memory block) to execute vectorized matrix mathematics using SIMD CPU instructions, effectively bypassing the Python Global Interpreter Lock (GIL).

    - wikiapi : fetches wiki articles in json format and then we retrieve summary section of that for fact checking.
---

## Project Structure: 

hallucination-detector/
├── .env                    ← API keys — never commit
├── .env.example
├── .gitignore
├── requirements.txt
├── Procfile                ← Railway: web: uvicorn api:app --host 0.0.0.0 --port $PORT
├── runtime.txt             ← python-3.11
├── api.py                  ← FastAPI backend
├── core/
│   ├── __init__.py
│   ├── embedder.py         ← embed() + cosine_similarity()
│   ├── fetcher.py          ← fetch_facts() + infer_topic()
│   ├── vector_store.py     ← build_collection() + retrieve_closest()
│   ├── consistency.py      ← 3x generation + pairwise embed similarity
│   ├── llm_evaluator.py    ← NLI + confidence in one Gemini call
│   └── scorer.py           ← orchestration layer
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
    



# Hallucination Detector

Detects and flags hallucinated claims in LLM-generated text by grounding output against Wikipedia facts using four independent signals — vector similarity, self-consistency, embedding similarity, and NLI classification.

Built for a hackathon submission. Live demo hosted on Vercel + Railway.

---

## Pipeline
```
INPUT TEXT
"Einstein won Nobel for relativity. He was born in 1879."
         │
         ▼
┌─────────────────────┐
│   Topic Inference   │  Gemini extracts main entity → "Albert Einstein"
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Wikipedia Fetcher  │  Fetches article summary → splits into ~12 sentences
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   ChromaDB (RAM)    │  Embeds all facts → stored as vectors in-memory
└────────┬────────────┘
         │
         ├───────────────────── per sentence ──────────────────────────┐
         │                                                             │
         ▼                                                             │
┌─────────────────────────────────────────────────────────────────┐   │
│                       SCORING ENGINE                            │   │
│                                                                 │   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │   │
│  │Wikipedia Grounding│  │Self-Consistency │  │     NLI       │  │   │
│  │                  │  │                  │  │               │  │   │
│  │embed(claim) →    │  │Gemini 3x at      │  │Gemini labels  │  │   │
│  │ChromaDB query →  │  │temp=0.8 → embed  │  │each fact as:  │  │   │
│  │closest fact      │  │responses →       │  │ENTAILMENT /   │  │   │
│  │                  │  │pairwise cosine   │  │NEUTRAL /      │  │   │
│  │1/(1+distance)    │  │avg = score       │  │CONTRADICTION  │  │   │
│  │→ grounding_score │  │                  │  │→ nli_score    │  │   │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘  │   │
│           │                     │                     │         │   │
│           └─────────────────────┼─────────────────────┘         │   │
│                                 ▼                               │   │
│   final = grounding*0.15 + consistency*0.35 + embedding*0.10    │   │
│         + nli*0.4                                               │   │
│                                                                 │   │
│   label = "hallucinated" if final < 0.45 else "grounded"        │   │
└─────────────────────────────────────────────────────────────────┘   │
                                  └───────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   FastAPI Backend   │  POST /analyze → returns JSON
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Frontend (JS)      │  Sentences highlighted red / green
│                     │  Click → show evidence + consistency responses
└─────────────────────┘


