# api.py
# Interface between FastAPI backend and core scoring pipeline.
# Serves the /analyze endpoint and mounts the frontend as static files.

import os
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

from core.scorer import analyze_text


# ── Rate limiter ───────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hallucination Detector API",
    description=(
        "Detects hallucinated claims in LLM-generated text. "
        "Grounds each sentence against domain-appropriate sources (Wikipedia, PubMed, CourtListener) "
        "using four independent signals: NLI classification, self-consistency, "
        "vector grounding, and embedding similarity."
    ),
    version="2.0.0",
)

app.state.limiter = limiter

# ── Middleware ─────────────────────────────────────────────────────────
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SlowAPIMiddleware)
# ── Exception handlers ─────────────────────────────────────────────────

# FIX: Register slowapi's handler so rate-limit hits return 429 with a
# readable message instead of an unformatted 500 traceback.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Request / Response models ──────────────────────────────────────────

class TextInput(BaseModel):
    text: str


# ── Routes ────────────────────────────────────────────────────────────

@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze(request: Request, input: TextInput):
    """
    Analyzes input text for hallucinated claims.

    Returns per-sentence scores across four signals:
      - nli_score        (weight 0.40) — Gemini NLI vs source facts
      - consistency_score (weight 0.35) — multi-model consensus
      - grounding_score  (weight 0.15) — ChromaDB vector distance
      - embedding_score  (weight 0.10) — cosine similarity to closest fact

    Each sentence result includes:
      evidence, evidence_source, evidence_url, top_facts (top-3 with URLs),
      contradicting_fact, nli_labels, consistency_verdicts, consistency_responses.
    """

    # ── Input validation ───────────────────────────────────────────────
    text = input.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    # FIX: raised from 500 → 2000 chars.
    # Original 500 was too tight for real multi-sentence LLM output.
    # Free-tier quota is protected by the rate limiter (5/minute), not char count.
    if len(text) > 2000:
        raise HTTPException(
            status_code=400,
            detail="Input too long. Maximum 2000 characters."
        )

    # ── Analysis ───────────────────────────────────────────────────────
    # FIX: analyze_text is synchronous and takes 30-90s (Gemini calls,
    # local model inference, ChromaDB operations). Running it directly
    # inside async def blocks the entire uvicorn event loop — no other
    # request can be served while it runs.
    # asyncio.to_thread() offloads it to a thread pool, keeping the loop free.
    try:
        result = await asyncio.to_thread(analyze_text, text)
    except Exception as exc:
        # FIX: catch unexpected exceptions (model load failures, ChromaDB errors,
        # network timeouts that bypass internal retries) and return a clean 500
        # instead of an unformatted traceback.
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed unexpectedly: {str(exc)}"
        )

    # analyze_text returns {"error": "..."} on known failures (no facts found etc.)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/health")
async def health():
    """Health check — used by Railway and other deployment platforms."""
    return {"status": "ok"}


# ── Static frontend ────────────────────────────────────────────────────
# Mounted LAST so that /analyze and /health always take priority.
# FIX: guard against missing frontend/ directory at startup.
# Without this guard, StaticFiles raises RuntimeError immediately and
# the entire API refuses to start.
_FRONTEND_DIR = "frontend"
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
else:
    print(
        f"[api] WARNING: '{_FRONTEND_DIR}/' directory not found. "
        "Frontend will not be served. API routes (/analyze, /health) still work."
    )


# ── Dev entrypoint ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)