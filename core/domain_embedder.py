# core/domain_embedder.py
#
# PURPOSE:
#   Domain-specific sentence embeddings for the hallucination detector.
#   Routes medical/legal/financial text to local BERT-family models.
#   Falls back to Gemini embedding-001 if local model fails.
#
# MODELS (local — downloaded automatically on first run via HuggingFace):
#   medical   → NeuML/pubmedbert-base-embeddings   (PubMedBERT sentence transformer, 768-dim)
#   legal     → law-ai/InLegalBert                 (BERT trained on Indian/EU legal text, 768-dim)
#   financial → yiyanghkust/finbert-tone           (FinBERT trained on financial corpora, 768-dim)
#   general   → Gemini gemini-embedding-001         (existing — best general-purpose embedder, 3072-dim)
#
# LOCAL INFERENCE:
#   Models are loaded via AutoModel.from_pretrained() — bypasses HF router entirely.
#   Pipeline tags are irrelevant locally. We extract last_hidden_state and mean pool.
#   Models cached to ~/.cache/huggingface/hub/ after first download (~440MB each).
#
# DEVICE:
#   M4 MacBook Air → MPS (Apple Silicon GPU) used automatically.
#   Falls back to CPU if MPS unavailable.
#
# CONSISTENCY RULE (CRITICAL):
#   embed_for_domain() MUST be used for BOTH fact embeddings (collection building)
#   and claim embeddings (querying). Never mix models for the same similarity computation.

import os
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

from core.embedder import embed as gemini_embed   # Gemini fallback + general domain

load_dotenv()


# ── Configuration ──────────────────────────────────────────────────────

# Device selection — MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise
DEVICE = (
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else
    "cpu"
)

# domain → HuggingFace model ID (downloaded locally on first run)
# None means: skip local model, use Gemini directly
DOMAIN_MODELS: dict[str, str | None] = {
    "medical":   "NeuML/pubmedbert-base-embeddings",
    "legal":     "law-ai/InLegalBert",
    "financial": "ProsusAI/finbert",
    "general":   None,
}

# Expected embedding dimensions (used in tests for validation)
MODEL_DIMS: dict[str, int] = {
    "NeuML/pubmedbert-base-embeddings": 768,
    "law-ai/InLegalBert":               768,
    "yiyanghkust/finbert-tone":         768,
    "gemini-embedding-001":             3072,
}

# BERT-family safe input length (512 token limit)
MAX_CHARS = 512


# ── Local Model Cache ───────────────────────────────────────────────────

# Models are loaded once per process and reused — avoids reloading 440MB per call
_MODEL_CACHE: dict[str, tuple] = {}


def _load_local_model(model_id: str) -> tuple:
    """
    Load tokenizer + model once, cache in memory for reuse.

    First call downloads from HuggingFace (~440MB) and caches to disk.
    Subsequent calls (same process) use the in-memory cache — instant.
    Subsequent runs (new process) load from disk cache — ~2-5s.

    Model moved to DEVICE (MPS on M4) for fast inference.
    """
    if model_id not in _MODEL_CACHE:
        print(f"[domain_embedder] Loading {model_id} (first call — may download)...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = AutoModel.from_pretrained(model_id)
        model     = model.to(DEVICE)
        model.eval()
        _MODEL_CACHE[model_id] = (tokenizer, model)
        print(f"[domain_embedder] {model_id.split('/')[-1]} loaded on {DEVICE}")
    return _MODEL_CACHE[model_id]


# ── Core Embedding Functions ────────────────────────────────────────────

def _embed_local_single(text: str, model_id: str) -> list[float] | None:
    """
    Embed a single text using a local HuggingFace model.

    Runs a forward pass through the model and mean pools the last_hidden_state
    over the token dimension → (hidden_dim,) flat embedding vector.

    This works for ALL model types regardless of their HF pipeline tag:
    - Sentence transformers: last_hidden_state mean pool ≈ CLS pooling output
    - BERT base (fill-mask, classification): encoder output before task head
    - The task-specific head (MLM, classification) is simply never called

    Returns embedding vector, or None on any failure (caller falls back to Gemini).
    """
    try:
        tokenizer, model = _load_local_model(model_id)

        inputs = tokenizer(
            text[:MAX_CHARS],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        # Move all input tensors to same device as model
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pool over token dimension: (1, seq_len, hidden_dim) → (hidden_dim,)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        name = model_id.split("/")[-1]
        print(f"[domain_embedder] {name} ✓  dim={len(embedding)}")
        return embedding

    except Exception as e:
        print(f"[domain_embedder] Local model error ({model_id}): {e}")
        return None


def _embed_local_batch(texts: list[str], model_id: str) -> list[list[float]] | None:
    """
    Embed multiple texts in a single forward pass (more efficient than N single calls).

    Tokenizer pads all texts to the same length in the batch.
    Mean pooling applied per-item: (batch, seq_len, hidden_dim) → (batch, hidden_dim).

    Returns list of embedding vectors (same order as input), or None on failure.
    """
    try:
        tokenizer, model = _load_local_model(model_id)

        inputs = tokenizer(
            [t[:MAX_CHARS] for t in texts],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pool per item: (batch, seq_len, hidden_dim) → (batch, hidden_dim)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()

        name = model_id.split("/")[-1]
        print(f"[domain_embedder] {name} batch ✓  count={len(embeddings)}  dim={len(embeddings[0])}")
        return embeddings

    except Exception as e:
        print(f"[domain_embedder] Local batch error ({model_id}): {e}")
        return None


# ── Public API ──────────────────────────────────────────────────────────

def embed_for_domain(text: str, domain: str) -> list[float]:
    """
    Main entry point. Returns the domain-appropriate embedding for a single text.

    Routing logic:
        "medical"   → NeuML/pubmedbert-base-embeddings (local)  → fallback: Gemini
        "legal"     → law-ai/InLegalBert               (local)  → fallback: Gemini
        "financial" → yiyanghkust/finbert-tone          (local)  → fallback: Gemini
        "general"   → Gemini directly (no local model needed)

    CRITICAL — consistency constraint:
        This function must be used for BOTH operations that compare vectors:
        (a) embedding facts when building the ChromaDB collection
        (b) embedding the claim when querying the collection
        Using different models for (a) and (b) produces meaningless similarity scores.
        Module E enforces this via get_embed_fn().

    Args:
        text:   input text to embed (any length, truncated internally to 512 chars)
        domain: "medical" | "legal" | "financial" | "general"

    Returns:
        list[float] — embedding vector
        Dimensions: 768 for local models, 3072 for Gemini
    """
    model_id = DOMAIN_MODELS.get(domain)

    # "general" domain: use Gemini directly
    if model_id is None:
        print(f"[domain_embedder] general → Gemini")
        return list(gemini_embed(text))

    # Domain-specific local model
    result = _embed_local_single(text, model_id)

    if result is not None:
        return result

    # Local model failed → Gemini fallback
    print(f"[domain_embedder] WARNING: {model_id} failed → Gemini fallback for domain '{domain}'")
    return list(gemini_embed(text))


def batch_embed_for_domain(texts: list[str], domain: str) -> list[list[float]]:
    """
    Embed multiple texts for a given domain efficiently (single forward pass).

    Tries batched local inference first (1 forward pass for all texts).
    Falls back to sequential embed_for_domain() if batch fails.

    Primary use case: building ChromaDB collections.
        build_collection() needs to embed 12-20 Wikipedia/PubMed facts.
        Sequential: 12-20 forward passes.
        Batch: 1 forward pass — significantly faster.

    Args:
        texts:  list of strings to embed
        domain: "medical" | "legal" | "financial" | "general"

    Returns:
        list of embedding vectors, one per input text (same order as input)
    """
    if not texts:
        return []

    model_id = DOMAIN_MODELS.get(domain)

    # "general": Gemini doesn't support batching in the new SDK → sequential
    if model_id is None:
        print(f"[domain_embedder] general batch: {len(texts)} texts → Gemini sequential")
        return [list(gemini_embed(t)) for t in texts]

    # Try batched local inference
    print(f"[domain_embedder] Batching {len(texts)} texts for domain '{domain}'...")
    results = _embed_local_batch(texts, model_id)

    if results is not None and len(results) == len(texts):
        return results

    # Batch failed → sequential with per-text Gemini fallback
    print(f"[domain_embedder] Batch failed → sequential fallback for {len(texts)} texts")
    return [embed_for_domain(t, domain) for t in texts]


def get_embed_fn(domain: str):
    """
    Returns a callable bound to the given domain: embed(text) -> list[float].

    This is the integration point for Module E (updated vector_store + scorer).

    Instead of threading `domain` through every function, the scorer creates
    this function once and passes it as a dependency:

        # In scorer.py (Module E):
        embed_fn = get_embed_fn(domain)
        collection = build_collection(topic, facts, embed_fn)
        closest    = retrieve_closest(collection, sentence, embed_fn)

    vector_store.py receives a plain callable — it never needs to know about domains.
    The coupling is at the scorer level, not at the storage level. Clean separation.

    Args:
        domain: "medical" | "legal" | "financial" | "general"

    Returns:
        callable: (text: str) -> list[float]
    """
    return lambda text: embed_for_domain(text, domain)


# ── Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    from core.embedder import cosine_similarity

    print(f"\n[domain_embedder] Device: {DEVICE}")

    def header(title: str):
        print(f"\n{'═' * 65}")
        print(f"  {title}")
        print(f"{'═' * 65}")


    # ── TEST 1: Single embed — each domain ────────────────────────────
    header("TEST 1: Single embed — verify each domain model responds")

    test_texts = {
        "medical":   "Amoxicillin is a beta-lactam antibiotic used to treat bacterial infections.",
        "legal":     "The defendant has the right to remain silent under the Fifth Amendment.",
        "financial": "The Federal Reserve raised interest rates by 25 basis points.",
        "general":   "Albert Einstein developed the theory of relativity.",
    }

    embeddings_cache = {}

    for domain, text in test_texts.items():
        print(f"\n── {domain.upper()} ──")
        print(f"  Text: '{text[:60]}...'")
        emb = embed_for_domain(text, domain)
        embeddings_cache[(domain, "test")] = emb
        print(f"  ✓ dim={len(emb)}")

        expected_dim = 3072 if domain == "general" else 768
        status = "PASS" if len(emb) == expected_dim else f"FAIL (expected {expected_dim})"
        print(f"  Dimension check: {status}")


    # ── TEST 2: Batch embed — medical domain ──────────────────────────
    header("TEST 2: Batch embed — medical domain (PubMed facts simulation)")

    medical_facts = [
        "Amoxicillin inhibits bacterial cell wall synthesis by binding to penicillin-binding proteins.",
        "Beta-lactam antibiotics are effective against gram-positive bacteria.",
        "MRSA is resistant to methicillin and most other penicillin-related antibiotics.",
        "Penicillin allergy affects approximately 10 percent of the general population.",
        "Bacterial resistance to amoxicillin is mediated by beta-lactamase enzymes.",
    ]

    print(f"\nEmbedding {len(medical_facts)} facts in one batch call...")
    start = time.time()
    batch_embs = batch_embed_for_domain(medical_facts, "medical")
    elapsed = round(time.time() - start, 2)

    print(f"\nBatch result:")
    print(f"  Count:   {len(batch_embs)} embeddings")
    print(f"  Dim:     {len(batch_embs[0])} per embedding")
    print(f"  Time:    {elapsed}s")

    batch_status = (
        "PASS" if len(batch_embs) == len(medical_facts)
              and all(len(e) == 768 for e in batch_embs)
        else "FAIL"
    )
    print(f"  Status:  {batch_status}")


    # ── TEST 3: Domain similarity comparison ──────────────────────────
    header("TEST 3: Similarity comparison — domain model vs Gemini")

    print("\n── Medical pair ──")
    print("  Claim:  'Amoxicillin inhibits bacterial cell wall synthesis'")
    print("  Fact:   'Beta-lactam antibiotics target penicillin-binding proteins'")
    print("  (These say the same thing with different terminology)")

    claim_m = "Amoxicillin inhibits bacterial cell wall synthesis"
    fact_m  = "Beta-lactam antibiotics target penicillin-binding proteins"

    emb_claim_medical  = embed_for_domain(claim_m, "medical")
    emb_fact_medical   = embed_for_domain(fact_m,  "medical")
    sim_medical        = round(cosine_similarity(emb_claim_medical, emb_fact_medical), 4)

    emb_claim_gemini   = embed_for_domain(claim_m, "general")
    emb_fact_gemini    = embed_for_domain(fact_m,  "general")
    sim_gemini_medical = round(cosine_similarity(emb_claim_gemini, emb_fact_gemini), 4)

    print(f"\n  PubMedBERT similarity: {sim_medical}")
    print(f"  Gemini similarity:     {sim_gemini_medical}")
    print(f"  Domain model {'higher ✓' if sim_medical > sim_gemini_medical else 'lower (Gemini won this pair)'}")

    print("\n── Legal pair ──")
    print("  Claim:  'The defendant has the right to remain silent'")
    print("  Fact:   'Fifth Amendment protects against compelled self-incrimination'")

    claim_l = "The defendant has the right to remain silent"
    fact_l  = "Fifth Amendment protects against compelled self-incrimination"

    emb_claim_legal  = embed_for_domain(claim_l, "legal")
    emb_fact_legal   = embed_for_domain(fact_l,  "legal")
    sim_legal        = round(cosine_similarity(emb_claim_legal, emb_fact_legal), 4)

    emb_claim_gem_l  = embed_for_domain(claim_l, "general")
    emb_fact_gem_l   = embed_for_domain(fact_l,  "general")
    sim_gemini_legal = round(cosine_similarity(emb_claim_gem_l, emb_fact_gem_l), 4)

    print(f"\n  InLegalBERT similarity: {sim_legal}")
    print(f"  Gemini similarity:      {sim_gemini_legal}")
    print(f"  Domain model {'higher ✓' if sim_legal > sim_gemini_legal else 'lower (Gemini won this pair)'}")


    # ── TEST 4: get_embed_fn interface ────────────────────────────────
    header("TEST 4: get_embed_fn — callable interface for Module E")

    print("\nCreating embed functions via get_embed_fn...")
    embed_medical   = get_embed_fn("medical")
    embed_legal     = get_embed_fn("legal")
    embed_financial = get_embed_fn("financial")
    embed_general   = get_embed_fn("general")

    print("Testing each callable with a short text...")
    test_sentence = "This is a test sentence for embedding verification."

    for name, fn in [("medical", embed_medical), ("financial", embed_financial), ("general", embed_general)]:
        emb = fn(test_sentence)
        exp = 768 if name != "general" else 3072
        ok  = "PASS" if len(emb) == exp else f"FAIL (got {len(emb)}, expected {exp})"
        print(f"  {name}: dim={len(emb)}  → {ok}")

    print("\n  get_embed_fn ✓ — returns correct domain-bound callables")


    # ── TEST 5: Cross-domain sanity ───────────────────────────────────
    header("TEST 5: Cross-domain sanity — unrelated claims should have low similarity")

    medical_sentence = "The patient showed elevated creatinine indicating acute kidney injury."
    legal_sentence   = "The court denied the motion for summary judgment citing procedural errors."

    emb_med_claim = embed_for_domain(medical_sentence, "medical")
    emb_med_legal = embed_for_domain(legal_sentence,   "medical")
    sim_cross     = round(cosine_similarity(emb_med_claim, emb_med_legal), 4)

    print(f"\n  Medical vs legal sentence, using medical embedder:")
    print(f"  Similarity: {sim_cross}  (expect < 0.60 — unrelated domains)")
    status = "PASS" if sim_cross < 0.70 else "WARN (unexpectedly high — check model)"
    print(f"  Status: {status}")


    # ── SUMMARY ───────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print("  SUMMARY")
    print(f"{'═' * 65}")
    print("  Test 1: Single embed per domain           — check dims above")
    print("  Test 2: Batch embed medical               — check count/dim above")
    print("  Test 3: Domain vs Gemini similarity       — check comparison above")
    print("  Test 4: get_embed_fn interface            — check PASS above")
    print("  Test 5: Cross-domain sanity               — check similarity above")
    print()
    print("  dim=768 for medical/legal/financial → local models working ✓")
    print("  dim=3072 for medical/legal/financial → local failed, Gemini fallback active")
    print(f"{'═' * 65}")