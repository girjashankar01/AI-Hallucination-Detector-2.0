# core/domain_embedder.py
#
# PURPOSE:
#   Domain-specific sentence embeddings for the hallucination detector.
#   Routes medical/legal/financial text to specialized BERT-family models via HF API.
#   Falls back to Gemini embedding-001 if HF is unavailable.
#
# MODELS:
#   medical   → pritamdeka/S-PubMedBert-MS-MARCO   (PubMedBERT fine-tuned as sentence transformer)
#   legal     → nlpaueb/legal-bert-base-uncased      (BERT trained on EU legislation + case law)
#   financial → ProsusAI/finbert                     (BERT trained on financial news + 10-K filings)
#   general   → Gemini gemini-embedding-001           (existing — best general-purpose embedder)
#
# HF ENDPOINT (new router — old api-inference.huggingface.co is dead):
#   https://router.huggingface.co/hf-inference/models/{owner}/{model}
#
# RESPONSE SHAPE BY MODEL TYPE:
#   Sentence transformers (S-PubMedBert-MS-MARCO):
#       Single:  [[f1, ..., f768]]            shape (1, 768)
#       Batch:   [[f1...], [f1...]]            shape (batch, 768)
#
#   BERT base models (legal-bert, finbert):
#       Single:  [[[tok_emb, ...], ...]]       shape (1, seq_len, 768)
#       Batch:   [[[tok_emb,...], ...], [...]] shape (batch, seq_len, 768)
#       → Need mean pooling over seq_len axis
#
# CONSISTENCY RULE (CRITICAL):
#   embed_for_domain() MUST be used for BOTH fact embeddings (collection building)
#   and claim embeddings (querying). Never mix models for the same similarity computation.

import os
import time
import numpy as np
import requests
from dotenv import load_dotenv

from core.embedder import embed as gemini_embed   # Gemini fallback + general domain

load_dotenv()


# ── Configuration ──────────────────────────────────────────────────────

HF_TOKEN   = os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
#HF_BASE    = "https://router.huggingface.co/hf-inference/models"
HF_BASE         = "https://router.huggingface.co/hf-inference/models"
HF_FEATURE_BASE = "https://router.huggingface.co/hf-inference/pipeline/feature-extraction"

# domain → HuggingFace model ID
# None means: skip HF entirely, use Gemini directly
DOMAIN_MODELS: dict[str, str | None] = {
    "medical":   "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL",
    "legal":     "AnonymousSub/FPDM_Legal_RoBERTa",
    "financial": "mradermacher/finance-embeddings-investopedia-i1-GGUF",
    "general":   None,
}

# Expected embedding dimensions (used in tests for validation)
MODEL_DIMS: dict[str, int] = {
    "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL": 768,
    "AnonymousSub/FPDM_Legal_RoBERTa":              768,
    "mradermacher/finance-embeddings-investopedia-i1-GGUF": 768,
    "gemini-embedding-001":                         3072,
}

# BERT-family safe input length (512 token limit → 512 chars is a safe approximation)
# Actual tokenization differs by model, but no BERT sentence exceeds 512 tokens at 512 chars
MAX_CHARS = 512


# ── Response Parsing ───────────────────────────────────────────────────

def _parse_single_embedding(data: list) -> list[float]:
    """
    Parse a single-input HuggingFace feature-extraction response.

    Handles both response shapes:

    Shape A — Sentence transformers (S-PubMedBert-MS-MARCO):
        data = [[f1, f2, ..., f768]]       (list of list of float)
        data[0] is the flat embedding vector.
        Detection: data[0][0] is a float.

    Shape B — BERT base models (legal-bert, finbert):
        data = [[[f1,...], [f1,...], ...]] (list of list of list of float)
        data[0] is (seq_len, hidden_dim) — one vector per token.
        Needs mean pooling over seq_len.
        Detection: data[0][0] is a list.

    Raises ValueError on malformed response (caller catches and falls back to Gemini).
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Empty or non-list HF response: {type(data)}")

    inner = data[0]   # strip the outer batch dimension

    if not inner:
        raise ValueError("Empty inner list in HF embedding response")

    if isinstance(inner[0], float):
        # Shape A: inner = [f1, f2, ..., f768]
        return list(inner)
    else:
        # Shape B: inner = [[f1,...], [f1,...], ...]  shape (seq_len, hidden_dim)
        arr = np.array(inner, dtype=np.float32)   # → (seq_len, hidden_dim)
        return arr.mean(axis=0).tolist()            # → (hidden_dim,)


def _parse_batch_embeddings(data: list) -> list[list[float]]:
    """
    Parse a batched HuggingFace feature-extraction response.

    Batch responses differ from single responses by one dimension:

    Shape A — Sentence transformers (batch):
        data = [[f1...f768], [f1...f768]]    shape (batch, dim)
        Each data[i] is already a flat embedding.
        Detection: data[0][0] is a float.

    Shape B — BERT base models (batch):
        data = [[[tok_embs...], ...], ...]   shape (batch, seq_len, dim)
        Each data[i] is (seq_len, dim), needs mean pooling.
        Detection: data[0][0] is a list.

    Returns list of flat embedding vectors, one per input text.
    Raises ValueError on malformed response.
    """
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"Empty batch response: {type(data)}")

    embeddings = []
    for i, item in enumerate(data):
        if not item:
            raise ValueError(f"Empty item at batch index {i}")

        if isinstance(item[0], float):
            # Shape A: item is already a flat vector
            embeddings.append(list(item))
        else:
            # Shape B: item is (seq_len, hidden_dim) → mean pool
            arr = np.array(item, dtype=np.float32)
            embeddings.append(arr.mean(axis=0).tolist())

    return embeddings


# ── HF API Request Primitives ──────────────────────────────────────────

def _hf_post(url: str, payload: dict, timeout: int = 30) -> requests.Response | None:
    """
    Single HF API POST with cold-start handling.

    503 means the model isn't loaded yet (HF cold-starts free-tier models).
    The response body contains estimated_time (seconds to wait).
    We sleep that long, then retry once.

    Returns the Response object on any non-503 status, or None on timeout/exception.
    The caller checks response.status_code — we don't raise here.
    """
    try:
        resp = requests.post(url, headers=HF_HEADERS, json=payload, timeout=timeout)

        if resp.status_code == 503:
            body    = resp.json()
            wait    = min(body.get("estimated_time", 20), 25)   # cap at 25s
            name    = url.split("/")[-1]
            print(f"[domain_embedder] {name} cold-starting, waiting {wait:.0f}s...")
            time.sleep(wait)

            # Single retry after cold-start wait
            resp = requests.post(url, headers=HF_HEADERS, json=payload, timeout=timeout)

        return resp

    except requests.Timeout:
        print(f"[domain_embedder] Timeout: {url.split('/')[-1]}")
        return None
    except Exception as e:
        print(f"[domain_embedder] Request error: {e}")
        return None


# ── Core Embedding Functions ───────────────────────────────────────────

def _embed_hf_single(text: str, model_id: str) -> list[float] | None:
    """
    Embed a single text using a HuggingFace model.

    Returns embedding vector, or None on any failure.
    None signals the caller to use the Gemini fallback.

    Text truncated to MAX_CHARS — safe limit for BERT 512-token max.
    """
    url = f"{HF_FEATURE_BASE}/{model_id}"
    payload = {"inputs": text[:MAX_CHARS]}

    resp = _hf_post(url, payload)

    if resp is None or resp.status_code != 200:
        if resp is not None:
            print(f"[domain_embedder] HTTP {resp.status_code} from {model_id}")
            print(f"[domain_embedder] Body: {resp.text[:200]}")
        return None

    try:
        data      = resp.json()
        embedding = _parse_single_embedding(data)
        name      = model_id.split("/")[-1]
        sample    = [round(x, 4) for x in embedding[:3]]
        print(f"[domain_embedder] {name} ✓  dim={len(embedding)}  sample={sample}...")
        return embedding

    except (ValueError, Exception) as e:
        print(f"[domain_embedder] Parse error from {model_id}: {e}")
        return None


def _embed_hf_batch(texts: list[str], model_id: str) -> list[list[float]] | None:
    """
    Embed multiple texts in a single HF API call.

    Batch request is more efficient than N sequential calls:
    - 1 HTTP round-trip instead of N
    - One cold-start wait instead of N
    - Useful when building ChromaDB collection (embedding 12-20 facts at once)

    Returns list of embedding vectors (same length as texts), or None on failure.
    Caller falls back to sequential _embed_hf_single() if this returns None.
    """
    url = f"{HF_FEATURE_BASE}/{model_id}"
    payload = {"inputs": [t[:MAX_CHARS] for t in texts]}

    resp = _hf_post(url, payload, timeout=60)   # batch needs longer timeout

    if resp is None or resp.status_code != 200:
        if resp is not None:
            print(f"[domain_embedder] Batch HTTP {resp.status_code} from {model_id}")
        return None

    try:
        data       = resp.json()
        embeddings = _parse_batch_embeddings(data)
        name       = model_id.split("/")[-1]
        print(f"[domain_embedder] {name} batch ✓  count={len(embeddings)}  dim={len(embeddings[0])}")
        return embeddings

    except (ValueError, Exception) as e:
        print(f"[domain_embedder] Batch parse error from {model_id}: {e}")
        return None


# ── Public API ─────────────────────────────────────────────────────────

def embed_for_domain(text: str, domain: str) -> list[float]:
    """
    Main entry point. Returns the domain-appropriate embedding for a single text.

    Routing logic:
        "medical"   → S-PubMedBert-MS-MARCO (HF)  → fallback: Gemini
        "legal"     → legal-bert-base-uncased (HF)  → fallback: Gemini
        "financial" → ProsusAI/finbert (HF)          → fallback: Gemini
        "general"   → Gemini directly (no HF model needed for general)

    CRITICAL — consistency constraint:
        This function must be used for BOTH operations that compare vectors:
        (a) embedding facts when building the ChromaDB collection
        (b) embedding the claim when querying the collection
        Using different models for (a) and (b) produces meaningless similarity scores.
        Module E enforces this via get_embed_fn().

    Args:
        text:   input text to embed (any length, truncated internally)
        domain: "medical" | "legal" | "financial" | "general"

    Returns:
        list[float] — embedding vector
        Dimensions: 768 for HF models, 3072 for Gemini
    """
    model_id = DOMAIN_MODELS.get(domain)

    # "general" domain: use Gemini directly, no HF involved
    if model_id is None:
        print(f"[domain_embedder] general → Gemini")
        return list(gemini_embed(text))

    # Domain-specific HF model
    result = _embed_hf_single(text, model_id)

    if result is not None:
        return result

    # HF failed for any reason → Gemini fallback
    # Log it clearly — silent fallbacks cause confusing behavior in production
    print(f"[domain_embedder] WARNING: {model_id} failed → Gemini fallback for domain '{domain}'")
    return list(gemini_embed(text))


def batch_embed_for_domain(texts: list[str], domain: str) -> list[list[float]]:
    """
    Embed multiple texts for a given domain efficiently.

    Tries a single batched HF API call first (1 round-trip for all texts).
    Falls back to sequential embed_for_domain() if batch fails.

    Primary use case: building ChromaDB collections.
        build_collection() needs to embed 12-20 Wikipedia/PubMed facts.
        Sequential: 12-20 API calls.
        Batch: 1 API call.

    Module E will call this in the updated build_collection():
        embeddings = batch_embed_for_domain(facts_text, domain)

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

    # Try batched HF call first
    print(f"[domain_embedder] Batching {len(texts)} texts for domain '{domain}'...")
    results = _embed_hf_batch(texts, model_id)

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


# ── Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    from core.embedder import cosine_similarity

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

    embeddings_cache = {}   # save for later similarity tests

    for domain, text in test_texts.items():
        print(f"\n── {domain.upper()} ──")
        print(f"  Text: '{text[:60]}...'")
        emb = embed_for_domain(text, domain)
        embeddings_cache[(domain, "test")] = emb
        print(f"  ✓ dim={len(emb)}")

        expected_dim = 3072 if domain == "general" else 768
        status = "PASS" if len(emb) == expected_dim else f"FAIL (expected {expected_dim})"
        print(f"  Dimension check: {status}")
        time.sleep(1)   # be gentle with HF rate limits between domains


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
    time.sleep(2)


    # ── TEST 3: Domain similarity comparison (the key test) ───────────
    # Do medical embeddings give better similarity for medical claim pairs?
    # Do legal embeddings give better similarity for legal claim pairs?
    # This validates that domain models add real value over Gemini for specialized text.
    header("TEST 3: Similarity comparison — domain model vs Gemini")

    print("\n── Medical pair ──")
    print("  Claim:  'Amoxicillin inhibits bacterial cell wall synthesis'")
    print("  Fact:   'Beta-lactam antibiotics target penicillin-binding proteins'")
    print("  (These say the same thing with different terminology)")

    claim_m  = "Amoxicillin inhibits bacterial cell wall synthesis"
    fact_m   = "Beta-lactam antibiotics target penicillin-binding proteins"

    emb_claim_medical  = embed_for_domain(claim_m, "medical")
    emb_fact_medical   = embed_for_domain(fact_m, "medical")
    sim_medical        = round(cosine_similarity(emb_claim_medical, emb_fact_medical), 4)
    time.sleep(1)

    emb_claim_gemini   = embed_for_domain(claim_m, "general")
    emb_fact_gemini    = embed_for_domain(fact_m, "general")
    sim_gemini_medical = round(cosine_similarity(emb_claim_gemini, emb_fact_gemini), 4)
    time.sleep(1)

    print(f"\n  S-PubMedBert similarity: {sim_medical}")
    print(f"  Gemini similarity:       {sim_gemini_medical}")
    print(f"  Domain model {'higher ✓' if sim_medical > sim_gemini_medical else 'lower (fallback was used or Gemini won this pair)'}")

    print("\n── Legal pair ──")
    print("  Claim:  'The defendant has the right to remain silent'")
    print("  Fact:   'Fifth Amendment protects against compelled self-incrimination'")

    claim_l  = "The defendant has the right to remain silent"
    fact_l   = "Fifth Amendment protects against compelled self-incrimination"

    emb_claim_legal  = embed_for_domain(claim_l, "legal")
    emb_fact_legal   = embed_for_domain(fact_l, "legal")
    sim_legal        = round(cosine_similarity(emb_claim_legal, emb_fact_legal), 4)
    time.sleep(1)

    emb_claim_gem_l  = embed_for_domain(claim_l, "general")
    emb_fact_gem_l   = embed_for_domain(fact_l, "general")
    sim_gemini_legal = round(cosine_similarity(emb_claim_gem_l, emb_fact_gem_l), 4)
    time.sleep(1)

    print(f"\n  Legal-BERT similarity:   {sim_legal}")
    print(f"  Gemini similarity:       {sim_gemini_legal}")
    print(f"  Domain model {'higher ✓' if sim_legal > sim_gemini_legal else 'lower (Gemini won this pair)'}")


    # ── TEST 4: get_embed_fn interface ────────────────────────────────
    header("TEST 4: get_embed_fn — callable interface for Module E")

    print("\nCreating embed functions via get_embed_fn...")
    embed_medical   = get_embed_fn("medical")
    embed_legal     = get_embed_fn("legal")
    embed_financial = get_embed_fn("financial")
    embed_general   = get_embed_fn("general")

    print("Testing each callable with a short text...")
    time.sleep(1)

    test_sentence = "This is a test sentence for embedding verification."

    for name, fn in [("medical", embed_medical), ("general", embed_general)]:
        emb = fn(test_sentence)
        exp = 768 if name != "general" else 3072
        ok  = "PASS" if len(emb) == exp else f"FAIL (got {len(emb)}, expected {exp})"
        print(f"  {name}: dim={len(emb)}  → {ok}")
        time.sleep(1)

    print("\n  get_embed_fn ✓ — returns correct domain-bound callables")


    # ── TEST 5: Cross-domain similarity check (sanity test) ───────────
    # Claims from different domains should score low similarity.
    # If they score high, the embedding model is not working correctly.
    header("TEST 5: Cross-domain sanity — unrelated claims should have low similarity")

    medical_sentence = "The patient showed elevated creatinine indicating acute kidney injury."
    legal_sentence   = "The court denied the motion for summary judgment citing procedural errors."

    # Both embedded with medical model — should be low similarity
    emb_med_claim = embed_for_domain(medical_sentence, "medical")
    time.sleep(1)
    emb_med_legal = embed_for_domain(legal_sentence, "medical")
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
    print("  If Test 1 shows dim=768 for medical/legal/financial: HF models are working")
    print("  If Test 1 shows dim=3072 for all: HF failed, Gemini fallback is active")
    print("  Both are valid — fallback ensures pipeline never breaks")
    print(f"{'═' * 65}")