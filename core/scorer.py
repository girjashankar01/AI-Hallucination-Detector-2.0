# core/scorer.py

import re
import os
from dotenv import load_dotenv
from google import genai

from core.embedder          import cosine_similarity
from core.fetcher           import infer_topic
from core.vector_store      import build_collection, retrieve_closest
from core.consistency       import check_consistency
from core.nli               import check_nli, check_nli_batch
from core.domain_classifier import classify_domain
from core.domain_embedder   import get_embed_fn
from core.sources           import fetch_facts_for_domain

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── Score weights ──────────────────────────────────────────────────────
_W_NLI         = 0.40
_W_CONSISTENCY = 0.35
_W_GROUNDING   = 0.15
_W_EMBEDDING   = 0.10

# Hard caps
_CAP_CONTRADICTION = 0.25
_CAP_ALL_FALSE     = 0.30
_HALLUC_THRESHOLD  = 0.45


# ── Sentence splitter ──────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in re.split(r'(?<=[.!?]) +', text)
        if len(s.strip()) > 20
    ]


# ── Per-sentence scorer ────────────────────────────────────────────────

def score_sentence(
    sentence:    str,
    collection,
    embed_fn,                           # NEW — domain-bound callable
    _embedding:  list       | None = None,
    _closest:    list       | None = None,
    _nli_result: dict       | None = None,
) -> dict:
    """
    Scores one sentence against the Wikipedia/PubMed/etc collection.

    embed_fn must be the SAME callable used to build the collection —
    using a different model here vs build_collection produces wrong
    cosine distances.
    """

    # ── Method 1: Wikipedia/source grounding ──────────────────────────
    sentence_embedding = _embedding if _embedding is not None else embed_fn(sentence)
    closest = _closest if _closest is not None else retrieve_closest(
        collection, sentence, n=3,
        embed_fn=embed_fn,
        query_embedding=sentence_embedding,
    )

    best_distance   = closest[0]["distance"]
    grounding_score = round(1 / (1 + best_distance), 4)

    # ── Method 2: Self-consistency (Module D) ─────────────────────────
    consistency_result = check_consistency(sentence, n=3)
    consistency_score  = consistency_result["consistency_score"]
    all_false          = consistency_result["false_count"] == 3

    # ── Method 3: Embedding similarity ────────────────────────────────
    fact_embedding  = embed_fn(closest[0]["fact"])
    raw_cosine      = cosine_similarity(sentence_embedding, fact_embedding)
    embedding_score = round((raw_cosine + 1) / 2, 4)

    # ── Method 4: NLI ─────────────────────────────────────────────────
    if _nli_result is not None:
        nli_result = _nli_result
    else:
        nli_result = check_nli(
            claim = sentence,
            facts = [item["fact"] for item in closest],
        )

    nli_score = nli_result["nli_score"]

    # ── Weighted combination ───────────────────────────────────────────
    final_score = round(
        grounding_score   * _W_GROUNDING   +
        consistency_score * _W_CONSISTENCY +
        embedding_score   * _W_EMBEDDING   +
        nli_score         * _W_NLI,
        4
    )

    # ── Hard caps ─────────────────────────────────────────────────────
    if nli_result["verdict"] == "CONTRADICTION":
        final_score = min(final_score, _CAP_CONTRADICTION)
    elif all_false:
        final_score = min(final_score, _CAP_ALL_FALSE)

    label = "hallucinated" if final_score < _HALLUC_THRESHOLD else "grounded"

    return {
        "sentence":              sentence,
        "final_score":           final_score,
        "label":                 label,
        "grounding_score":       grounding_score,
        "consistency_score":     consistency_score,
        "embedding_score":       embedding_score,
        "nli_score":             nli_score,
        "nli_verdict":           nli_result["verdict"],
        "evidence":              closest[0]["fact"],
        "evidence_source":       closest[0]["source"],    # NEW
        "evidence_url":          closest[0]["url"],       # NEW
        "evidence_distance":     closest[0]["distance"],
        "top_facts":             closest,
        "contradicting_fact":    nli_result["contradicting_fact"],
        "nli_labels":            nli_result["nli_labels"],
        "consistency_responses": consistency_result["responses"],
        "consistency_verdicts":  consistency_result["verdicts"],
    }


# ── Full text analyzer ─────────────────────────────────────────────────

def analyze_text(text: str) -> dict:
    """
    Main entry point. Module E version.

    Workflow:
    1.  Classify domain (medical / legal / financial / general)
    2.  Infer Wikipedia/source topic
    3.  Fetch facts from domain-appropriate source (with fallbacks)
    4.  Build domain-specific embed_fn via get_embed_fn(domain)
    5.  Build ChromaDB collection using embed_fn
    6.  Pre-compute embeddings + closest facts for all sentences
    7.  Batch NLI (1 API call)
    8.  Score each sentence
    9.  Aggregate

    Key change from Module D scorer:
        embed_fn is created once from the domain and threaded through
        build_collection, retrieve_closest, and score_sentence.
        This guarantees vector space consistency (same model for build + query).
    """

    # ── Step 1: Classify domain ────────────────────────────────────────
    domain = classify_domain(text)
    print(f"\n[scorer] Domain: {domain}")

    # ── Step 2: Infer topic ────────────────────────────────────────────
    topic = infer_topic(text)

    # ── Step 3: Fetch domain-appropriate facts ─────────────────────────
    facts = fetch_facts_for_domain(domain, topic)

    if not facts:
        return {
            "error":  f"No facts found for topic '{topic}' in domain '{domain}'",
            "topic":  topic,
            "domain": domain,
        }

    # ── Step 4: Create domain-bound embed function ─────────────────────
    # CRITICAL: this callable is used for BOTH building and querying the
    # collection. Never split them (e.g. build with PubMedBERT, query with Gemini).
    embed_fn = get_embed_fn(domain)
    print(f"[scorer] embed_fn bound to domain='{domain}'")

    # ── Step 5: Split sentences ────────────────────────────────────────
    sentences = split_sentences(text)
    if not sentences:
        return {
            "error":  "No scoreable sentences found.",
            "topic":  topic,
            "domain": domain,
        }

    # ── Step 6: Build vector store ─────────────────────────────────────
    collection = build_collection(topic, facts, embed_fn)

    # ── Step 7: Pre-compute embeddings + closest ───────────────────────
    print(f"\n[scorer] Pre-computing embeddings for {len(sentences)} sentence(s)...")
    sentence_data = []
    for sentence in sentences:
        emb     = embed_fn(sentence)
        closest = retrieve_closest(
            collection, sentence, n=3,
            embed_fn=embed_fn,
            query_embedding=emb,
        )
        sentence_data.append({
            "sentence":  sentence,
            "embedding": emb,
            "closest":   closest,
        })

    # ── Step 8: Batch NLI (1 API call for all sentences) ──────────────
    print(f"[scorer] Running batch NLI for {len(sentences)} sentence(s)...")
    nli_pairs   = [
        {"claim": d["sentence"], "facts": [f["fact"] for f in d["closest"]]}
        for d in sentence_data
    ]
    nli_results = check_nli_batch(nli_pairs)

    # ── Step 9: Score ──────────────────────────────────────────────────
    print(f"\n[scorer] Scoring {len(sentences)} sentence(s) against topic: '{topic}'")
    results = []
    for i, d in enumerate(sentence_data):
        print(f"\n[scorer] ── Sentence {i+1}/{len(sentences)}: '{d['sentence'][:60]}...'")
        result = score_sentence(
            d["sentence"],
            collection,
            embed_fn,                          # pass domain embed_fn
            _embedding  = d["embedding"],
            _closest    = d["closest"],
            _nli_result = nli_results[i],
        )
        result["domain"] = domain             # attach domain to each result
        results.append(result)
        print(f"[scorer]    final={result['final_score']}  "
              f"nli={result['nli_verdict']}  "
              f"consistency={result['consistency_score']}  "
              f"label={result['label']}  "
              f"source={result['evidence_source']}")

    # ── Step 10: Aggregate ─────────────────────────────────────────────
    hallucinated  = [r for r in results if r["label"] == "hallucinated"]
    overall_score = round(sum(r["final_score"] for r in results) / len(results), 4)
    overall_label = "likely hallucinated" if overall_score < 0.50 else "mostly grounded"

    return {
        "topic":              topic,
        "domain":             domain,           # NEW
        "overall_score":      overall_score,
        "overall_label":      overall_label,
        "sentence_count":     len(results),
        "hallucinated_count": len(hallucinated),
        "grounded_count":     len(results) - len(hallucinated),
        "results":            results,
    }


# ── Tests ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    def header(title: str):
        print(f"\n{'═' * 65}")
        print(f"  {title}")
        print(f"{'═' * 65}")

    cases = [
        # (description, text, expected_domain, expected_label)
        (
            "General — mixed (1 hallucinated + 1 true)",
            "Einstein won the Nobel Prize for the theory of relativity. "
            "He was born on 14 March 1879 in Ulm, Germany.",
            "general",
        ),
        (
            "General — fully grounded",
            "Albert Einstein was a German-born theoretical physicist. "
            "He received the Nobel Prize in Physics in 1921.",
            "general",
        ),
        (
            "General — Satya Nadella Nobel (should be hallucinated)",
            "Satya Nadella won the Nobel Prize.",
            "general",
        ),
        (
            "Medical — antibiotic claim (domain routing test)",
            "Amoxicillin is used to treat bacterial infections by inhibiting "
            "cell wall synthesis.",
            "medical",
        ),
        (
            "Legal — Fourth Amendment (domain routing test)",
            "The Fourth Amendment of the United States Constitution protects "
            "citizens from unreasonable searches and seizures.",
            "legal",
        ),
    ]

    for desc, text, expected_domain in cases:
        header(f"TEST: {desc}")
        out = analyze_text(text)
        if "error" in out:
            print(f"  ERROR: {out['error']}")
        else:
            domain_check = "✓" if out["domain"] == expected_domain else f"✗ (got {out['domain']}, expected {expected_domain})"
            print(f"\n  Topic:    {out['topic']}")
            print(f"  Domain:   {out['domain']}  {domain_check}")
            print(f"  Overall:  {out['overall_score']}  →  {out['overall_label']}")
            for r in out["results"]:
                icon = "🔴" if r["label"] == "hallucinated" else "🟢"
                src  = f"[{r['evidence_source']}]"
                print(f"  {icon} [{r['final_score']}] nli={r['nli_verdict']} "
                      f"con={r['consistency_score']} {src} | {r['sentence'][:60]}")