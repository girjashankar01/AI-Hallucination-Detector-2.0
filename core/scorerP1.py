# core/scorer.py

import re
import os
from dotenv import load_dotenv
from google import genai
from core.embedder import embed, cosine_similarity
from core.fetcher import fetch_facts, infer_topic
from core.vector_store import build_collection, retrieve_closest
from core.consistency import check_consistency
from core.nli import check_nli, check_nli_batch

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── Score weights ──────────────────────────────────────────────────────
# Rationale:
#   NLI (0.40)         — direct logical entailment/contradiction vs Wikipedia facts.
#                        Most reliable when a relevant fact exists.
#   Consistency (0.35) — Gemini's factual verdict (TRUE/FALSE) across 3 evals.
#                        Catches hallucinations when Wikipedia has no direct fact.
#                        (e.g. "Satya Nadella won Nobel" — NLI is NEUTRAL, but
#                         Gemini says FALSE 3 times → consistency=0.0)
#   Grounding (0.15)   — semantic proximity to the nearest Wikipedia sentence.
#                        Weak signal: topical relevance ≠ factual correctness.
#                        "Einstein won Nobel for relativity" is close to
#                        "Einstein won Nobel for photoelectric effect" → high score.
#                        Kept low, used mainly as a "topic covered" indicator.
#   Embedding (0.10)   — cosine similarity of claim vs best Wikipedia fact.
#                        Nearly redundant with grounding (both use same embedding).
#                        Minimal weight.
_W_NLI         = 0.40
_W_CONSISTENCY = 0.35
_W_GROUNDING   = 0.15
_W_EMBEDDING   = 0.10

# Hard caps — applied AFTER weighted sum regardless of other scores
_CAP_CONTRADICTION = 0.25   # NLI says CONTRADICTION → score cannot exceed 0.25
_CAP_ALL_FALSE     = 0.30   # All consistency verdicts are FALSE → cap at 0.30
_HALLUC_THRESHOLD  = 0.45   # below this → label as hallucinated


# ── Sentence Splitter ──────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in re.split(r'(?<=[.!?]) +', text)
        if len(s.strip()) > 20
    ]


# ── Per-Sentence Scorer ────────────────────────────────────────────────

def score_sentence(
    sentence:    str,
    collection,
    _embedding:  list | None = None,
    _closest:    list | None = None,
    _nli_result: dict | None = None,
) -> dict:

    # ── Method 1: Wikipedia Grounding ─────────────────────────────────
    sentence_embedding = _embedding if _embedding is not None else embed(sentence)
    closest = _closest if _closest is not None else retrieve_closest(
        collection, sentence, n=3, query_embedding=sentence_embedding
    )

    best_distance  = closest[0]['distance']
    grounding_score = round(1 / (1 + best_distance), 4)

    # ── Method 2: Self-Consistency (fixed) ────────────────────────────
    # consistency_score = fraction of TRUE verdicts from Gemini.
    # All FALSE → 0.0, All TRUE → 1.0.
    consistency_result = check_consistency(sentence, n=3)
    consistency_score  = consistency_result['consistency_score']
    all_false          = consistency_result['false_count'] == 3

    # ── Method 3: Embedding Similarity ────────────────────────────────
    fact_embedding  = embed(closest[0]['fact'])
    raw_cosine      = cosine_similarity(sentence_embedding, fact_embedding)
    embedding_score = round((raw_cosine + 1) / 2, 4)

    # ── Method 4: NLI ─────────────────────────────────────────────────
    if _nli_result is not None:
        nli_result = _nli_result
    else:
        top_facts_text = [item['fact'] for item in closest]
        nli_result = check_nli(claim=sentence, facts=top_facts_text)

    nli_score = nli_result['nli_score']

    # ── Weighted combination ───────────────────────────────────────────
    final_score = round(
        grounding_score   * _W_GROUNDING   +
        consistency_score * _W_CONSISTENCY +
        embedding_score   * _W_EMBEDDING   +
        nli_score         * _W_NLI,
        4
    )

    # ── Hard caps (strict priority) ────────────────────────────────────
    # CONTRADICTION from NLI is a definitive signal — one Wikipedia fact
    # directly refutes the claim. Cap hard regardless of other scores.
    if nli_result['verdict'] == "CONTRADICTION":
        final_score = min(final_score, _CAP_CONTRADICTION)

    # All 3 Gemini consistency evals say FALSE — Gemini's world-knowledge
    # unanimously rejects the claim. Cap even if NLI is NEUTRAL.
    elif all_false:
        final_score = min(final_score, _CAP_ALL_FALSE)

    # ── Label ──────────────────────────────────────────────────────────
    label = "hallucinated" if final_score < _HALLUC_THRESHOLD else "grounded"

    return {
        "sentence":              sentence,
        "final_score":           final_score,
        "label":                 label,
        "grounding_score":       grounding_score,
        "consistency_score":     consistency_score,
        "embedding_score":       embedding_score,
        "nli_score":             nli_score,
        "nli_verdict":           nli_result['verdict'],
        "evidence":              closest[0]['fact'],
        "evidence_distance":     closest[0]['distance'],
        "top_facts":             closest,
        "contradicting_fact":    nli_result['contradicting_fact'],
        "nli_labels":            nli_result['nli_labels'],
        "consistency_responses": consistency_result['responses'],
        "consistency_verdicts":  consistency_result['verdicts'],
    }


# ── Full Text Analyzer ─────────────────────────────────────────────────

def analyze_text(text: str) -> dict:
    """
    Main entry point.

    Workflow:
    1. Infer Wikipedia topic
    2. Fetch facts (3-tier fallback: exact → search → short-query search)
    3. Build ChromaDB collection
    4. Pre-compute embeddings + closest facts for all sentences
    5. Batch NLI (1 API call)
    6. Score each sentence
    7. Aggregate
    """

    # ── Step 1: Infer topic ────────────────────────────────────────────
    topic = infer_topic(text)

    # ── Step 2: Fetch facts with fallback ─────────────────────────────
    facts = fetch_facts(topic)

    if not facts:
        return {
            "error": f"No Wikipedia article found for inferred topic: '{topic}'",
            "topic": topic,
        }

    # ── Step 3: Split sentences ────────────────────────────────────────
    sentences = split_sentences(text)
    if not sentences:
        return {
            "error": "No scoreable sentences found (all too short).",
            "topic": topic,
        }

    # ── Step 4: Build vector store ─────────────────────────────────────
    collection = build_collection(topic, facts)

    # ── Step 5a: Pre-compute embeddings + closest ──────────────────────
    print(f"\n[scorer] Pre-computing embeddings for {len(sentences)} sentence(s)...")
    sentence_data = []
    for sentence in sentences:
        emb     = embed(sentence)
        closest = retrieve_closest(collection, sentence, n=3, query_embedding=emb)
        sentence_data.append({"sentence": sentence, "embedding": emb, "closest": closest})

    # ── Step 5b: Batch NLI ─────────────────────────────────────────────
    print(f"[scorer] Running batch NLI for {len(sentences)} sentence(s)...")
    nli_pairs   = [{"claim": d["sentence"], "facts": [f["fact"] for f in d["closest"]]} for d in sentence_data]
    nli_results = check_nli_batch(nli_pairs)

    # ── Step 5c: Score ─────────────────────────────────────────────────
    print(f"\n[scorer] Scoring {len(sentences)} sentence(s) against topic: '{topic}'")
    results = []
    for i, d in enumerate(sentence_data):
        print(f"\n[scorer] ── Sentence {i+1}/{len(sentences)}: '{d['sentence'][:60]}...'")
        result = score_sentence(
            d["sentence"], collection,
            _embedding=d["embedding"],
            _closest=d["closest"],
            _nli_result=nli_results[i],
        )
        results.append(result)
        print(f"[scorer]    final={result['final_score']}  nli={result['nli_verdict']}  "
              f"consistency={result['consistency_score']}  label={result['label']}")

    # ── Step 6: Aggregate ──────────────────────────────────────────────
    hallucinated  = [r for r in results if r['label'] == 'hallucinated']
    overall_score = round(sum(r['final_score'] for r in results) / len(results), 4)
    overall_label = "likely hallucinated" if overall_score < 0.50 else "mostly grounded"

    return {
        "topic":              topic,
        "overall_score":      overall_score,
        "overall_label":      overall_label,
        "sentence_count":     len(results),
        "hallucinated_count": len(hallucinated),
        "grounded_count":     len(results) - len(hallucinated),
        "results":            results,
    }


# ── Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        # (description, text)
        ("Mixed: 1 hallucinated + 1 true",
         "Einstein won the Nobel Prize for the theory of relativity. He was born on 14 March 1879 in Ulm, Germany."),

        ("Fully grounded",
         "Albert Einstein was a German-born theoretical physicist. He received the Nobel Prize in Physics in 1921."),

        ("Satya Nadella Nobel (should be hallucinated)",
         "Satya Nadella won the Nobel Prize."),

        ("GitHub acquisition (topic inference test)",
         "The acquisition of Microsoft by GitHub was finalized in 2018 for $7.5 billion."),
    ]

    for desc, text in cases:
        print("\n" + "═" * 65)
        print(f"TEST: {desc}")
        print("═" * 65)
        out = analyze_text(text)
        if "error" in out:
            print(f"  ERROR: {out['error']}")
        else:
            print(f"  Topic:    {out['topic']}")
            print(f"  Overall:  {out['overall_score']}  →  {out['overall_label']}")
            for r in out['results']:
                icon = "🔴" if r['label'] == 'hallucinated' else "🟢"
                print(f"  {icon} [{r['final_score']}] nli={r['nli_verdict']} "
                      f"con={r['consistency_score']} | {r['sentence'][:65]}")
