
# core/vector_store.py

import chromadb
from core.embedder import embed

# ── Client Setup ───────────────────────────────────────────────────────
# Module-level client — created once when this module is imported.
# In-memory: all data is lost when the Python process ends.
# This is correct for our use case — we rebuild from Wikipedia each request.
client = chromadb.Client()


# ── Core Functions ──────────────────────────────────────────────────────

def build_collection(topic: str, facts: list[dict], embed_fn=None):
    from core.embedder import embed as default_embed
    embed_fn = embed_fn or default_embed
    
    embeddings = [embed_fn(f["text"]) for f in facts]   # ← injectable
    """
    Creates (or retrieves) a ChromaDB collection for a topic and
    populates it with embedded Wikipedia facts.

    topic: string used to name the collection (e.g. "Albert Einstein")
    facts: list of sentences from fetch_facts()

    Returns: the ChromaDB collection object (used by retrieve_closest)
    """

    # Collection names must be 3-63 chars, alphanumeric + underscores/hyphens.
    # Topic strings like "Albert Einstein" contain spaces — sanitize them.
    # [:20] to stay under the 63-char limit.
    safe_name = "facts_" + topic.lower().replace(" ", "_")[:20]

    # get_or_create: safe for repeated calls in the same process.
    # create_collection would crash if called twice with the same name.
    collection = client.get_or_create_collection(name=safe_name)

    # Only populate if empty — prevents duplicate entries on repeat calls.
    # count() returns number of documents currently stored.
    if collection.count() == 0:
        print(f"[vector_store] Building collection '{safe_name}' with {len(facts)} facts...")

        # Embed all facts — this is the expensive step (N API calls to Gemini).
        # Done once per topic per session.
        embeddings = [embed(f) for f in facts]

        # add() takes three parallel lists:
        #   documents: the raw text (returned in query results for display)
        #   embeddings: the vectors (used for similarity search)
        #   ids: unique string identifiers per document
        collection.add(
            documents=facts,
            embeddings=embeddings,
            ids=[f"f{i}" for i in range(len(facts))]
        )
        print(f"[vector_store] Collection built. {collection.count()} documents stored.")
    else:
        print(f"[vector_store] Collection '{safe_name}' already exists ({collection.count()} docs). Reusing.")

    return collection


def retrieve_closest(collection, claim: str, n: int = 3, embed_fn=None) -> list[dict]:
    from core.embedder import embed as default_embed
    embed_fn = embed_fn or default_embed
    claim_embedding = embed_fn(claim)   # ← was: embed(claim)
    ...
    """
    Finds the n Wikipedia facts most semantically similar to the claim.

    query_embedding: pass pre-computed embedding to skip redundant API call.
    If None, computes it internally (backward compatible).
    """

    # Use pre-computed embedding if provided — saves 1 embed API call
    # when scorer.py already computed embed(sentence) for other purposes
    claim_embedding = query_embedding if query_embedding is not None else embed(claim)

    results = collection.query(
        query_embeddings=[claim_embedding],
        n_results=min(n, collection.count())
    )

    closest = [
        {
            "fact": doc,
            "distance": round(dist, 4)
        }
        for doc, dist in zip(
            results['documents'][0],
            results['distances'][0]
        )
    ]

    return closest


# ── Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from core.fetcher import fetch_facts

    print("═" * 55)
    print("TEST 1: Build collection from Wikipedia facts")
    print("═" * 55)

    facts = fetch_facts("Albert Einstein")
    print(f"\nFetched {len(facts)} facts from Wikipedia")

    collection = build_collection("Albert Einstein", facts)

    # ── TEST 2: Retrieve closest fact to a TRUE claim ──────────────────
    print("\n" + "═" * 55)
    print("TEST 2: True claim — should find close match")
    print("═" * 55)

    true_claim = "Einstein received the Nobel Prize in Physics"
    closest = retrieve_closest(collection, true_claim, n=3)

    print(f"\nClaim: '{true_claim}'")
    print("\nTop 3 closest facts:")
    for i, item in enumerate(closest):
        print(f"  [{i+1}] distance={item['distance']}  →  {item['fact']}")

    # ── TEST 3: Retrieve closest fact to a FALSE/HALLUCINATED claim ────
    print("\n" + "═" * 55)
    print("TEST 3: Hallucinated claim — should find poor match")
    print("═" * 55)

    false_claim = "Einstein won the Nobel Prize for the theory of relativity"
    closest2 = retrieve_closest(collection, false_claim, n=3)

    print(f"\nClaim: '{false_claim}'")
    print("\nTop 3 closest facts:")
    for i, item in enumerate(closest2):
        print(f"  [{i+1}] distance={item['distance']}  →  {item['fact']}")

    # ── TEST 4: Repeat call — should reuse, not rebuild ────────────────
    print("\n" + "═" * 55)
    print("TEST 4: Repeat build_collection — should reuse existing")
    print("═" * 55)
    collection2 = build_collection("Albert Einstein", facts)
    print(f"Collection count: {collection2.count()} (should be same as before)")