# core/vector_store.py

import chromadb
from core.embedder import embed as default_embed

# Module-level client — in-memory, rebuilds each session.
# Correct for our use case: we always rebuild from the source.
_client = chromadb.Client()


def build_collection(
    topic:          str,
    facts:          list[dict],   # list[{"text": str, "source": str, "url": str}]
    embed_fn        = None,       # callable(text: str) -> list[float]
                                  # If None: uses Gemini (default_embed)
                                  # Module E passes get_embed_fn(domain) here
    domain:         str = "general",  # Included in collection name to prevent
                                       # dimension collisions across domains for same topic.
    embed_batch_fn  = None,       # NEW — optional callable(texts: list[str]) -> list[list[float]]
                                  # When provided, used instead of N sequential embed_fn calls.
                                  # Reduces 25 Gemini API calls → 1 for the general domain.
                                  # Module E passes: lambda texts: batch_embed_for_domain(texts, domain)
) -> object:
    """
    Creates (or retrieves) a ChromaDB collection for a topic and
    populates it with embedded facts.

    facts: list[{"text": str, "source": str, "url": str}]
        All three keys are required. "source" and "url" are stored as
        ChromaDB metadata and surfaced in retrieve_closest() results.

    embed_fn: the embedding callable to use for ALL vectors in this collection.
        MUST match the embed_fn passed to retrieve_closest() for the same collection.
        Using different functions for build vs query produces wrong similarity scores.

    embed_batch_fn: optional batch version of embed_fn.
        If provided, all fact embeddings are computed in ONE call instead of N calls.
        This is the fix for the "25 sequential Gemini embed calls" issue — the general
        domain now passes batch_embed_for_domain(texts, "general") which uses
        the Gemini batch API (1 call for N facts).
        embed_fn is still required for retrieve_closest() (single-query embedding).

    domain: used only to namespace the collection name — prevents the singleton
        _client from returning a stale 768-dim collection when the domain changes.

    Returns: ChromaDB collection object.
    """
    embed_fn = embed_fn or default_embed

    # domain prefix isolates collections per embedding model
    safe_topic = topic.lower().replace(" ", "_")[:20].rstrip("_")
    safe_name  = f"facts_{domain}_{safe_topic}"
    collection = _client.get_or_create_collection(name=safe_name)

    if collection.count() == 0:
        texts     = [f["text"]   for f in facts]
        sources   = [f.get("source", "Unknown") for f in facts]
        urls      = [f.get("url",    "")         for f in facts]
        metadatas = [{"source": s, "url": u} for s, u in zip(sources, urls)]

        if embed_batch_fn is not None:
            # Batch path: 1 API call for all facts
            print(f"[vector_store] Building '{safe_name}' — batch embedding {len(texts)} facts (1 API call)...")
            embeddings = embed_batch_fn(texts)
        else:
            # Sequential path: N API calls (fallback when no batch fn provided)
            print(f"[vector_store] Building '{safe_name}' — embedding {len(texts)} facts sequentially...")
            embeddings = [embed_fn(t) for t in texts]

        collection.add(
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metadatas,
            ids        = [f"f{i}" for i in range(len(texts))],
        )
        print(f"[vector_store] Done. {collection.count()} documents stored.")
    else:
        print(f"[vector_store] '{safe_name}' already exists ({collection.count()} docs). Reusing.")

    return collection


def retrieve_closest(
    collection,
    claim:           str,
    n:               int  = 3,
    embed_fn               = None,        # same callable used in build_collection
    query_embedding: list | None = None,  # pass pre-computed to skip re-embedding
) -> list[dict]:
    """
    Finds the n facts most semantically similar to the claim.

    query_embedding: pass pre-computed embedding to avoid a redundant API call.
        scorer.py pre-computes embed_fn(sentence) for use here and elsewhere;
        passing it in saves one embed API call per sentence.

    Returns list[{"fact", "distance", "source", "url"}].
    """
    embed_fn = embed_fn or default_embed

    if query_embedding is not None:
        claim_embedding = query_embedding
    else:
        claim_embedding = embed_fn(claim)

    results = collection.query(
        query_embeddings = [claim_embedding],
        n_results        = min(n, collection.count()),
        include          = ["documents", "distances", "metadatas"],
    )

    return [
        {
            "fact":     doc,
            "distance": round(dist, 4),
            "source":   meta.get("source", "Unknown"),
            "url":      meta.get("url",    ""),
        }
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ]