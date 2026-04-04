# core/embedder.py
from google import genai
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# New SDK: client-based instead of global configure
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed(text: str) -> list[float]:
    """
    Converts text into a vector using Gemini's embedding model.
    New Software Development Kit (SDK) uses client.models.embed_content() instead of genai.embed_content()
    Response structure is also different: result.embeddings[0].values
    """
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text           # new SDK uses 'contents' not 'content'
    )
    return result.embeddings[0].values   # new SDK: object attrs not dict keys


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Batch embed multiple texts in ONE Gemini API call.

    Fix for: 25 sequential Gemini calls when building the general-domain
    ChromaDB collection. Previously domain_embedder.batch_embed_for_domain()
    fell back to a loop for 'general' because "Gemini doesn't support batching".
    The new Gemini SDK DOES support batch via contents=[list of strings].

    Reduces N API calls → 1 call. Significant latency improvement for
    Wikipedia facts (typically 15-25 sentences).

    Returns list[list[float]] in the same order as input texts.
    """
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texts          # list[str] → batched, returns result.embeddings[N]
    )
    return [list(e.values) for e in result.embeddings]


def cosine_similarity(a: list, b: list) -> float:
    """
    Measures similarity between two vectors via angle between them.
    Returns float between -1 and 1.
    >0.85 = same meaning, <0.5 = different meaning
    """
    a, b = np.array(a), np.array(b)
    dot = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return float(dot / (magnitude_a * magnitude_b))

if __name__ == "__main__":
    print("Fetching embeddings from Gemini API...")

    e1 = embed("Gandhi was born in 1869")
    e2 = embed("Gandhi's birth year was 1869")
    e3 = embed("The Eiffel Tower is in Paris")

    print(f"\nVector length: {len(e1)} dimensions")
    print(f"First 5 values of e1: {[round(x, 4) for x in e1[:5]]}")

    print("\n── Similarity Scores ──")
    print(f"e1 vs e2 (same meaning):    { round(cosine_similarity(e1, e2), 4) }")
    print(f"e1 vs e3 (different topic): { round(cosine_similarity(e1, e3), 4) }")
    print(f"e1 vs e1 (identical):       { round(cosine_similarity(e1, e1), 4) }")

    print("\n── Batch Embed Test ──")
    texts = ["Gandhi was born in 1869", "The Eiffel Tower is in Paris", "Python is a programming language"]
    batch_result = embed_batch(texts)
    print(f"Batch returned {len(batch_result)} vectors, each dim={len(batch_result[0])}")
    # Verify batch[0] matches single embed[0]
    sim = cosine_similarity(e1, batch_result[0])
    print(f"Batch[0] vs single embed cosine: {round(sim, 4)} (expect ~1.0)")