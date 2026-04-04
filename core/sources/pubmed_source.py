# core/sources/pubmed_source.py

import re
import requests

# ── PubMed E-utilities API ─────────────────────────────────────────────
# Completely free, no API key required
# NCBI (National Center for Biotechnology Information) hosts this
# Rate limit: 3 requests/second without key, 10/second with NCBI API key
# We stay well under this with sequential calls
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# How many articles to fetch — 3 gives enough sentences without being slow
MAX_ARTICLES = 3


def fetch_pubmed_facts(topic: str) -> list[dict]:
    """
    Searches PubMed for a topic and returns sentences from article abstracts.
    
    Two-step process:
        1. ESearch: get article IDs matching the topic
        2. EFetch: get abstract text for those IDs
    
    Returns list of:
        {"text": sentence, "source": "PubMed", "url": pubmed_article_url}
    
    Why PubMed over Wikipedia for medical claims?
    PubMed abstracts are peer-reviewed, citable, and authoritative.
    Wikipedia medical articles are simplified and sometimes outdated.
    For a hallucination detector in medical context, peer-reviewed
    literature is the correct ground truth.
    """

    # ── Step 1: Search for article IDs ────────────────────────────────
    search_params = {
        "db":       "pubmed",
        "term":     topic,
        "retmax":   MAX_ARTICLES,    # top N most relevant
        "retmode":  "json",
        "sort":     "relevance"      # best match first
    }

    try:
        r = requests.get(ESEARCH_URL, params=search_params, timeout=10)
        r.raise_for_status()
        ids = r.json()["esearchresult"]["idlist"]
    except requests.Timeout:
        print(f"[pubmed] Search timed out for: '{topic}'")
        return []
    except Exception as e:
        print(f"[pubmed] Search failed for '{topic}': {e}")
        return []

    if not ids:
        print(f"[pubmed] No articles found for: '{topic}'")
        return []

    print(f"[pubmed] Found article IDs: {ids}")

    # ── Step 2: Fetch abstracts for those IDs ─────────────────────────
    # EFetch returns plain text (retmode=text, rettype=abstract)
    # One call fetches all IDs — comma-separated
    fetch_params = {
        "db":      "pubmed",
        "id":      ",".join(ids),
        "rettype": "abstract",
        "retmode": "text"
    }

    try:
        r = requests.get(EFETCH_URL, params=fetch_params, timeout=15)
        r.raise_for_status()
        raw_text = r.text
    except requests.Timeout:
        print(f"[pubmed] Abstract fetch timed out for IDs: {ids}")
        return []
    except Exception as e:
        print(f"[pubmed] Abstract fetch failed: {e}")
        return []

    # ── Step 3: Split into sentences and attach metadata ──────────────
    # raw_text is one big string containing all abstracts concatenated
    # Split on sentence-ending punctuation followed by space
    all_sentences = re.split(r'(?<=[.!?]) +', raw_text)

    facts = []
    # Distribute sentences across article IDs for URL attribution
    # Each article gets ~sentences_per_article sentences
    # This is approximate — we don't have exact boundaries between abstracts
    sentences_per_article = max(1, len(all_sentences) // len(ids))

    for i, pmid in enumerate(ids):
        # Slice the sentences roughly attributed to this article
        start = i * sentences_per_article
        end   = start + sentences_per_article if i < len(ids) - 1 else len(all_sentences)
        chunk = all_sentences[start:end]

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        for s in chunk:
            s = s.strip()
            # Filter out:
            # - Very short fragments (< 40 chars) — not real sentences
            # - Lines that are just metadata (PMID:, Author:, etc.)
            # - Lines starting with digits (article numbers in plain text format)
            if (len(s) > 40
                    and not s.startswith("PMID")
                    and not s.startswith("Author")
                    and not s[:1].isdigit()):
                facts.append({
                    "text":   s,
                    "source": "PubMed",
                    "url":    url
                })

    print(f"[pubmed] Extracted {len(facts)} facts for: '{topic}'")
    return facts