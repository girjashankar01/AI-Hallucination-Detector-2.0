# core/sources/wikipedia_source.py

import re
import wikipediaapi

# ── Wikipedia Client ───────────────────────────────────────────────────
# Same user_agent requirement as original fetcher.py
# Wikipedia blocks requests without a proper user agent string
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='HallucinationDetector/2.0 (student-hackathon-project)'
)


def fetch_wikipedia_facts(topic: str) -> list[dict]:
    """
    Fetches Wikipedia summary for a topic and splits into sentences.
    
    Returns list of:
        {"text": sentence, "source": "Wikipedia", "url": article_url}
    
    Returns [] if article not found — caller handles fallback.
    
    Why return dicts instead of plain strings (like original fetcher.py)?
    The scorer needs to show citations in the frontend. Every fact needs
    its source and URL attached at fetch time — not reconstructed later.
    """
    page = wiki.page(topic)

    if not page.exists():
        print(f"[wikipedia] No article found for: '{topic}'")
        return []

    # Split summary into individual sentences
    # Same regex as original fetcher.py — lookbehind on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?]) +', page.summary)

    facts = []
    for s in sentences:
        s = s.strip()
        # Filter fragments shorter than 30 chars — artifacts of splitting
        # e.g. "Jr." or "U.S." getting split incorrectly
        if len(s) > 30:
            facts.append({
                "text":   s,
                "source": "Wikipedia",
                "url":    page.fullurl    # direct link to the article
            })

    print(f"[wikipedia] Found {len(facts)} facts for: '{topic}'")
    return facts