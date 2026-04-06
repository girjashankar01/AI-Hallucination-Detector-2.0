# core/fetcher.py

import re
import os
import requests
import wikipediaapi
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='HallucinationDetector/1.0 (student-hackathon-project)'
)

_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_WIKI_TIMEOUT    = 6  # seconds


# ── Internal helpers ───────────────────────────────────────────────────

def _extract_facts(page) -> list[str]:
    """Splits a Wikipedia page summary into scoreable fact sentences."""
    sentences = re.split(r'(?<=[.!?]) +', page.summary)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def _wiki_search(query: str, limit: int = 5) -> list[str]:
    """
    Calls the MediaWiki search API and returns up to `limit` article titles.
    Returns [] on any network or parse error.
    """
    try:
        resp = requests.get(
            _WIKI_SEARCH_URL,
            params={
                "action":   "query",
                "list":     "search",
                "srsearch": query,
                "format":   "json",
                "srlimit":  limit,
            },
            headers={"User-Agent": "HallucinationDetector/1.0"},
            timeout=_WIKI_TIMEOUT,
        )
        resp.raise_for_status()
        return [item["title"] for item in resp.json()["query"]["search"]]
    except Exception as e:
        print(f"[fetcher] Wikipedia search failed: {e}")
        return []


def _try_fetch_page(title: str) -> list[str]:
    """
    Tries to fetch a Wikipedia page by exact title.
    Returns facts list (may be empty) or [] on error.
    """
    try:
        page = wiki.page(title)
        if page.exists():
            facts = _extract_facts(page)
            if facts:
                print(f"[fetcher] Loaded '{title}' ({len(facts)} facts)")
                return facts
    except Exception as e:
        print(f"[fetcher] Error fetching '{title}': {e}")
    return []


# ── Public API ─────────────────────────────────────────────────────────

def infer_topic(text: str) -> str:
    """
    Uses Gemini to extract the Wikipedia article title most relevant to the text.

    Old prompt: "What is the single main person, place, or thing?"
    Problem: for "acquisition of Microsoft by GitHub", Gemini returned
    "GitHub acquisition" — no such Wikipedia article exists.

    New prompt: asks for the *exact Wikipedia article title* by name.
    Also extracts the primary named entity as a simpler fallback candidate.
    """
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL"),
            contents=(
                "Your task is to identify the best Wikipedia article to fact-check the following text.\n"
                "Rules:\n"
                "1. Reply with the exact Wikipedia article title (e.g. 'Albert Einstein', 'GitHub', 'Python (programming language)').\n"
                "2. Prefer the primary named entity (person, company, product) over an event or relationship.\n"
                "3. If the text is about an acquisition, use the acquiree's article (e.g. 'GitHub' not 'GitHub acquisition').\n"
                "4. 1-4 words only. No punctuation. No explanation.\n"
                f"Text: \"{text}\""
            )
        )
        topic = response.text.strip().strip('."\'')
        print(f"[fetcher] Inferred topic: '{topic}'")
        return topic
    except Exception as e:
        # Last-resort: grab first capitalized word sequence from text
        print(f"[fetcher] infer_topic API error: {e}, using regex fallback")
        m = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
        return m.group(1) if m else text[:30]


def fetch_facts(topic: str) -> list[str]:
    """
    Fetches Wikipedia facts for a topic with a 3-tier fallback:

    Tier 1 — Exact title match (fast, 0 network calls if cached)
    Tier 2 — MediaWiki search on the inferred topic, try top-5 results
    Tier 3 — MediaWiki search on individual words from the topic

    Returns [] only if all tiers fail — caller decides what to do.
    """

    # Tier 1: exact title
    facts = _try_fetch_page(topic)
    if facts:
        return facts

    print(f"[fetcher] Exact match failed for '{topic}', trying search...")

    # Tier 2: search with the full topic string
    for title in _wiki_search(topic, limit=5):
        facts = _try_fetch_page(title)
        if facts:
            return facts

    # Tier 3: search with just the first significant word(s)
    # Handles cases like "GitHub acquisition" → search "GitHub"
    words = [w for w in topic.split() if len(w) > 3]
    if words:
        short_query = " ".join(words[:2])
        if short_query.lower() != topic.lower():
            print(f"[fetcher] Tier-3 fallback search: '{short_query}'")
            for title in _wiki_search(short_query, limit=3):
                facts = _try_fetch_page(title)
                if facts:
                    return facts

    print(f"[fetcher] All tiers exhausted for topic: '{topic}'")
    return []


# ── Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    cases = [
        "Einstein won the Nobel Prize for the theory of relativity",
        "the acquisition of Microsoft by GitHub was finalized in 2018 for $7.5 billion",
        "Satya Nadella won the Nobel Prize",
        "quantum mechanics aligns with classical physics",
        "Zyx Qrplmno invented the flux capacitor in 1985",  # unknown
    ]

    for text in cases:
        print("═" * 60)
        topic = infer_topic(text)
        facts = fetch_facts(topic)
        print(f"Text:   {text[:70]}")
        print(f"Topic:  {topic}")
        print(f"Facts:  {len(facts)} fetched")
        if facts:
            print(f"  [0] {facts[0]}")
        print()
