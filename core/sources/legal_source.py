# core/sources/legal_source.py

import re
import requests

# ── CourtListener API ──────────────────────────────────────────────────
# Free, open database of US federal and state court opinions
# No API key required for basic search
# Docs: https://www.courtlistener.com/help/api/
COURTLISTENER_SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"

# Required — CourtListener blocks requests without a user agent
HEADERS = {
    "User-Agent": "HallucinationDetector/2.0 (student-hackathon-project)"
}


def fetch_legal_facts(topic: str) -> list[dict]:
    """
    Searches CourtListener for US court opinions relevant to a legal topic.
    Extracts text snippets from opinion excerpts.
    
    Returns list of:
        {"text": sentence, "source": "CourtListener", "url": opinion_url}
    
    Why CourtListener over Wikipedia for legal claims?
    Legal accuracy requires citation to actual case law, not encyclopedia summaries.
    "The defendant has the right to remain silent" needs grounding in
    Miranda v. Arizona (384 U.S. 436), not a Wikipedia paragraph about it.
    CourtListener provides direct links to actual court opinions — judges can
    click through and verify the source, which is a strong demo moment.
    """

    params = {
        "q":        topic,
        "type":     "o",              # "o" = opinions (actual court decisions)
                                      # other types: "r" = recap, "p" = people
        "order_by": "score desc",     # most relevant first
        "format":   "json"
    }

    try:
        r = requests.get(
            COURTLISTENER_SEARCH_URL,
            params=params,
            headers=HEADERS,
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])

    except requests.Timeout:
        print(f"[legal] CourtListener timed out for: '{topic}'")
        return []
    except Exception as e:
        print(f"[legal] CourtListener failed for '{topic}': {e}")
        return []

    if not results:
        print(f"[legal] No court opinions found for: '{topic}'")
        return []

    print(f"[legal] Found {len(results)} court opinions for: '{topic}'")

    facts = []

    for result in results[:3]:    # top 3 most relevant opinions
        # snippet: HTML excerpt from the opinion text
        # caseName: e.g. "Miranda v. Arizona"
        # absolute_url: relative path like "/opinion/12345/miranda-v-arizona/"
        snippet   = result.get("snippet", "") or ""
        case_name = result.get("caseName", "Unknown Case")
        rel_url   = result.get("absolute_url", "")
        full_url  = f"https://www.courtlistener.com{rel_url}"

        # snippet contains HTML highlight tags like <mark>search term</mark>
        # Strip all HTML tags to get clean text
        clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()

        if not clean_snippet:
            # If no snippet, use case name as a minimal fact
            if len(case_name) > 10:
                facts.append({
                    "text":   f"Case: {case_name}",
                    "source": "CourtListener",
                    "url":    full_url
                })
            continue

        # Split snippet into sentences
        sentences = re.split(r'(?<=[.!?]) +', clean_snippet)

        for s in sentences:
            s = s.strip()
            if len(s) > 40:
                facts.append({
                    "text":   s,
                    "source": "CourtListener",
                    "url":    full_url
                })

    print(f"[legal] Extracted {len(facts)} facts for: '{topic}'")
    return facts