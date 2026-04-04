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

# FIX: was 3 — often all 3 returned procedural case metadata (case name, docket
# number, court) rather than substantive legal text. Fetching 5 gives more
# chances of finding an opinion snippet with actual legal reasoning.
MAX_RESULTS = 5

# Minimum sentence length — procedural snippets tend to be very short or just
# case names. Increasing from 40 → 60 chars filters more metadata noise.
MIN_SENTENCE_LEN = 60


def _is_procedural(text: str) -> bool:
    """
    Returns True if a snippet looks like case metadata rather than legal text.

    CourtListener snippets often contain just the case name, docket number,
    or procedural boilerplate. These provide no useful grounding signal.

    Examples of procedural noise to filter:
      "Miranda v. Arizona, 384 U.S. 436 (1966)"
      "Case: 21-cv-01234 | Filed: January 5, 2022"
      "UNITED STATES DISTRICT COURT FOR THE DISTRICT OF COLUMBIA"
    """
    procedural_patterns = [
        r'^\s*\d{2,4}-cv-\d+',           # docket numbers like "21-cv-01234"
        r'^\s*Case\s+No',                  # "Case No. ..."
        r'DISTRICT COURT',
        r'CIRCUIT COURT',
        r'SUPREME COURT OF',
        r'^\s*v\.\s+[A-Z]',               # "v. Defendant" standalone
        r'^\s*\d+\s+[A-Z][a-z]+\.',       # "123 U.S. 456" citation style
        r'Filed:\s+\w+\s+\d+,\s+\d{4}',  # "Filed: January 5, 2022"
    ]
    return any(re.search(p, text) for p in procedural_patterns)


def fetch_legal_facts(topic: str) -> list[dict]:
    """
    Searches CourtListener for US court opinions relevant to a legal topic.
    Extracts text snippets from opinion excerpts.

    Returns list of:
        {"text": sentence, "source": "CourtListener", "url": opinion_url}

    FIX: Was fetching only top 3 results, getting ~3 procedural snippets.
    Now fetches top 5 results with procedural noise filtering, yielding
    more substantive legal text for NLI comparison.

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

    for result in results[:MAX_RESULTS]:
        snippet   = result.get("snippet", "") or ""
        case_name = result.get("caseName", "Unknown Case")
        rel_url   = result.get("absolute_url", "")
        full_url  = f"https://www.courtlistener.com{rel_url}"

        # Strip HTML highlight tags (<mark>...</mark> and others)
        clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()

        if not clean_snippet:
            # Minimal fallback: at least store the case name as a fact
            if len(case_name) > 10 and not _is_procedural(case_name):
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
            # Filter short fragments and procedural boilerplate
            if len(s) >= MIN_SENTENCE_LEN and not _is_procedural(s):
                facts.append({
                    "text":   s,
                    "source": "CourtListener",
                    "url":    full_url
                })

    print(f"[legal] Extracted {len(facts)} facts for: '{topic}'")

    # If we still got very few substantive snippets after filtering,
    # log a warning — caller (__init__.py) will fall back to Wikipedia
    if len(facts) < 2:
        print(f"[legal] WARNING: only {len(facts)} substantive snippet(s) found. "
              f"Caller will fall back to Wikipedia if 0.")

    return facts