# core/sources/__init__.py

from core.sources.wikipedia_source import fetch_wikipedia_facts
from core.sources.pubmed_source     import fetch_pubmed_facts
from core.sources.legal_source      import fetch_legal_facts


def fetch_facts_for_domain(domain: str, topic: str) -> list[dict]:
    """
    Routes to the appropriate fact source based on domain classification.
    
    domain: one of "medical" | "legal" | "financial" | "general"
    topic:  the extracted entity/subject to look up (from infer_topic())
    
    Returns list of {"text": str, "source": str, "url": str}
    
    Fallback chain:
        medical   → PubMed   → (if empty) Wikipedia
        legal     → CourtListener → (if empty) Wikipedia
        financial → Wikipedia  (yfinance skipped — adds complexity, demo is stable without it)
        general   → Wikipedia
    
    Why always fall back to Wikipedia and not return []?
    An empty fact list kills the entire pipeline — scorer gets no grounding signal.
    Wikipedia almost always has something. A weak grounding signal is better than
    no grounding signal — the embedding and consistency scores still work without it.
    """

    print(f"[sources] Domain: '{domain}' | Topic: '{topic}'")

    if domain == "medical":
        facts = fetch_pubmed_facts(topic)
        if not facts:
            print("[sources] PubMed returned nothing — falling back to Wikipedia")
            facts = fetch_wikipedia_facts(topic)

    elif domain == "legal":
        facts = fetch_legal_facts(topic)
        if not facts:
            print("[sources] CourtListener returned nothing — falling back to Wikipedia")
            facts = fetch_wikipedia_facts(topic)

    else:
        # financial and general both use Wikipedia
        # financial: Wikipedia has company/market articles
        # general:   Wikipedia is the correct source
        facts = fetch_wikipedia_facts(topic)

    if not facts:
        print(f"[sources] All sources empty for topic: '{topic}'")

    return facts