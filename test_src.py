# test_sources.py — run from project root: python test_sources.py

import time
from core.sources import fetch_facts_for_domain
from core.sources.wikipedia_source import fetch_wikipedia_facts
from core.sources.pubmed_source     import fetch_pubmed_facts
from core.sources.legal_source      import fetch_legal_facts


def print_facts(facts: list[dict], limit: int = 3):
    if not facts:
        print("  ← EMPTY — source returned nothing")
        return
    for i, f in enumerate(facts[:limit]):
        print(f"  [{i+1}] [{f['source']}] {f['text'][:90]}...")
        print(f"       URL: {f['url']}")


print("═" * 65)
print("TEST 1: Wikipedia source directly")
print("═" * 65)
facts = fetch_wikipedia_facts("Albert Einstein")
print_facts(facts)

time.sleep(1)

print("\n" + "═" * 65)
print("TEST 2: PubMed source directly")
print("═" * 65)
facts = fetch_pubmed_facts("amoxicillin pneumonia treatment")
print_facts(facts)

time.sleep(1)

print("\n" + "═" * 65)
print("TEST 3: CourtListener source directly")
print("═" * 65)
facts = fetch_legal_facts("fourth amendment unreasonable search")
print_facts(facts)

time.sleep(1)

print("\n" + "═" * 65)
print("TEST 4: Router — medical domain")
print("═" * 65)
facts = fetch_facts_for_domain("medical", "myocardial infarction LDL cholesterol")
print_facts(facts)

time.sleep(1)

print("\n" + "═" * 65)
print("TEST 5: Router — legal domain")
print("═" * 65)
facts = fetch_facts_for_domain("legal", "Miranda rights self-incrimination")
print_facts(facts)

time.sleep(1)

print("\n" + "═" * 65)
print("TEST 6: Router — general domain (should use Wikipedia)")
print("═" * 65)
facts = fetch_facts_for_domain("general", "Albert Einstein")
print_facts(facts)

time.sleep(1)

print("\n" + "═" * 65)
print("TEST 7: Fallback — PubMed with nonsense topic")
print("═" * 65)
facts = fetch_facts_for_domain("medical", "xyzzy nonexistent disease 99999")
print(f"  Facts returned: {len(facts)}")
if facts:
    print(f"  Source used: {facts[0]['source']}  ← should be Wikipedia (fallback)")