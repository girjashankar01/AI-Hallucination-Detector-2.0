# core/nli.py

import os
import re
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-3.1-flash-lite-preview"


def classify_nli(premise: str, hypothesis: str) -> str:
    """
    Classifies logical relationship: premise (Wikipedia fact) vs hypothesis (claim).

    Returns: "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL"

    NEUTRAL is correct when the Wikipedia fact is simply off-topic.
    e.g. "Satya Nadella is a business executive" vs "Satya Nadella won Nobel Prize"
    → NEUTRAL (fact doesn't address the claim at all, not a contradiction).
    This is expected and correct. The consistency signal handles these cases.
    """
    prompt = (
        "You are a strict NLI (Natural Language Inference) classifier.\n"
        "Given a PREMISE and a HYPOTHESIS, output exactly ONE word.\n\n"
        f"PREMISE: {premise}\n"
        f"HYPOTHESIS: {hypothesis}\n\n"
        "Rules:\n"
        "- ENTAILMENT    → premise directly supports or confirms the hypothesis\n"
        "- CONTRADICTION → premise directly refutes or contradicts the hypothesis\n"
        "  (Only use CONTRADICTION if there is a clear, direct conflict — not just absence of info)\n"
        "- NEUTRAL       → premise is off-topic or does not address the hypothesis either way\n\n"
        "Output ONLY one word. No explanation. No punctuation."
    )

    result = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config={"temperature": 0.1}
    )

    raw = result.text.strip().upper()
    for label in ("ENTAILMENT", "CONTRADICTION", "NEUTRAL"):
        if label in raw:
            return label

    print(f"  [nli] WARNING: unexpected output '{raw}', defaulting to NEUTRAL")
    return "NEUTRAL"


def check_nli(claim: str, facts: list[str]) -> dict:
    """
    Runs NLI between the claim and each fact.

    Priority: CONTRADICTION > ENTAILMENT > NEUTRAL
    Any CONTRADICTION is sufficient to reject the claim.
    """
    print(f"  [nli] Checking {len(facts)} facts against claim...")

    labels = []
    for fact in facts:
        label = classify_nli(premise=fact, hypothesis=claim)
        labels.append((fact, label))
        print(f"  [nli] {label:13s} | {fact[:75]}")

    found = [l for _, l in labels]

    if "CONTRADICTION" in found:
        cf = next(f for f, l in labels if l == "CONTRADICTION")
        return {"nli_score": 0.0, "verdict": "CONTRADICTION", "nli_labels": labels, "contradicting_fact": cf}

    if "ENTAILMENT" in found:
        return {"nli_score": 1.0, "verdict": "ENTAILMENT", "nli_labels": labels, "contradicting_fact": None}

    return {"nli_score": 0.5, "verdict": "NEUTRAL", "nli_labels": labels, "contradicting_fact": None}


def check_nli_batch(pairs: list[dict]) -> list[dict]:
    """
    Batch NLI: 1 API call for all (claim, facts) pairs.

    pairs: [{"claim": str, "facts": [str, ...]}, ...]
    Returns: list of dicts matching check_nli() output format.

    Key prompt improvement: explicitly tells the model that NEUTRAL means
    "off-topic or irrelevant", not "weakly contradicts". This prevents
    over-triggering CONTRADICTION on tangentially related facts.
    """
    if not pairs:
        return []

    items_text = ""
    for i, p in enumerate(pairs):
        facts_fmt = "\n".join(f"    F{j}: {f}" for j, f in enumerate(p["facts"]))
        items_text += f"\nItem {i}:\n  Claim: \"{p['claim']}\"\n  Facts:\n{facts_fmt}\n"

    prompt = (
        "You are a strict NLI (Natural Language Inference) classifier.\n"
        "For each item, determine the relationship between each fact and the claim.\n\n"
        "Classification rules (apply strictly):\n"
        "- ENTAILMENT    → fact directly supports or confirms the claim\n"
        "- CONTRADICTION → fact directly and clearly refutes the claim\n"
        "  IMPORTANT: Use CONTRADICTION only when the fact explicitly states something\n"
        "  that conflicts with the claim. Absence of information is NOT a contradiction.\n"
        "  Example of NEUTRAL (not CONTRADICTION): claim='X won Nobel Prize',\n"
        "  fact='X is a business executive' → NEUTRAL (doesn't address Nobel Prize at all)\n"
        "  Example of CONTRADICTION: claim='Einstein won Nobel for relativity',\n"
        "  fact='Einstein won Nobel for the photoelectric effect' → CONTRADICTION\n"
        "- NEUTRAL       → fact is off-topic or does not directly address the claim\n\n"
        "Return ONLY a JSON array — no markdown, no backticks, no explanation.\n"
        f"Exactly {len(pairs)} objects, one per item:\n"
        "  {\n"
        "    \"verdict\": \"CONTRADICTION\" | \"ENTAILMENT\" | \"NEUTRAL\",\n"
        "    \"contradicting_fact\": \"<exact text of contradicting fact, or empty string>\",\n"
        "    \"nli_labels\": [\"LABEL_FOR_F0\", \"LABEL_FOR_F1\", ...]\n"
        "  }\n\n"
        "Priority per item: CONTRADICTION > ENTAILMENT > NEUTRAL.\n"
        f"Items:{items_text}\n"
        f"Return exactly {len(pairs)} objects."
    )

    result = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config={"temperature": 0.1}
    )

    raw = result.text.strip()
    raw = re.sub(r'^```json?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("not a list")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [nli_batch] JSON parse failed: {e} — NEUTRAL fallback for all")
        return [
            {"nli_score": 0.5, "verdict": "NEUTRAL", "contradicting_fact": None, "nli_labels": []}
            for _ in pairs
        ]

    results = []
    score_map = {"CONTRADICTION": 0.0, "ENTAILMENT": 1.0, "NEUTRAL": 0.5}

    for i, item in enumerate(parsed[:len(pairs)]):
        verdict = item.get("verdict", "NEUTRAL").strip().upper()
        if verdict not in score_map:
            verdict = "NEUTRAL"

        contradicting_fact = item.get("contradicting_fact") or None
        if contradicting_fact == "":
            contradicting_fact = None

        raw_labels = item.get("nli_labels", [])
        facts = pairs[i]["facts"]
        nli_labels = [
            (facts[j], raw_labels[j] if j < len(raw_labels) else "NEUTRAL")
            for j in range(len(facts))
        ]

        results.append({
            "nli_score":          score_map[verdict],
            "verdict":            verdict,
            "nli_labels":         nli_labels,
            "contradicting_fact": contradicting_fact,
        })
        print(f"  [nli_batch] Item {i}: verdict={verdict}")

    while len(results) < len(pairs):
        results.append({"nli_score": 0.5, "verdict": "NEUTRAL", "contradicting_fact": None, "nli_labels": []})

    return results
