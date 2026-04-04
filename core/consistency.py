# core/consistency.py

import re
import json
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-3.1-flash-lite-preview"


def check_consistency(claim: str, n: int = 3) -> dict:
    """
    Asks Gemini to evaluate the claim n times and extract an explicit TRUE/FALSE
    verdict from each response.

    Previous bug: measured cosine similarity between the n responses.
    "The statement is false because X" and "The statement is false because Y"
    are semantically similar → similarity = 0.98 → treated as high confidence.
    That's the exact opposite of correct — high similarity on "false" responses
    is a hallucination signal, not a grounding signal.

    Fix: parse the verdict (TRUE/FALSE/UNCERTAIN) from each response.
    consistency_score = fraction of responses that say TRUE.
      All TRUE  → 1.0  (strong grounding signal)
      All FALSE → 0.0  (strong hallucination signal)
      Mixed     → 0.33 / 0.67 (uncertain)
    """

    prompt = (
        f"Evaluate this statement {n} times independently, as if you have no memory "
        f"of your previous evaluations.\n"
        f"Statement: \"{claim}\"\n\n"
        f"For each evaluation, determine if the statement is factually TRUE or FALSE.\n"
        f"Return ONLY a JSON array of exactly {n} objects. No markdown, no backticks, no preamble.\n"
        f"Format: "
        f'[{{"verdict": "TRUE", "reason": "one sentence"}}, ...]\n'
        f"verdict must be exactly \"TRUE\" or \"FALSE\"."
    )

    result = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": 0.7}
    )

    raw = result.text.strip()
    raw = re.sub(r'^```json?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("not a list")
        parsed = parsed[:n]
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract verdicts via regex from raw text
        print(f"  [consistency] JSON parse failed, falling back to regex")
        true_matches  = len(re.findall(r'\bTRUE\b',  raw, re.IGNORECASE))
        false_matches = len(re.findall(r'\bFALSE\b', raw, re.IGNORECASE))
        total = true_matches + false_matches or 1
        score = round(true_matches / total, 4)
        responses = [raw[:200]]
        return {
            "consistency_score": score,
            "verdicts":          ["TRUE"] * true_matches + ["FALSE"] * false_matches,
            "responses":         responses,
            "true_count":        true_matches,
            "false_count":       false_matches,
        }

    verdicts  = []
    responses = []
    for item in parsed:
        v = str(item.get("verdict", "FALSE")).strip().upper()
        if v not in ("TRUE", "FALSE"):
            v = "FALSE"  # unknown → treat as false, safer than treating as true
        verdicts.append(v)
        responses.append(str(item.get("reason", "")))

    # Pad if model returned fewer items
    while len(verdicts) < n:
        verdicts.append("FALSE")
        responses.append("")

    true_count  = sum(1 for v in verdicts if v == "TRUE")
    false_count = sum(1 for v in verdicts if v == "FALSE")

    # Score = fraction that say TRUE
    # This is now a genuine factual confidence signal.
    consistency_score = round(true_count / n, 4)

    print(f"  [consistency] verdicts: {verdicts}  →  score={consistency_score}")
    for i, (v, r) in enumerate(zip(verdicts, responses)):
        print(f"  [consistency] {i+1}. [{v}] {r}")

    return {
        "consistency_score": consistency_score,
        "verdicts":          verdicts,
        "responses":         responses,
        "true_count":        true_count,
        "false_count":       false_count,
    }


# ── Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 60)
    print("TEST 1: True fact — expect score ~1.0")
    print("═" * 60)
    r1 = check_consistency("Albert Einstein was born in 1879", n=3)
    print(f"  Score: {r1['consistency_score']}  verdicts: {r1['verdicts']}\n")

    print("═" * 60)
    print("TEST 2: Hallucinated claim — expect score ~0.0")
    print("═" * 60)
    r2 = check_consistency("Einstein won the Nobel Prize for the theory of relativity", n=3)
    print(f"  Score: {r2['consistency_score']}  verdicts: {r2['verdicts']}\n")

    print("═" * 60)
    print("TEST 3: Satya Nadella Nobel Prize (clear false) — expect score ~0.0")
    print("═" * 60)
    r3 = check_consistency("Satya Nadella won the Nobel Prize", n=3)
    print(f"  Score: {r3['consistency_score']}  verdicts: {r3['verdicts']}")
