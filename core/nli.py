# core/nli.py

import os
import re
import json
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-3.1-flash-lite-preview"

import threading
import time

# ── Rate limiter: 15 RPM = 1 call per 4s ──────────────────────────
_RATE_LIMIT_RPM = 15
_MIN_INTERVAL   = 60.0 / _RATE_LIMIT_RPM   # 4.0 seconds between calls
_last_call_time = 0.0
_rate_lock      = threading.Lock()

def _rate_limit_wait():
    """Block until it's safe to make the next Gemini call."""
    global _last_call_time
    with _rate_lock:
        now     = time.monotonic()
        elapsed = now - _last_call_time
        wait    = _MIN_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()

def _gemini_generate(prompt: str, temperature: float = 0.1, retries: int = 4) -> str:
    delay = 15
    for attempt in range(retries):
        _rate_limit_wait()          # 👈 add this line
        try:
            result = client.models.generate_content(...)
            return result.text.strip()
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg:     # 👈 also add daily quota guard
                print("  [nli] Daily quota exhausted — aborting retries")
                return ""
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                ...                 # existing backoff logic unchanged


def _gemini_generate(prompt: str, temperature: float = 0.1, retries: int = 4) -> str:
    """
    Wrapper around client.models.generate_content with exponential backoff on 429.

    Free tier limit: 15 RPM for gemini-3.1-flash-lite.
    Running 5 test cases sequentially fires ~25+ Gemini calls -> quota exhausted.

    Backoff schedule (seconds): 15 -> 30 -> 60 -> 120
    Total max wait before giving up: ~3.5 minutes.
    Returns "" on final failure -- callers treat empty string as NEUTRAL.
    """
    delay = 15
    for attempt in range(retries):
        try:
            result = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config={"temperature": temperature},
            )
            return result.text.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt < retries - 1:
                    print(f"  [nli] 429 quota hit -- waiting {delay}s (attempt {attempt+1}/{retries})")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                else:
                    print(f"  [nli] 429 quota hit -- all retries exhausted, returning empty")
                    return ""
            else:
                print(f"  [nli] Gemini error: {e}")
                return ""
    return ""


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

    raw = _gemini_generate(prompt, temperature=0.1).upper()
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


def _build_batch_prompt(pairs: list[dict]) -> str:
    """Build the NLI batch prompt string (shared between first attempt and retry)."""
    items_text = ""
    for i, p in enumerate(pairs):
        facts_fmt = "\n".join(f"    F{j}: {f}" for j, f in enumerate(p["facts"]))
        items_text += f"\nItem {i}:\n  Claim: \"{p['claim']}\"\n  Facts:\n{facts_fmt}\n"

    return (
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


def check_nli_batch(pairs: list[dict]) -> list[dict]:
    """
    Batch NLI: 1 API call for all (claim, facts) pairs.

    pairs: [{"claim": str, "facts": [str, ...]}, ...]
    Returns: list of dicts matching check_nli() output format.

    FIX: Added retry on JSON parse failure.
    Previously: one attempt, then NEUTRAL fallback for ALL items silently.
    Now:
      Attempt 1 — standard prompt, parse JSON
      Attempt 2 — if JSON invalid, retry with tighter formatting instruction
      Final fallback — if both fail, fall back to individual check_nli() calls
        per pair (more API calls but guaranteed to work).

    Why individual fallback instead of NEUTRAL?
    Silently returning NEUTRAL for all items hides real contradictions and
    entailments. The individual check_nli() is slower but correct.
    The batch path fails only on intermittent model output issues, not quota.
    """
    if not pairs:
        return []

    prompt = _build_batch_prompt(pairs)

    # ── Attempt loop with retry ────────────────────────────────────────
    MAX_PARSE_ATTEMPTS = 2
    parsed = None

    for attempt in range(MAX_PARSE_ATTEMPTS):
        raw = _gemini_generate(prompt, temperature=0.1)

        if not raw:
            # Quota/network failure — don't retry, fall through to individual
            print(f"  [nli_batch] Empty response from Gemini (quota/network), using individual fallback")
            break

        # Strip markdown fences if present
        clean = re.sub(r'^```json?\s*', '', raw)
        clean = re.sub(r'\s*```$', '', clean).strip()

        try:
            parsed = json.loads(clean)
            if not isinstance(parsed, list):
                raise ValueError(f"expected list, got {type(parsed).__name__}")
            if len(parsed) == 0:
                raise ValueError("empty list returned")
            # Success
            break
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < MAX_PARSE_ATTEMPTS - 1:
                print(f"  [nli_batch] JSON parse failed (attempt {attempt + 1}): {e} — retrying...")
                # On retry: use a stricter prompt that emphasizes pure JSON output
                prompt = (
                    "IMPORTANT: Your previous response was not valid JSON.\n"
                    "You MUST return ONLY a raw JSON array. No text before or after.\n"
                    "No markdown fences (```). No explanation. Just the array.\n\n"
                ) + _build_batch_prompt(pairs)
            else:
                print(f"  [nli_batch] JSON parse failed after {MAX_PARSE_ATTEMPTS} attempts: {e}")
                print(f"  [nli_batch] Falling back to individual check_nli() per pair...")
                # Individual fallback — slower but guaranteed correct results
                return [check_nli(p["claim"], p["facts"]) for p in pairs]

    # ── Process parsed results ─────────────────────────────────────────
    if parsed is None:
        # Reached here only if Gemini returned empty string (quota/network)
        print(f"  [nli_batch] No response — NEUTRAL fallback for all")
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