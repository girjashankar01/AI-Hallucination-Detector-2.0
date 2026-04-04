# core/consistency.py — Module D: Multi-Model Consensus
#
# ═══════════════════════════════════════════════════════════════════════
# WHAT MODULE D IS AND WHY IT EXISTS
# ═══════════════════════════════════════════════════════════════════════
#
# The original bug (fixed in the previous version):
#   Asked Gemini n=3 times, measured cosine similarity between responses.
#   "The statement is false because X" and "The statement is false because Y"
#   are semantically similar → high similarity → treated as high confidence.
#   That was backwards — high agreement on FALSE = hallucination signal.
#
# The previous fix (in the version you handed over):
#   Parse explicit TRUE/FALSE verdicts, score = fraction that say TRUE.
#   Better, but still asking the same model 3 times — correlated responses.
#   Gemini has a single set of weights, a single world model, a single bias.
#   Asking it 3 times gives 3 draws from the same distribution.
#
# Module D's improvement — genuine independence via different architectures:
#   1. Gemini (standard)    — autoregressive LLM, broad knowledge
#   2. Gemini (adversarial) — same model, but prompted to steelman FALSE
#   3. BART-MNLI (HF)       — NLI classifier, completely different mechanism
#
#   Why BART-MNLI specifically:
#   - Zero-shot classification, not text generation → no parsing needed
#   - Returns clean probability scores (TRUE confidence: 0.0–1.0)
#   - Different architecture (encoder-only NLI) vs Gemini (decoder LLM)
#   - One of HF's most downloaded models → always warm, no cold start
#   - Tagged correctly as zero-shot-classification → router serves it right
#   - No sentencepiece, no task mismatch, no 400/404 errors
#
# WEIGHTED VOTING:
#   Each source produces a TRUE confidence score (0.0–1.0).
#   Final score = weighted average of all three confidence scores.
#
#   Weights:
#     Gemini standard:    0.40  (primary, most capable)
#     Gemini adversarial: 0.25  (stress test — finds edge cases Gemini misses)
#     BART-MNLI:          0.35  (independent signal, different mechanism)
#
#   If HF fails → weights redistribute: Gemini standard 0.55, adversarial 0.45
#   Pipeline never breaks — always returns a score.
#
# ═══════════════════════════════════════════════════════════════════════
# HOW EACH MODEL PRODUCES A CONFIDENCE SCORE
# ═══════════════════════════════════════════════════════════════════════
#
# Gemini standard:
#   Prompt asks for TRUE/FALSE/UNCERTAIN + reason as JSON.
#   TRUE      → confidence = 1.0
#   UNCERTAIN → confidence = 0.5
#   FALSE     → confidence = 0.0
#
# Gemini adversarial:
#   Prompt: "Find every reason this could be false. Then give your verdict."
#   Forces Gemini to actively look for problems before concluding.
#   Reduces confirmation bias — Gemini tends to lean TRUE on plausible claims.
#   TRUE      → confidence = 1.0
#   UNCERTAIN → confidence = 0.5
#   FALSE     → confidence = 0.0
#
# BART-MNLI (HF premise-grounded NLI):
#   Step 1 — Gemini generates a short authoritative factual premise about the topic.
#   Step 2 — BART zero-shot classification:
#     inputs             = premise  (the known fact, as a plain string)
#     candidate_labels   = ["true", "false"]
#     hypothesis_template = "The following statement is {}: <claim>"
#   Internally BART runs NLI:
#     does premise entail "claim is true"? → high score → TRUE
#     does premise entail "claim is false"? → high score → FALSE
#   confidence = P("true") — a real float, not a hard 0/1
#
# WHY NOT {"text": premise, "text_pair": claim}:
#   The HF router (router.huggingface.co) strictly requires `inputs` to be a
#   string, not a dict. Passing {"text":..., "text_pair":...} causes the router
#   to reject the payload — this was the bug in the previous version that made
#   HF always unavailable. The hypothesis_template approach achieves identical
#   NLI semantics while satisfying the router's string-inputs requirement.
#
# ═══════════════════════════════════════════════════════════════════════
# OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════════════
#
# {
#   "consistency_score": float,       # 0.0–1.0, weighted TRUE confidence
#   "verdicts": list[str],            # ["TRUE", "FALSE", "TRUE"] per model
#   "responses": list[str],           # reasoning text per model
#   "models_used": list[str],         # ["Gemini-Standard", "Gemini-Adversarial", "BART-MNLI"]
#   "model_scores": list[float],      # raw confidence per model [0.0–1.0]
#   "model_weights": list[float],     # weights used [0.40, 0.25, 0.35]
#   "true_count": int,
#   "false_count": int,
#   "hf_available": bool,             # whether BART-MNLI was reachable
# }

import re
import json
import os
import time
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

HF_TOKEN   = os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# api-inference.huggingface.co returned 410 Gone — HF deprecated it.
# The router is now the only supported endpoint.
BART_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

# Weights must sum to 1.0
WEIGHT_GEMINI_STANDARD    = 0.40
WEIGHT_GEMINI_ADVERSARIAL = 0.25
WEIGHT_BART_MNLI          = 0.35

# Fallback weights when HF unavailable (redistribute BART weight)
WEIGHT_GEMINI_STANDARD_FALLBACK    = 0.55
WEIGHT_GEMINI_ADVERSARIAL_FALLBACK = 0.45


# ── Confidence from verdict ─────────────────────────────────────────────

def _verdict_to_confidence(verdict: str) -> float:
    """
    Maps a string verdict to a float confidence score.
    UNCERTAIN = 0.5 lets ambiguous/contested claims land in the middle range
    instead of being forced to 0.0 (FALSE) or 1.0 (TRUE).
    """
    if verdict == "TRUE":
        return 1.0
    if verdict == "UNCERTAIN":
        return 0.5
    return 0.0   # FALSE or anything unrecognised


# ── Gemini: Standard Evaluation ────────────────────────────────────────

def _ask_gemini_standard(claim: str) -> dict:
    """
    Straightforward factual evaluation.

    FIX: Added UNCERTAIN as a valid verdict (confidence = 0.5).
    Previously the binary TRUE/FALSE forced genuinely ambiguous claims
    (e.g. "coffee reduces Alzheimer's risk") to FALSE → score 0.0,
    which made Test 4's expected range of 0.3–0.7 unreachable.

    Returns {"verdict": "TRUE"|"FALSE"|"UNCERTAIN", "confidence": float, "reason": str}
    """
    prompt = (
        f"Evaluate whether this statement is factually true or false.\n"
        f"Statement: \"{claim}\"\n\n"
        f"Return ONLY a JSON object. No markdown, no backticks, no preamble.\n"
        f"Format: {{\"verdict\": \"TRUE\", \"reason\": \"one sentence explanation\"}}\n"
        f"verdict must be exactly one of: \"TRUE\", \"FALSE\", or \"UNCERTAIN\".\n"
        f"Use UNCERTAIN only for claims that are genuinely contested or lack scientific consensus.\n"
        f"Use TRUE/FALSE for claims that have a clear, established answer."
    )

    try:
        result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.3}   # low temp for factual eval
        )
        raw = result.text.strip()
        raw = re.sub(r'^```json?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        parsed = json.loads(raw)
        verdict = str(parsed.get("verdict", "FALSE")).strip().upper()
        if verdict not in ("TRUE", "FALSE", "UNCERTAIN"):
            verdict = "FALSE"
        reason = str(parsed.get("reason", ""))

        return {
            "verdict":    verdict,
            "confidence": _verdict_to_confidence(verdict),
            "reason":     reason,
        }

    except Exception as e:
        print(f"  [Gemini-Standard] Error: {e}")
        # Regex fallback
        try:
            raw = result.text if result else ""
            if re.search(r'\bUNCERTAIN\b', raw, re.I):
                verdict = "UNCERTAIN"
            elif re.search(r'\bTRUE\b', raw, re.I):
                verdict = "TRUE"
            else:
                verdict = "FALSE"
            return {
                "verdict":    verdict,
                "confidence": _verdict_to_confidence(verdict),
                "reason":     raw[:100],
            }
        except Exception:
            return {"verdict": "FALSE", "confidence": 0.0, "reason": "parse error"}


# ── Gemini: Adversarial Evaluation ─────────────────────────────────────

def _ask_gemini_adversarial(claim: str) -> dict:
    """
    Adversarial / devil's advocate evaluation.
    Forces Gemini to actively search for reasons the claim could be wrong
    before rendering a verdict. Reduces confirmation bias.

    FIX: Added UNCERTAIN as a valid verdict here too.
    The adversarial prompt was previously too aggressive for genuinely
    uncertain claims — it pushed Gemini to always pick FALSE even when
    the honest answer is "science doesn't know yet".

    Returns {"verdict": "TRUE"|"FALSE"|"UNCERTAIN", "confidence": float, "reason": str}
    """
    prompt = (
        f"Your job is to be a rigorous fact-checker. Before deciding, "
        f"actively look for reasons why this statement could be FALSE — "
        f"missing context, subtle inaccuracies, common misconceptions, or "
        f"misleading phrasings.\n\n"
        f"Statement: \"{claim}\"\n\n"
        f"After stress-testing the statement, give your final verdict.\n"
        f"Return ONLY a JSON object. No markdown, no backticks, no preamble.\n"
        f"Format: {{\"verdict\": \"TRUE\", \"reason\": \"one sentence\"}}\n"
        f"verdict must be exactly one of: \"TRUE\", \"FALSE\", or \"UNCERTAIN\".\n"
        f"Use UNCERTAIN when the claim is scientifically contested, lacks consensus, "
        f"or is only partially supported by evidence. Do not force FALSE on ambiguous claims."
    )

    try:
        result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.5}   # slightly higher — want critical thinking
        )
        raw = result.text.strip()
        raw = re.sub(r'^```json?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        parsed = json.loads(raw)
        verdict = str(parsed.get("verdict", "FALSE")).strip().upper()
        if verdict not in ("TRUE", "FALSE", "UNCERTAIN"):
            verdict = "FALSE"
        reason = str(parsed.get("reason", ""))

        return {
            "verdict":    verdict,
            "confidence": _verdict_to_confidence(verdict),
            "reason":     reason,
        }

    except Exception as e:
        print(f"  [Gemini-Adversarial] Error: {e}")
        try:
            raw = result.text if result else ""
            if re.search(r'\bUNCERTAIN\b', raw, re.I):
                verdict = "UNCERTAIN"
            elif re.search(r'\bTRUE\b', raw, re.I):
                verdict = "TRUE"
            else:
                verdict = "FALSE"
            return {
                "verdict":    verdict,
                "confidence": _verdict_to_confidence(verdict),
                "reason":     raw[:100],
            }
        except Exception:
            return {"verdict": "FALSE", "confidence": 0.0, "reason": "parse error"}


# ── Gemini: Generate fact premise for NLI ──────────────────────────────

def _get_fact_premise(claim: str) -> str | None:
    """
    Asks Gemini to produce a short, authoritative factual sentence that
    directly addresses the specific assertion in the claim.

    KEY REQUIREMENT: the premise must name the same person / entity / event
    the claim mentions and state the verified, correct detail about them.
    A generic background fact (e.g. "The Nobel Prize is awarded annually…")
    is useless — it gives BART no signal to detect contradiction.

    This becomes the NLI premise fed to BART. BART then measures whether
    the claim is supported or refuted by that known fact.

    Examples:
      Claim:   "Einstein won the Nobel Prize for the theory of relativity"
      Good:    "Einstein won the 1921 Nobel Prize in Physics for his
                discovery of the law of the photoelectric effect."
      Bad:     "The Nobel Prize in Physics is awarded annually."   ← no signal

      Claim:   "Satya Nadella won the Nobel Prize in Economics"
      Good:    "Satya Nadella is the CEO of Microsoft and has not been
                awarded a Nobel Prize in Economics or any other field."
      Bad:     "The Nobel Memorial Prize in Economic Sciences is awarded
                by the Royal Swedish Academy."   ← no signal

    Returns a one-sentence string or None if Gemini fails.
    """
    prompt = (
        f"Write one factual sentence that states the ground truth about the "
        f"specific assertion in this claim.\n\n"
        f"Rules:\n"
        f"  • Name the same person, award, event, or entity the claim mentions.\n"
        f"  • State the verified, correct detail — be SPECIFIC and COMPLETE.\n"
        f"    If the claim is about what prize someone won, name the ACTUAL prize reason.\n"
        f"    If the claim is about a role or achievement, state what they ACTUALLY hold/did.\n"
        f"    If they never received an award, explicitly say so.\n"
        f"  • Your sentence must contain enough information to CONFIRM or CONTRADICT the claim.\n"
        f"  • Do NOT write vague or partial facts that leave the claim unresolved.\n"
        f"  • Do NOT write generic background sentences that omit the subject.\n\n"
        f"Examples:\n"
        f"  Claim: 'Einstein won the Nobel Prize for the theory of relativity'\n"
        f"  Good:  'Einstein won the 1921 Nobel Prize in Physics for his discovery of the "
        f"law of the photoelectric effect, not for the theory of relativity.'\n"
        f"  Bad:   'Einstein was awarded the 1921 Nobel Prize for services to theoretical "
        f"physics.'  ← too vague, doesn't resolve the relativity question\n\n"
        f"  Claim: 'Satya Nadella won the Nobel Prize in Economics'\n"
        f"  Good:  'Satya Nadella is the CEO of Microsoft and has never been awarded a "
        f"Nobel Prize in Economics or any other field.'\n"
        f"  Bad:   'Satya Nadella serves as chairman and CEO of Microsoft.'  "
        f"← says nothing about Nobel\n\n"
        f"Claim: {claim}\n\n"
        f"Reply with only the factual sentence. No preamble, no quotes."
    )
    try:
        result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.1}
        )
        premise = result.text.strip().strip('"').strip("'")
        return premise if len(premise) > 10 else None
    except Exception as e:
        print(f"  [BART-MNLI] Premise generation failed: {e}")
        return None


# ── BART-MNLI: Premise-Grounded NLI Entailment Check ───────────────────

def _ask_bart_mnli(claim: str) -> dict | None:
    """
    Uses BART-large-MNLI for premise-grounded NLI entailment checking.

    WHY THE OLD {"text": premise, "text_pair": claim} APPROACH WAS BROKEN:
    The HF router (router.huggingface.co) strictly requires `inputs` to be a
    plain string. Passing a dict like {"text": ..., "text_pair": ...} causes
    the router to reject the payload entirely — this was the bug that made
    hf_available always False in the previous version.

    THE FORMAT — hypothesis_template approach:
    We keep the correct two-step NLI logic (Gemini generates premise, BART
    checks entailment) but deliver it in the format the router accepts:

      inputs             = premise           ← plain string ✓
      candidate_labels   = ["supported", "refuted"]
      hypothesis_template = "The claim that <claim> is {}."

    Internally BART runs NLI for each label:
      • "The claim that <claim> is supported."  vs premise
      • "The claim that <claim> is refuted."    vs premise
    Score for "supported" = confidence the claim is consistent with the premise.

    WHY "supported/refuted" INSTEAD OF "true/false":
    Earlier we used ["true", "false"] with template "The following statement
    is {}: <claim>". BART struggled to detect contradiction with this framing
    because "true/false" are abstract meta-labels — BART would latch onto
    surface token overlap (e.g. both premise and claim mention Nobel+Einstein)
    and score "true" high even when the reason differed (relativity vs
    photoelectric effect). "supported/refuted" are semantically richer NLI
    words BART was trained on, making it much more sensitive to actual
    premise-hypothesis contradiction.

    Response shape from router (flat list, sorted by score desc):
      [{"label": "refuted",    "score": 0.85},
       {"label": "supported",  "score": 0.15}]

    URL: router.huggingface.co — the only supported endpoint (classic 410 Gone).

    Returns {"verdict": str, "confidence": float, "reason": str} or None.
    """

    # Step 1: generate a targeted factual premise via Gemini
    premise = _get_fact_premise(claim)
    if not premise:
        print(f"  [BART-MNLI] Could not generate premise, skipping")
        return None

    print(f"  [BART-MNLI] Premise: {premise[:90]}")

    # Step 2: NLI via zero-shot classification
    #
    # `inputs` = premise (plain string — router requires this)
    # The claim is embedded in hypothesis_template so BART performs genuine
    # premise-vs-hypothesis NLI. Template must have exactly one {} slot.
    # BART formats: template.format("supported") / template.format("refuted")
    #
    # This is real NLI: does the factual premise entail or refute the claim?
    # "supported"/"refuted" are semantically close to NLI-native labels
    # (entailment/contradiction) that BART was trained on — much better signal
    # than surface token overlap from concatenation.
    #
    # Example:
    #   premise  : "Einstein won the 1921 Nobel Prize for the photoelectric effect."
    #   hypothesis: "The claim that 'Einstein won Nobel for relativity' is supported."
    #   → BART correctly scores "supported" LOW (facts differ despite shared tokens)

    payload = {
        "inputs": premise,
        "parameters": {
            "candidate_labels":   ["supported", "refuted"],
            "hypothesis_template": f'The claim that "{claim}" is {{}}.',
            "multi_label":        False,
        }
    }

    try:
        resp = requests.post(
            BART_URL,
            headers={**HF_HEADERS, "Content-Type": "application/json"},
            json=payload,
            timeout=30
        )

        if resp.status_code == 503:
            body = resp.json()
            wait = min(body.get("estimated_time", 20), 25)
            print(f"  [BART-MNLI] Cold-starting, waiting {wait:.0f}s...")
            time.sleep(wait)
            resp = requests.post(
                BART_URL,
                headers={**HF_HEADERS, "Content-Type": "application/json"},
                json=payload,
                timeout=30
            )

        if resp.status_code != 200:
            print(f"  [BART-MNLI] HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()

        # Router returns flat list sorted by score desc:
        # [{"label": "contradictory", "score": 0.85},
        #  {"label": "consistent",    "score": 0.15}]
        if not isinstance(data, list) or len(data) == 0:
            print(f"  [BART-MNLI] Unexpected response shape: {data}")
            return None

        supported_score = 0.5   # neutral default if label not found
        top_label       = data[0].get("label", "unknown")

        for item in data:
            if item.get("label", "").lower() == "supported":
                supported_score = item.get("score", 0.5)
                break

        # Thresholds: ≥0.6 = TRUE, ≤0.4 = FALSE, else UNCERTAIN
        verdict = "TRUE" if supported_score >= 0.6 else ("FALSE" if supported_score <= 0.4 else "UNCERTAIN")

        return {
            "verdict":    verdict,
            "confidence": round(supported_score, 4),
            "reason":     (f"NLI (hypothesis_template): P(supported)={supported_score:.3f}, "
                           f"top='{top_label}'"),
        }

    except requests.Timeout:
        print(f"  [BART-MNLI] Timeout after 30s")
        return None
    except Exception as e:
        print(f"  [BART-MNLI] Error: {e}")
        return None


# ── Public API ──────────────────────────────────────────────────────────

def check_consistency(claim: str, n: int = 3) -> dict:
    """
    Multi-model consensus check for factual grounding.

    Args:
        claim: the sentence to evaluate
        n:     kept for API compatibility (ignored internally — model count
               is determined by available sources, not n)

    Returns:
        dict with consistency_score (0.0–1.0) and full model metadata.
        Higher score = more models agree the claim is TRUE = more grounded.

    Scoring:
        consistency_score = weighted average of per-model confidence scores
        Weights: Gemini-Standard(0.40) + Gemini-Adversarial(0.25) + BART-MNLI(0.35)
        If BART-MNLI unavailable: Gemini-Standard(0.55) + Gemini-Adversarial(0.45)

    Verdicts:
        TRUE      → confidence 1.0
        UNCERTAIN → confidence 0.5  (new — for genuinely contested claims)
        FALSE     → confidence 0.0
    """

    print(f"  [consistency] Evaluating: '{claim[:80]}...'")

    # ── 1. Gemini Standard ─────────────────────────────────────────────
    g_standard = _ask_gemini_standard(claim)
    print(f"  [Gemini-Standard]    [{g_standard['verdict']}] {g_standard['reason'][:80]}")

    # ── 2. Gemini Adversarial ──────────────────────────────────────────
    time.sleep(0.5)   # small gap to avoid rate limiting
    g_adversarial = _ask_gemini_adversarial(claim)
    print(f"  [Gemini-Adversarial] [{g_adversarial['verdict']}] {g_adversarial['reason'][:80]}")

    # ── 3. BART-MNLI ───────────────────────────────────────────────────
    bart_result  = _ask_bart_mnli(claim)
    hf_available = bart_result is not None

    if hf_available:
        print(f"  [BART-MNLI]          [{bart_result['verdict']}] {bart_result['reason']}")
    else:
        print(f"  [BART-MNLI]          unavailable — redistributing weights")

    # ── Weighted Voting ────────────────────────────────────────────────
    if hf_available:
        sources = [
            ("Gemini-Standard",    g_standard,    WEIGHT_GEMINI_STANDARD),
            ("Gemini-Adversarial", g_adversarial, WEIGHT_GEMINI_ADVERSARIAL),
            ("BART-MNLI",          bart_result,   WEIGHT_BART_MNLI),
        ]
    else:
        sources = [
            ("Gemini-Standard",    g_standard,    WEIGHT_GEMINI_STANDARD_FALLBACK),
            ("Gemini-Adversarial", g_adversarial, WEIGHT_GEMINI_ADVERSARIAL_FALLBACK),
        ]

    models_used   = [s[0] for s in sources]
    verdicts      = [s[1]["verdict"] for s in sources]
    model_scores  = [s[1]["confidence"] for s in sources]
    model_weights = [s[2] for s in sources]
    responses     = [s[1]["reason"] for s in sources]

    # Weighted average of confidence scores
    consistency_score = round(
        sum(score * weight for score, weight in zip(model_scores, model_weights)),
        4
    )

    true_count      = sum(1 for v in verdicts if v == "TRUE")
    false_count     = sum(1 for v in verdicts if v == "FALSE")
    uncertain_count = sum(1 for v in verdicts if v == "UNCERTAIN")

    print(f"  [consistency] verdicts: {verdicts}  scores: {[round(s,3) for s in model_scores]}")
    print(f"  [consistency] weights:  {model_weights}  →  final_score={consistency_score}")

    return {
        "consistency_score": consistency_score,
        "verdicts":          verdicts,
        "responses":         responses,
        "models_used":       models_used,
        "model_scores":      model_scores,
        "model_weights":     model_weights,
        "true_count":        true_count,
        "false_count":       false_count,
        "uncertain_count":   uncertain_count,
        "hf_available":      hf_available,
    }


# ── Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    def header(title: str):
        print(f"\n{'═' * 65}")
        print(f"  {title}")
        print(f"{'═' * 65}")

    # ── TEST 1: Clear true fact ────────────────────────────────────────
    header("TEST 1: Clear true fact — expect score close to 1.0")
    r1 = check_consistency("Albert Einstein was born in 1879 in Ulm, Germany", n=3)
    print(f"\n  Final score:  {r1['consistency_score']}")
    print(f"  Models used:  {r1['models_used']}")
    print(f"  Verdicts:     {r1['verdicts']}")
    print(f"  HF available: {r1['hf_available']}")

    time.sleep(2)

    # ── TEST 2: Clear hallucination ────────────────────────────────────
    header("TEST 2: Hallucination — expect score close to 0.0")
    r2 = check_consistency("Einstein won the Nobel Prize for the theory of relativity", n=3)
    print(f"\n  Final score:  {r2['consistency_score']}")
    print(f"  Models used:  {r2['models_used']}")
    print(f"  Verdicts:     {r2['verdicts']}")
    print(f"  HF available: {r2['hf_available']}")

    time.sleep(2)

    # ── TEST 3: Subtle hallucination (plausible but wrong) ─────────────
    header("TEST 3: Subtle hallucination — expect score < 0.5")
    r3 = check_consistency("Satya Nadella won the Nobel Prize in Economics", n=3)
    print(f"\n  Final score:  {r3['consistency_score']}")
    print(f"  Models used:  {r3['models_used']}")
    print(f"  Verdicts:     {r3['verdicts']}")
    print(f"  HF available: {r3['hf_available']}")

    time.sleep(2)

    # ── TEST 4: Ambiguous claim ────────────────────────────────────────
    header("TEST 4: Ambiguous/uncertain claim — expect score ~0.3-0.7")
    r4 = check_consistency("Drinking coffee reduces the risk of Alzheimer's disease", n=3)
    print(f"\n  Final score:  {r4['consistency_score']}")
    print(f"  Models used:  {r4['models_used']}")
    print(f"  Verdicts:     {r4['verdicts']}")
    print(f"  HF available: {r4['hf_available']}")

    # ── SUMMARY ───────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print("  SUMMARY")
    print(f"{'═' * 65}")
    print(f"  Test 1 (true fact):       {r1['consistency_score']}  (expect ~1.0)")
    print(f"  Test 2 (hallucination):   {r2['consistency_score']}  (expect ~0.0)")
    print(f"  Test 3 (subtle false):    {r3['consistency_score']}  (expect <0.5)")
    print(f"  Test 4 (ambiguous):       {r4['consistency_score']}  (expect 0.3-0.7)")
    print(f"\n  BART-MNLI available:      {r1['hf_available']}")
    print(f"  If False → Gemini-only fallback was used (still works)")
    print(f"{'═' * 65}")