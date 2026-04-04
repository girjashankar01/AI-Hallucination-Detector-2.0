# core/domain_classifier.py

import os
import time
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ── HuggingFace Setup ──────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# BART-MNLI: purpose-built zero-shot classification model
# No training data needed for our labels — it infers via NLI
BART_MNLI_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

# Valid domain labels — these become the NLI hypothesis classes
DOMAINS = ["medical", "legal", "financial", "general"]

# ── Gemini fallback client ─────────────────────────────────────────────
# Only used if BART-MNLI is unavailable (loading / rate limited)
Client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ── Core Functions ─────────────────────────────────────────────────────

def _classify_with_bart(text: str) -> str | None:
    """
    Calls HuggingFace BART-MNLI for zero-shot classification.
    
    Returns the top label as a string, or None on failure.
    
    Why None and not a fallback label?
    The caller decides what to do on failure — maybe it retries,
    maybe it calls Gemini. Returning a fake label silently would
    hide errors in production.
    """
    payload = {
        "inputs": text[:512],   # BART max input is 1024 tokens; 512 chars is safe
        "parameters": {
            "candidate_labels": DOMAINS
        }
    }
    
    try:
        response = requests.post(
            BART_MNLI_URL,
            headers=HF_HEADERS,
            json=payload,
            timeout=30
        )
        
        # 503 = model is cold-starting (not loaded in HF's inference server)
        # estimated_time tells you how long to wait
        if response.status_code == 503:
            data = response.json()
            wait_time = data.get("estimated_time", 20)
            print(f"[domain] BART-MNLI loading, waiting {wait_time:.0f}s...")
            time.sleep(min(wait_time, 25))   # cap at 25s — don't hang forever
            
            # Retry once after wait
            response = requests.post(
                BART_MNLI_URL,
                headers=HF_HEADERS,
                json=payload,
                timeout=30
            )
        
        if response.status_code != 200:
            print(f"[domain] BART-MNLI returned {response.status_code}: {response.text[:100]}")
            return None
        
        data = response.json()

        if isinstance(data, list) and len(data) > 0 and "label" in data[0]:
            top_label = data[0]["label"]
            top_score = round(data[0]["score"], 3)
            all_scores = {item["label"]: round(item["score"], 3) for item in data}

            second_score = round(data[1]["score"], 3) if len(data) > 1 else 0
            if top_score - second_score < 0.05:
                print(f"[domain] BART uncertain (gap: {round(top_score - second_score, 3)}), defaulting to 'general'")
                return "general"

        else:
            if isinstance(data, list):
                data = data[0]
            top_label = data["labels"][0]
            top_score = round(data["scores"][0], 3)
            all_scores = {l: round(s, 3) for l, s in zip(data["labels"], data["scores"])}

        print(f"[domain] BART-MNLI classified as '{top_label}' (confidence: {top_score})")
        print(f"[domain] All scores: {all_scores}")
        return top_label

    

    except requests.Timeout:
        print("[domain] BART-MNLI timed out")
        return None
    except Exception as e:
        print(f"[domain] BART-MNLI error: {e}")
        return None


def _classify_with_gemini(text: str) -> str:
    """
    Gemini fallback for domain classification.
    Less ideal than BART-MNLI (no confidence scores, uses generation credits)
    but always available.
    """
    response = Client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=(
            f"Classify this text into exactly one category.\n"
            f"Categories: medical, legal, financial, general\n"
            f"Reply with one word only — the category name, nothing else.\n\n"
            f"Text: '{text[:500]}'"
        ),
        config={"temperature": 0.0}   # deterministic — classification should not be random
    )
    
    result = response.text.strip().lower()
    
    # Sanitize: Gemini might return "Medical" or "This is medical" despite instructions
    for domain in DOMAINS:
        if domain in result:
            print(f"[domain] Gemini fallback classified as '{domain}'")
            return domain
    
    print(f"[domain] Gemini returned unexpected: '{result}', defaulting to 'general'")
    return "general"


def classify_domain(text: str) -> str:
    """
    Main entry point. Classifies input text into one of four domains.
    
    Primary: BART-MNLI via HuggingFace Inference API (zero-shot NLI)
    Fallback: Gemini prompt (if BART unavailable)
    
    Returns: "medical" | "legal" | "financial" | "general"
    
    Why this two-model approach?
    BART-MNLI is more principled (calibrated probabilities, not string generation)
    but has cold-start latency. Gemini is always warm. Two-level fallback ensures
    the classifier never blocks the pipeline — it degrades gracefully.
    """
    # Primary: BART-MNLI
    result = _classify_with_bart(text)
    
    if result and result in DOMAINS:
        return result
    
    # Fallback: Gemini
    print("[domain] Falling back to Gemini for classification")
    return _classify_with_gemini(text)


# ── Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    test_cases = [
        # Medical
        ("The patient was administered 500mg of amoxicillin for bacterial pneumonia",   "medical"),
        ("Myocardial infarction risk increases significantly with elevated LDL levels",  "medical"),
        
        # Legal
        ("The defendant was acquitted due to insufficient evidence under the burden of proof", "legal"),
        ("The Supreme Court ruling established new precedent for fourth amendment searches",    "legal"),
        
        # Financial
        ("The Federal Reserve raised interest rates by 25 basis points in Q3",        "financial"),
        ("The company reported a 34% increase in EBITDA for fiscal year 2024",        "financial"),
        
        # General
        ("Einstein won the Nobel Prize for the photoelectric effect in 1921",         "general"),
        ("The Great Wall of China was built during the Ming Dynasty",                  "general"),
    ]
    
    print("═" * 65)
    print("Domain Classifier Test — BART-MNLI + Gemini fallback")
    print("═" * 65)
    
    correct = 0
    for text, expected in test_cases:
        predicted = classify_domain(text)
        match = "✓" if predicted == expected else "✗"
        status = "CORRECT" if predicted == expected else f"WRONG (expected {expected})"
        print(f"\n{match} '{text[:60]}...'")
        print(f"  Predicted: {predicted}  |  {status}")
        if predicted == expected:
            correct += 1
    
    print(f"\n{'═' * 65}")
    print(f"Accuracy: {correct}/{len(test_cases)} = {100*correct//len(test_cases)}%")
    print(f"{'═' * 65}")