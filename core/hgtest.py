# debug_hf.py — minimal standalone test, no dotenv, no abstractions
# Usage:
#   python3 debug_hf.py hf_yourtoken
#   OR set HF_TOKEN in shell: HF_TOKEN=hf_xxx python3 debug_hf.py
#
# Pass the token directly as argv[1] to rule out .env loading issues entirely.

import sys
import os
import json
import requests

# ── Token: try argv first, then env, then .env file manually ──
token = None
if len(sys.argv) > 1:
    token = sys.argv[1].strip()
    print(f"[token] Using token from command line argument")
elif os.environ.get("HF_TOKEN"):
    token = os.environ["HF_TOKEN"].strip()
    print(f"[token] Using token from environment variable HF_TOKEN")
else:
    env_paths = [".env", "../.env", os.path.join(os.path.dirname(__file__), ".env")]
    for path in env_paths:
        if os.path.exists(path):
            print(f"[token] Reading .env manually from: {os.path.abspath(path)}")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            token = parts[1].strip().strip('"').strip("'")
                            print(f"[token] Found HF_TOKEN in .env file")
                            break
            if token:
                break

if not token:
    print("[token] No token found anywhere. Try: python3 debug_hf.py hf_yourtoken")
    sys.exit(1)

masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "too_short"
print(f"[token] Value (masked): {masked}")
print(f"[token] Length: {len(token)} chars")
print(f"[token] Starts with 'hf_': {token.startswith('hf_')}")
print()

# ── Test both endpoints ──
URLS = [
    ("Classic (api-inference)", "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"),
    ("Router (hf-inference)",   "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"),
]

PAYLOAD = {
    "inputs": "Albert Einstein was born in Germany.",
    "parameters": {
        "candidate_labels": ["factually correct", "factually incorrect"],
        "multi_label": False,
    },
}

for label, url in URLS:
    print("=" * 60)
    print(f"ENDPOINT: {label}")
    print(f"URL: {url}")
    print("=" * 60)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    print(f"Request headers sent: {headers}")
    print(f"Payload: {json.dumps(PAYLOAD)}")
    print()

    try:
        resp = requests.post(url, headers=headers, json=PAYLOAD, timeout=40)

        print(f"Status code : {resp.status_code}")
        print(f"Response headers: {dict(resp.headers)}")
        print(f"Raw body:")
        print(resp.text[:1000])
        print()

        if resp.status_code == 200:
            data = resp.json()
            print(f"SUCCESS")
            print(f"  Response type: {type(data).__name__}")
            if isinstance(data, list) and data:
                print(f"  data[0]: {data[0]}")
            elif isinstance(data, dict):
                print(f"  data: {data}")
        elif resp.status_code == 401:
            print("401 UNAUTHORIZED — token invalid or malformed")
        elif resp.status_code == 403:
            print("403 FORBIDDEN — token valid but missing Inference API permissions")
            print("  Go to huggingface.co/settings/tokens → edit token → enable Inference API")
        elif resp.status_code == 422:
            print("422 UNPROCESSABLE — payload rejected, see body above")
        elif resp.status_code == 503:
            print("503 — model loading, wait ~20s and retry")
        elif resp.status_code == 429:
            print("429 RATE LIMITED — check if Authorization header is actually being sent")
        else:
            print(f"Unexpected status: {resp.status_code}")

    except requests.Timeout:
        print("TIMEOUT — no response in 40s")
    except requests.ConnectionError as e:
        print(f"CONNECTION ERROR: {e}")
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")

    print()