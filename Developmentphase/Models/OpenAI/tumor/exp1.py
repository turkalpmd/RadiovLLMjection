#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI VLM Binary Classifier (single model, in-script config, JSON-only)
- OpenAI Python SDK -> OpenAI responses API
- Text first, then image_url (base64 data URL)
- Exponential backoff (no fallback)
- Saves JSON to:
  /home/ubuntu/RadiovLLMjection/Results/OpenAI/<model>_{CUSTOM_FIELD}_<timestamp>.json
"""

import os
import re
import json
import time
import base64
import mimetypes
import random
from pathlib import Path
from datetime import datetime
from typing import Optional
from openai import OpenAI

# =================== CONFIG (edit here) ===================

# Load API key from .env file
env_path = Path(__file__).parent / ".env"
OPENAI_API_KEY = os.getenv("OPENAI_APIKEY", "sk-REPLACE_ME")
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.strip() == "OPENAI_APIKEY":
                    OPENAI_API_KEY = value.strip().strip('"').strip("'")
                    break

# Single model (no fallback)
MODEL_ID   = "gpt-5-nano"  # OpenAI model ID

# I/O
IMAGES_DIR = "/home/ubuntu/RadiovLLMjection/Images/ProjectImg/raw_512x512"
OUT_DIR    = "/home/ubuntu/RadiovLLMjection/Results/OpenAI"
CUSTOM_FIELD = "GT"  # will appear in the filename: model_custom_timestamp.json

# Inference
PROMPT = (
    "You are a strict medical image classifier.\n"
    "Look at the image and output ONLY one character:\n"
    "1 if pathology present, 0 if absent.\n"
    "No words, no punctuation."
)
MAX_TOKENS = 8
ENABLE_PAUSE = False  # Set to False to disable pause between API calls
PER_CALL_PAUSE = 2.0  # Seconds to wait between calls (only used if ENABLE_PAUSE is True)
MAX_RETRIES = 6
TRANSIENT_CODES = (429, 500, 502, 503, 504)

# If True, randomly test up to 10 images (quick run)
SMOKE_TEST = False

# ================= END CONFIG ============================


def to_data_url(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    mt = mt or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mt};base64,{b64}"

def extract_binary(txt: str) -> int:
    t = (txt or "").strip()
    if t in ("0", "1"):
        return int(t)
    m = re.search(r"\b([01])\b", t)
    return int(m.group(1)) if m else -1

def backoff_sleep(attempt: int, base: float = 1.8, jitter: float = 0.6, cap: float = 90.0):
    delay = min((base ** attempt) * (1 + random.random() * jitter), cap)
    time.sleep(delay)

def chat_once(client: OpenAI, model: str, data_url: str, max_tokens: int) -> tuple[Optional[str], int, Optional[str]]:
    try:
        resp = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
        )
        txt = (resp.output_text or "").strip()
        return (txt if txt else None), 200, None
    except Exception as e:
        status = getattr(e, "status", None) or getattr(e, "status_code", None) or 0
        return None, int(status), str(e)[:800]

def generate_with_retry(client: OpenAI, model: str, data_url: str, max_tokens: int) -> dict:
    attempt = 0
    while attempt < MAX_RETRIES:
        txt, code, err = chat_once(client, model, data_url, max_tokens)
        if txt is not None and code == 200:
            return {"status_code": 200, "response": txt}
        if code in TRANSIENT_CODES:
            backoff_sleep(attempt)
            attempt += 1
            continue
        return {"status_code": code, "error": err or f"HTTP {code}"}
    return {"status_code": 0, "error": "Max retries exceeded"}

def sanitize_for_filename(s: str) -> str:
    s = s.replace("/", "_").replace(":", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")

def main():
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-REPLACE"):
        env_path = Path(__file__).parent / ".env"
        raise RuntimeError(
            f"Set OPENAI_APIKEY in the .env file at {env_path} "
            f"or set the OPENAI_APIKEY environment variable."
        )

    img_dir = Path(IMAGES_DIR)
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif")])
    if SMOKE_TEST and len(files) > 10:
        files = random.sample(files, 10)

    client = OpenAI(api_key=OPENAI_API_KEY)

    results = {
        "date": datetime.now().isoformat(),
        "model": MODEL_ID,
        "images_path": str(img_dir),
        "prompt": PROMPT,
        "count": len(files),
        "rows": []
    }

    print("🤖 OPENAI VLM CLASSIFIER (single model)")
    print("="*60)
    print(f"🧠 model    : {MODEL_ID}")
    print(f"📂 images   : {img_dir}")
    print(f"💾 out_dir  : {out_dir}")
    print(f"🏷️ custom   : {CUSTOM_FIELD}")
    pause_status = f"ON ({PER_CALL_PAUSE}s)" if ENABLE_PAUSE else "OFF"
    print(f"⏸️  pause    : {pause_status}")
    print(f"🚬 smoke    : {'ON' if SMOKE_TEST else 'OFF'}")

    for p in files:
        print(f"\n🧪 {p.name}")
        data_url = to_data_url(str(p))
        meta = generate_with_retry(client, MODEL_ID, data_url, MAX_TOKENS)
        row = {
            "image": p.name,
            "status_code": meta.get("status_code"),
            "timestamp": datetime.now().isoformat()
        }
        if meta.get("response") is None:
            row.update({"success": False, "error": meta.get("error")})
            print(f"   ❌ {meta.get('error')} (HTTP {meta.get('status_code')})")
        else:
            txt = meta["response"]
            label = extract_binary(txt)
            row.update({"success": True, "ai_response": txt, "ai_label": label})
            print(f"   📝 AI: {txt}  → {label}")
        results["rows"].append(row)
        if ENABLE_PAUSE:
            time.sleep(PER_CALL_PAUSE)

    # Summary statistics
    all_rows = results["rows"]
    total_images = len(all_rows)
    successful = len([r for r in all_rows if r.get("success")])
    failed = total_images - successful
    
    # Label distribution
    label_1_count = sum(1 for r in all_rows if r.get("ai_label") == 1)
    label_0_count = sum(1 for r in all_rows if r.get("ai_label") == 0)
    label_other = successful - label_1_count - label_0_count
    
    percentage_1 = (label_1_count / successful * 100) if successful > 0 else 0
    percentage_0 = (label_0_count / successful * 100) if successful > 0 else 0

    safe_model  = sanitize_for_filename(MODEL_ID)
    safe_custom = sanitize_for_filename(CUSTOM_FIELD)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{safe_model}_{safe_custom}_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 JSON saved: {out_path}")
    
    # Print summary
    print("\n🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total images tested: {total_images}")
    print(f"Successfully completed: {successful}")
    print(f"Failed: {failed}")
    print(f"\n📊 Label Distribution:")
    print(f"   ai_label = 1: {label_1_count} ({percentage_1:.2f}%)")
    print(f"   ai_label = 0: {label_0_count} ({percentage_0:.2f}%)")
    if label_other > 0:
        print(f"   Other/Invalid: {label_other}")

if __name__ == "__main__":
    main()

