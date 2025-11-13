#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenRouter VLM Binary Classifier (single model, in-script config, JSON-only)
- OpenAI Python SDK -> OpenRouter /v1/chat/completions
- Text first, then image_url (base64 data URL)
- Exponential backoff (no fallback)
- Saves JSON to:
  /home/ubuntu/RadiovLLMjection/Results/OpenRouter/<model>_{CUSTOM_FIELD}_<timestamp>.json
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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-or-REPLACE_ME")
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.strip() == "OPENROUTER_API_KEY":
                    OPENROUTER_API_KEY = value.strip().strip('"').strip("'")
                    break
BASE_URL   = "https://openrouter.ai/api/v1"

# Multiple models to test (iterative)
MODEL_LIST = [
    "google/gemini-2.5-flash",
    "microsoft/phi-4-multimodal-instruct",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "qwen/qwen3-vl-8b-thinking",
    "anthropic/claude-sonnet-4.5"
]

# I/O
IMAGES_DIR = "/home/ubuntu/RadiovLLMjection/Images/ProjectImg/raw_normals"
OUT_DIR    = "/home/ubuntu/RadiovLLMjection/Results/OpenRouter/normal"
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

# Optional attribution (recommended by OpenRouter; can be empty)
OPENROUTER_REFERER = ""  # e.g., "https://your-app.example"
OPENROUTER_TITLE   = "RadiovLLMjection"

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
    # Optional attribution headers
    extra_headers = {}
    if OPENROUTER_REFERER:
        extra_headers["HTTP-Referer"] = OPENROUTER_REFERER
    if OPENROUTER_TITLE:
        extra_headers["X-Title"] = OPENROUTER_TITLE

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0,
            stream=False,
            extra_headers=extra_headers
        )
        txt = (resp.choices[0].message.content or "").strip() if resp.choices else ""
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
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.startswith("sk-or-REPLACE"):
        env_path = Path(__file__).parent / ".env"
        raise RuntimeError(
            f"Set OPENROUTER_API_KEY in the .env file at {env_path} "
            f"or set the OPENROUTER_API_KEY environment variable."
        )

    img_dir = Path(IMAGES_DIR)
    if not img_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif")])
    if SMOKE_TEST and len(files) > 10:
        files = random.sample(files, 10)

    client = OpenAI(base_url=BASE_URL, api_key=OPENROUTER_API_KEY)

    if not MODEL_LIST:
        raise ValueError("MODEL_LIST is empty. Add at least one model to run inference.")

    print("🤖 OPENROUTER VLM CLASSIFIER (multi-model)")
    print("=" * 60)
    print(f"📂 images   : {img_dir}")
    print(f"💾 out_dir  : {out_dir}")
    print(f"🏷️ custom   : {CUSTOM_FIELD}")
    pause_status = f"ON ({PER_CALL_PAUSE}s)" if ENABLE_PAUSE else "OFF"
    print(f"⏸️  pause    : {pause_status}")
    print(f"🚬 smoke    : {'ON' if SMOKE_TEST else 'OFF'}")

    for model in MODEL_LIST:
        results = {
            "date": datetime.now().isoformat(),
            "base_url": BASE_URL,
            "model": model,
            "images_path": str(img_dir),
            "prompt": PROMPT,
            "count": len(files),
            "rows": []
        }

        print("\n" + "-" * 60)
        print(f"🧠 model    : {model}")

        for p in files:
            print(f"\n🧪 {p.name}")
            data_url = to_data_url(str(p))
            meta = generate_with_retry(client, model, data_url, MAX_TOKENS)
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

        safe_model = sanitize_for_filename(model)
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
