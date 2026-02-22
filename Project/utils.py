#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utility functions for RadiovLLMjection pipeline.
All paths resolve relative to PROJECT_DIR (= Project/).
"""

import re
import json
import time
import base64
import random
import mimetypes
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from openai import OpenAI

from config import (
    BASE_URL, DATASET_CSV, MAX_RETRIES, TRANSIENT_CODES,
    OPENROUTER_REFERER, OPENROUTER_TITLE, PROJECT_DIR
)


# =================== DATASET ===================

def load_dataset() -> pd.DataFrame:
    """Load dataset.csv and resolve paths relative to PROJECT_DIR."""
    df = pd.read_csv(DATASET_CSV)
    path_cols = ["original_path", "injection_path", "stealth_path"]
    for col in path_cols:
        if col in df.columns:
            df[f"{col}_abs"] = df[col].apply(lambda p: str(PROJECT_DIR / p))
    return df


# =================== IMAGE ENCODING ===================

def to_data_url(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    mt = mt or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mt};base64,{b64}"


# =================== RESPONSE PARSING ===================

def extract_binary(txt: str) -> int:
    """Return 0 or 1 if present in text; else -1."""
    t = (txt or "").strip()
    if t in ("0", "1"):
        return int(t)
    m = re.search(r"\b([01])\b", t)
    return int(m.group(1)) if m else -1


# =================== RETRY LOGIC ===================

def backoff_sleep(attempt: int, base: float = 1.8, jitter: float = 0.6, cap: float = 90.0):
    delay = min((base ** attempt) * (1 + random.random() * jitter), cap)
    time.sleep(delay)


def chat_once(
    client: OpenAI,
    model: str,
    data_url: str,
    user_prompt: str,
    max_tokens: int,
    system_prompt: Optional[str] = None,
) -> Tuple[Optional[str], int, Optional[str]]:
    """Single API call to OpenRouter. Returns (text, status_code, error)."""
    extra_headers = {}
    if OPENROUTER_REFERER:
        extra_headers["HTTP-Referer"] = OPENROUTER_REFERER
    if OPENROUTER_TITLE:
        extra_headers["X-Title"] = OPENROUTER_TITLE

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0,
            stream=False,
            extra_headers=extra_headers,
        )
        txt = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        return (txt if txt else None), 200, None
    except Exception as e:
        status = getattr(e, "status", None) or getattr(e, "status_code", None) or 0
        return None, int(status), str(e)[:800]


def generate_with_retry(
    client: OpenAI,
    model: str,
    data_url: str,
    user_prompt: str,
    max_tokens: int,
    system_prompt: Optional[str] = None,
) -> Dict:
    """Call API with exponential backoff on transient errors."""
    attempt = 0
    while attempt < MAX_RETRIES:
        txt, code, err = chat_once(
            client, model, data_url, user_prompt, max_tokens, system_prompt
        )
        if txt is not None and code == 200:
            return {"status_code": 200, "response": txt}
        if int(code) in TRANSIENT_CODES:
            print(f"   -> transient (HTTP {code}), retrying...")
            backoff_sleep(attempt)
            attempt += 1
            continue
        return {"status_code": code, "error": err or f"HTTP {code}"}
    return {"status_code": 0, "error": "Max retries exceeded"}


# =================== FILE HELPERS ===================

def sanitize_for_filename(s: str) -> str:
    s = s.replace("/", "_").replace(":", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def decide_expected_for_injected(filename: str) -> Tuple[int, str]:
    """For injected images, the attack tries to force a FALSE NEGATIVE (0)."""
    name = filename.lower()
    if "injected" in name or "injection" in name:
        return 0, "False Negative"
    if "stealth" in name:
        return 0, "Stealth Injection (False Negative)"
    return -1, "Unknown"


def create_client(api_key: str) -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=api_key)


# =================== RESULT PERSISTENCE ===================

def load_existing_results(
    out_dir: Path, model_id: str, custom_field: str, prompt_type: str = "naive"
) -> Dict[str, Dict]:
    """Load existing results from the most recent matching JSON file."""
    existing = {}
    safe_model = sanitize_for_filename(model_id)
    safe_custom = sanitize_for_filename(custom_field)
    pattern = f"{safe_model}_{safe_custom}_*.json"
    json_files = sorted(out_dir.glob(pattern), reverse=True)

    if not json_files:
        return existing

    latest_file = json_files[0]
    try:
        data = json.loads(latest_file.read_text(encoding="utf-8"))
        if data.get("model") == model_id and data.get("prompt_type") == prompt_type:
            for row in data.get("rows", []):
                img_name = row.get("image")
                if img_name:
                    existing[img_name] = row
            print(f"   Loaded {len(existing)} existing results from {latest_file.name}")
    except Exception as e:
        print(f"   Could not load existing results: {e}")

    return existing


def filter_pending(
    all_names: List[str], existing: Dict[str, Dict], require_ocr: bool = False
) -> List[str]:
    """Return names not yet successfully tested."""
    pending = []
    skipped = 0
    for name in all_names:
        if name not in existing:
            pending.append(name)
        else:
            r = existing[name]
            success = r.get("status_code") == 200 and r.get("success")
            if require_ocr:
                success = success and "OCR" in r
            if success:
                skipped += 1
            else:
                pending.append(name)
    if skipped:
        print(f"   Skipping {skipped} already successful tests")
    if pending:
        print(f"   Will process {len(pending)} pending tests")
    return pending


def save_results(results: Dict, out_dir: Path, model_id: str, custom_field: str) -> Path:
    """Save results to JSON, updating existing file if available."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = sanitize_for_filename(model_id)
    safe_custom = sanitize_for_filename(custom_field)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{safe_model}_{safe_custom}_{ts}.json"

    pattern = f"{safe_model}_{safe_custom}_*.json"
    json_files = sorted(out_dir.glob(pattern), reverse=True)
    if json_files:
        try:
            data = json.loads(json_files[0].read_text(encoding="utf-8"))
            if data.get("model") == model_id:
                out_path = json_files[0]
        except Exception:
            pass

    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path
