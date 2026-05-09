#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local HuggingFace inference for MedGemma-4B-IT.

Mirrors the return contract of Project/utils.py:generate_with_retry so the
runner can drop it in wherever the OpenRouter call would have gone.
"""

import os
import re
import io
import base64
from pathlib import Path
from typing import Optional, Dict, Tuple

from PIL import Image

_MODEL_CACHE: Dict[str, Tuple[object, object]] = {}
_DATA_URL_RE = re.compile(r"^data:[^;]+;base64,(.+)$", re.S)


def _load_hf_token() -> Optional[str]:
    """Read HF_TOKEN from env, Project/.env, or repo-root /.env (in that order)."""
    tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or ""
    here = Path(__file__).resolve().parent
    candidate_envs = [here.parent / ".env", here.parent.parent / ".env"]
    for env_path in candidate_envs:
        if tok:
            break
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.startswith("HF_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
    return tok or None


def _select_dtype():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _load(model_id: str):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    token = _load_hf_token()
    dtype = _select_dtype()
    print(f"[local] Loading {model_id} (dtype={dtype}, device_map=auto) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", token=token,
    )
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model.eval()
    _MODEL_CACHE[model_id] = (model, processor)
    print(f"[local] Loaded. Device: {model.device}")
    return model, processor


def _data_url_to_pil(data_url: str) -> Image.Image:
    m = _DATA_URL_RE.match(data_url)
    raw = base64.b64decode(m.group(1) if m else data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def local_generate(
    model_id: str,
    data_url: str,
    user_prompt: str,
    max_tokens: int,
    system_prompt: Optional[str] = None,
) -> Dict:
    """Single multimodal inference.

    Returns:
        {"status_code": 200, "response": str} on success
        {"status_code": 0, "error": str} on failure
    """
    try:
        import torch
        model, processor = _load(model_id)
        img = _data_url_to_pil(data_url)

        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": img},
            ],
        })

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=_select_dtype())

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        generated = out[0][input_len:]
        txt = processor.decode(generated, skip_special_tokens=True).strip()
        return {"status_code": 200, "response": txt}
    except Exception as e:
        return {
            "status_code": 0,
            "error": f"local_inference: {type(e).__name__}: {str(e)[:600]}",
        }
