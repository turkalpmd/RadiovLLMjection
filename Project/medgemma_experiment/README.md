# MedGemma Experiment

Standalone runner that adds **`google/medgemma-4b-it`** (Google's gated medical
VLM on HuggingFace) to the RadiovLLMjection benchmark — running locally on a
CUDA GPU through `transformers`.

This folder is fully isolated: it does **not** modify any existing `Project/`
file. It imports helpers from `Project/utils.py` and prompts from
`Project/config.py` read-only, and writes JSON results into the shared
`Project/Results/<experiment>/` tree using the same filename and schema
conventions as the other 5 models, so existing analysis tooling picks up
MedGemma automatically.

## Files

- `local_inference.py` — model load + cached `local_generate(...)` that mirrors
  the return contract of `Project/utils.py:generate_with_retry`.
- `run_medgemma.py` — main runner; loops MedGemma through all 5 experiments
  (baseline, injection_naive, stealth_naive, immune, stealth_immune).

## Prerequisites

1. **GPU** (CUDA with bf16/fp16 support — `torch.cuda.is_available()` must be `True`).
2. **HF account + license**: accept Google's MedGemma license at
   <https://huggingface.co/google/medgemma-4b-it>.
3. **HF_TOKEN** with `read` scope. Append to `Project/.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
4. **Python deps** (already added to repo `requirements.txt`):
   - `torch>=2.1.0`
   - `transformers>=4.50.0`
   - `accelerate>=0.30.0`

## Pre-warm (one-time, ~8 GB download)

```bash
cd [path/to/RadiovLLMjection]/Project
python -c "
import os, torch
from transformers import AutoProcessor, AutoModelForImageTextToText
AutoProcessor.from_pretrained('google/medgemma-4b-it', token=os.environ['HF_TOKEN'])
AutoModelForImageTextToText.from_pretrained(
  'google/medgemma-4b-it', torch_dtype=torch.bfloat16, device_map='auto',
  token=os.environ['HF_TOKEN'])
print('weights cached')
"
```

This catches gated-access 401s before any pipeline run.

## Usage

```bash
cd [path/to/RadiovLLMjection]/Project/medgemma_experiment

# Smoke test — 10 images, baseline only
python run_medgemma.py --smoke --step 2

# Smoke test — 10 images, all 5 experiments
python run_medgemma.py --smoke

# Full run — all 5 experiments, 600 images each (~4 hours on RTX 3090)
python run_medgemma.py

# Specific step
python run_medgemma.py --step 5
python run_medgemma.py --step 3-5
```

The runner is **resume-safe**: if interrupted, re-running picks up where it
left off via `load_existing_results()`.

## Output

JSONs land in `Project/Results/<experiment>/` with the same naming as other
models:

```
Project/Results/baseline/google_medgemma-4b-it_GT_<timestamp>.json
Project/Results/injection_naive/google_medgemma-4b-it_injected_<timestamp>.json
Project/Results/stealth_naive/google_medgemma-4b-it_stealth_injected_<timestamp>.json
Project/Results/injection_immune/google_medgemma-4b-it_injection_immune_<timestamp>.json
Project/Results/stealth_immune/google_medgemma-4b-it_stealth_injection_immune_<timestamp>.json
```

Each JSON has the same schema as the OpenRouter models, except:
- `"sdk": "huggingface-transformers (local)"` (vs `"openai-compatible (OpenRouter)"`)
- Otherwise identical: `model`, `prompt_type`, `rows[*]`, `summary{...}`.

## Notes

- The first inference in each process pays a ~30 s model load (bf16 fp16 fp32 fallback decided at load time).
- Subsequent calls reuse the cached model+processor in-process.
- Generation is `do_sample=False` (greedy) so results are deterministic given the same model weights.
- For non-bf16 GPUs (pre-Ampere), the loader auto-falls back to fp16 then fp32.
