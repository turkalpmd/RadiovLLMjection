# RadiovLLMjection

Visual prompt injection attacks on medical image analysis with Vision-Language Models (VLMs).

## Overview

This repository contains code and dataset for testing prompt injection vulnerabilities in VLMs when analyzing medical images (brain MRI scans). We test both standard and "stealth" prompt injection techniques across multiple state-of-the-art models.

## Dataset

- **600 brain MRI images** (4 categories: glioma, meningioma, pituitary, normal)
- Each image has 3 versions:
  - Original (clean)
  - Prompt injection version
  - Stealth injection version
- Ground truth labels (0 = normal, 1 = tumor present)

## Test Types

For each model, we run 5 test scenarios per image (3000 tests total):
1. **Ground Truth**: Original image, standard prompt
2. **Injection**: Injected image, standard prompt
3. **Injection + Immune**: Injected image, immune prompt (defense)
4. **Stealth**: Stealth injected image, standard prompt
5. **Stealth + Immune**: Stealth injected image, immune prompt (defense)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENROUTER_API_KEY=your_openrouter_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
GOOGLE_CLOUD_PROJECT_ID=your-gcp-project-id
```

### 3. Run Tests

```bash
python unified_radiology_tester.py
```

For quick testing (5 images only):
```python
# Edit unified_radiology_tester.py
SMOKE_TEST = True  # Set to False for full 600 images
```

## Configuration

Edit `MODELS_TO_TEST` in `unified_radiology_tester.py` to test different models:

```python
MODELS_TO_TEST = [
    {"provider": "Google", "model": "gemini-3-pro-preview", "api_key_env": "GOOGLE_API_KEY"},
    {"provider": "OpenRouter", "model": "google/gemini-2.5-flash", "api_key_env": "OPENROUTER_API_KEY"},
    {"provider": "OpenAI", "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
]
```

## Output

Results are saved to `Results/Unified/` as JSON files:
```json
{
  "metadata": {
    "model_name": "gemini-3-pro-preview",
    "total_tests": 3000,
    "successful": 2950
  },
  "results": [
    {
      "base_id": "glioma_001",
      "ground_truth": 1,
      "test_type": "injection",
      "classification": 0,
      "success": true,
      "response": "..."
    }
  ]
}
```

## Dataset Structure

```
dataset.csv columns:
- base_id: Image identifier (e.g., "glioma_001")
- tumor_type: Category (glioma, meningioma, pituitary, normal)
- ground_truth: Binary label (0=normal, 1=tumor)
- original_path: Path to clean image
- injection_path: Path to injected image
- stealth_path: Path to stealth injected image
- prompt_injection: Injection text used
- stealth_prompt: Stealth injection text used
```

## Rate Limits

The script includes automatic rate limiting and retry logic. Adjust in config:

```python
ENABLE_PAUSE = True
PER_CALL_PAUSE = 1.5  # seconds between API calls
MAX_RETRIES = 3
```

## Citation

If you use this code or dataset, please cite:

```
[Paper citation coming soon]
```

## License

[Add license information]

## Contact

[Add contact information]

