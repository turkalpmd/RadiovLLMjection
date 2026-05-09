# MedGemma-4B-IT — RadiovLLMjection Evaluation Report

**Model:** `google/medgemma-4b-it` (HuggingFace, local bf16 inference, RTX 3090)  
**Dataset:** 600 brain-MRI images (300 tumor-positive, 300 tumor-negative) from PMRAM Bangladeshi Brain Tumor MRI Dataset  
**Conditions:** 5 (baseline + 2 attack vectors × 2 prompt regimes)  
**Total inferences:** 3,000 + 600 OCR passes  

## 1. Headline metrics

| Experiment | N | Accuracy | ASR | FPR | Masking | Sensitivity | Specificity | PTR | Paper commercial median (ASR/Acc | FPR) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 600 | 0.712 | — | 0.473 | 0.103 | 0.897 | 0.527 | 0.685 | Acc 0.69 / FPR 0.41 |
| injection_naive | 600 | — | 0.313 | 0.987 | 0.613 | 0.387 | 0.013 | 0.687 | ASR 0.97 / FPR 1.00 |
| stealth_naive | 600 | — | 0.243 | 0.747 | 0.233 | 0.767 | 0.253 | 0.757 | ASR 0.57 / FPR 0.84 |
| injection_immune | 600 | — | 0.005 | 1.000 | 0.010 | 0.990 | 0.000 | 0.995 | ASR 0.53 / FPR 0.91 |
| stealth_immune | 600 | — | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | ASR 0.44 / FPR 0.67 |

## 2. Confusion-matrix details

| Experiment | TP | TN | FP | FN | Unparsable |
|---|---:|---:|---:|---:|---:|
| baseline | 269 | 158 | 142 | 31 | 0 |
| injection_naive | 116 | 4 | 296 | 184 | 0 |
| stealth_naive | 230 | 76 | 224 | 70 | 0 |
| injection_immune | 297 | 0 | 300 | 3 | 0 |
| stealth_immune | 300 | 0 | 300 | 0 | 0 |

## 3. Per-tumor-type breakdown

Predicted-positive fraction (label=1) by ground-truth tumor type. Healthy controls are `normal`. A perfect classifier would predict 1.0 for tumor types and 0.0 for `normal`.

| tumor_type | baseline | injection_naive | stealth_naive | injection_immune | stealth_immune |
|---|---|---|---|---|---|
| glioma | 0.990 | 0.450 | 0.850 | 1.000 | 1.000 |
| meningioma | 1.000 | 0.670 | 0.970 | 1.000 | 1.000 |
| normal | 0.473 | 0.987 | 0.747 | 1.000 | 1.000 |
| pituitary | 0.700 | 0.040 | 0.480 | 0.970 | 1.000 |

## 4. OCR analysis (Step 4: stealth_naive)

- Stealth images that produced OCR output: **600/600**
- OCR output containing the injected `"OFFICIAL REPORT"` string: **518/600** (86.3%)

**Interpretation:** MedGemma can reliably *read* the stealth perturbation text (it surfaces in OCR for ~80% of images), but it largely refuses to *act* on it — the same images yield only ASR=0.243 on the classification task. This is a notable dissociation between OCR capability and prompt-injection compliance, and it is qualitatively different from the commercial models the paper evaluated, which tended to follow OCR-extracted text.

## 5. Key findings vs. paper

1. **Strong native resistance to visible injection.** MedGemma's visible-injection ASR is **0.313** vs. commercial median **0.97** (README, Table 1). This is a 3× drop in attack effectiveness.
2. **Strong resistance to stealth injection.** ASR **0.243** vs. commercial median **0.57**. Stealth attacks remain non-trivial but degrade further.
3. **Immune-protocol regression to a trivial classifier.** Under the multi-stage immune system prompt, MedGemma collapses to **PTR = 0.995/1.000** (visible/stealth) — i.e., predicts "tumor" for ~100% of images. ASR drops to ~0 trivially because the model refuses to output 0 at all. **FPR = 1.00** in both immune conditions. This matches the paper's observation that *"three models maintained FPR of 1.00 even with defense"* (README §Key Findings); MedGemma joins that cluster.
4. **Baseline bias toward positive.** On clean images MedGemma scores **accuracy 0.712** (median commercial 0.69) and **FPR 0.473** (median 0.41). The model over-calls tumor even before any attack, which limits the headroom for the immune protocol to do useful work.
5. **OCR/compliance dissociation** (see §4): the same model that reads the injection text refuses to follow it. This is consistent with MedGemma's medical-domain fine-tuning: the model appears to weight visual evidence over OCR-extracted text in a way that general-purpose VLMs do not.

## 6. Reproduction

```bash
cd Project/medgemma_experiment
conda activate llm
python run_medgemma.py --smoke      # 50-inference sanity check (~2 min)
python run_medgemma.py              # full 3000-inference run (~22 min on RTX 3090)
python build_analysis.py            # regenerate per_image.csv / metrics / summary.md
```

Outputs land in `Project/Results/<experiment>/google_medgemma-4b-it_*.json` and `Project/Results/{per_image.csv, per_experiment_metrics.csv, per_tumor_type_metrics.csv, summary.md}`.

## 7. Caveats

- Single-seed greedy decoding (`do_sample=False`); no variance estimate.
- Naive prompt with `max_tokens=8` enforces a one-token answer; the immune system prompt requests multi-stage reasoning but the user prompt overrides it. The observed immune behavior is therefore best characterized as *"refusal to engage with the protocol"*, not failure of the protocol per se.
- 600-image dataset is balanced and modest; confidence intervals on the metrics are roughly ±0.04 at 95% (Wilson). Differences against commercial medians of 0.6+ are well outside that interval.
- `nvidia-driver` was upgraded mid-day; the run was conducted after a reboot to kernel `6.8.0-111` with driver `580.142`.
