import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def evaluate_binary_classification(
    df: pd.DataFrame,
    ground_truth_col: str = "img_lbl",
    threshold: float = 0.5,
    include_logloss: bool = True,
    enforce_order: bool = True,
) -> pd.DataFrame:
    """
    Binary classification değerlendirmesi yapar.
    - Ground truth: 'img_lbl' sütunundan alınır (0 veya 1)
    - -1 değerleri missing olarak işlenir ve drop edilir
    - ROC-AUC ve PR-AUC hesaplanır (her iki sınıf mevcut)
    - Missing oranı raporlanır
    
    Args:
        df: DataFrame with 'image', ground_truth_col, and prediction columns
        ground_truth_col: Ground truth sütun adı (default: 'img_lbl')
        threshold: Olasılık değerleri için threshold (default: 0.5)
        include_logloss: Log loss hesaplansın mı (default: True)
        enforce_order: Sütun sırasını düzenle (default: True)
    
    Returns:
        DataFrame with evaluation metrics for each prediction column
    """
    ignore = {"image", ground_truth_col}
    pred_cols = [c for c in df.columns if c not in ignore]
    if not pred_cols:
        raise ValueError("Değerlendirilecek en az bir sütun gerekli.")
    
    # Ground truth'u al
    if ground_truth_col not in df.columns:
        raise ValueError(f"Ground truth sütunu '{ground_truth_col}' bulunamadı.")
    
    y_true_raw = pd.to_numeric(df[ground_truth_col], errors="coerce").values
    y_true_mask = ~np.isnan(y_true_raw)
    
    rows = []
    for col in pred_cols:
        # Tahmin değerlerini al
        y_pred_raw = pd.to_numeric(df[col], errors="coerce").values
        
        # -1 değerlerini missing olarak işle (NaN'a çevir)
        y_pred_raw = np.where(y_pred_raw == -1, np.nan, y_pred_raw)
        
        # Hem ground truth hem de prediction için geçerli olan satırları bul
        valid_mask = y_true_mask & ~np.isnan(y_pred_raw)
        y_true = y_true_raw[valid_mask].astype(int)
        y_pred_clean = y_pred_raw[valid_mask]
        
        n_total = len(df)
        n = len(y_pred_clean)
        n_missing = n_total - n
        missing_rate = n_missing / n_total if n_total > 0 else 0.0
        
        if n == 0:
            rows.append({
                "model": col,
                "type": "nan",
                "n": 0,
                "n_total": n_total,
                "n_missing": n_missing,
                "missing_rate": missing_rate,
                "mean_score": np.nan,
                "median_score": np.nan,
                "frac_pred_pos": np.nan,
                "recall": np.nan,
                "accuracy": np.nan,
                "fnr": np.nan,
                "precision": np.nan,
                "ppv": np.nan,
                "f1": np.nan,
                "specificity": np.nan,
                "npv": np.nan,
                "brier_score": np.nan,
                "log_loss": np.nan if include_logloss else np.nan,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "mcc": np.nan,
                "threshold_used": np.nan
            })
            continue
        
        # Olasılık mı, sert etiket mi?
        unique_vals = pd.unique(pd.Series(y_pred_clean))
        looks_prob = (y_pred_clean.min() >= 0.0) and (y_pred_clean.max() <= 1.0) and (len(unique_vals) > 2 or not set(unique_vals).issubset({0, 1}))
        
        if looks_prob:
            y_score = y_pred_clean.astype(float)
            y_hat = (y_score >= threshold).astype(int)
            typ = "prob"
        else:
            y_hat = y_pred_clean.astype(int)
            y_score = y_hat.astype(float)
            typ = "label"
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        
        # Temel metrikler
        accuracy = accuracy_score(y_true, y_hat)
        precision = precision_score(y_true, y_hat, zero_division=0.0)
        recall = recall_score(y_true, y_hat, zero_division=0.0)
        f1 = f1_score(y_true, y_hat, zero_division=0.0)
        mcc = matthews_corrcoef(y_true, y_hat)
        
        # Spesifik metrikler
        fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        ppv = precision
        
        # Brier score ve log loss
        brier = float(np.mean((y_true.astype(float) - y_score)**2))
        if include_logloss:
            # Log loss için olasılık değerleri gerekli
            if looks_prob:
                y_probs = np.clip(y_score, 1e-12, 1 - 1e-12)
                logloss = float(-np.mean(y_true * np.log(y_probs) + (1 - y_true) * np.log(1 - y_probs)))
            else:
                logloss = np.nan
        else:
            logloss = np.nan
        
        # ROC-AUC ve PR-AUC
        # Her iki sınıf da mevcut olmalı
        if len(np.unique(y_true)) == 2:
            try:
                roc_auc = roc_auc_score(y_true, y_score)
            except ValueError:
                roc_auc = np.nan
            
            try:
                pr_auc = average_precision_score(y_true, y_score)
            except ValueError:
                pr_auc = np.nan
        else:
            roc_auc = np.nan
            pr_auc = np.nan
        
        rows.append({
            "model": col,
            "type": typ,
            "n": n,
            "n_total": n_total,
            "n_missing": n_missing,
            "missing_rate": float(missing_rate),
            "mean_score": float(np.mean(y_score)),
            "median_score": float(np.median(y_score)),
            "frac_pred_pos": float((y_hat == 1).mean()),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "fnr": float(fnr),
            "precision": float(precision),
            "ppv": float(ppv),
            "f1": float(f1),
            "specificity": float(specificity),
            "npv": float(npv),
            "brier_score": float(brier),
            "log_loss": float(logloss),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "mcc": float(mcc),
            "threshold_used": threshold if typ == "prob" else np.nan
        })
    
    out = pd.DataFrame(rows)
    
    if enforce_order:
        desired = ["GT", "injected", "stealth_injected", "injected_immune", "stealth_injected_immune"]
        aliases = {"injection_immune": "injected_immune",
                   "stealth_injection_immune": "stealth_injected_immune"}
        out = out.assign(_orig_idx=np.arange(len(out)))
        canonical = out["model"].map(lambda x: aliases.get(x, x))
        rank_map = {name: i for i, name in enumerate(desired)}
        order_rank = canonical.map(rank_map)
        fallback_rank = len(desired) + out["_orig_idx"]
        out["order_rank"] = order_rank.fillna(fallback_rank)
        out = (out
               .sort_values(by=["order_rank", "_orig_idx"])
               .drop(columns=["order_rank", "_orig_idx"])
               .reset_index(drop=True))
    
    return out

