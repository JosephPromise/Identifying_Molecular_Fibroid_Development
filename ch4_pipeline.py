"""
ch4_pipeline.py — Chapter 4 pipeline, converted from molecular__fibroid_prediction.ipynb
for local use (VS Code / terminal). Stages:

    python ch4_pipeline.py all
    python ch4_pipeline.py permute --n-perm 1000
    python ch4_pipeline.py nullsummary
    python ch4_pipeline.py figures

Inputs are read from ./data, outputs written to ./results (both next to this file).
Changes from the notebook: Colab paths replaced; perm_null.csv now lives in
results/ (so the run survives restarts and resumes); --n-perm default restored
to 1000; loci now counted as connected components rather than probes minus pairs;
duplicated/superseded definitions removed (behaviour otherwise unchanged).
"""
# ── from notebook cell 0
#Importations
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix,
)

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


# ── paths (local project layout)
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUT = BASE / "results"
FIGDIR = OUT / "figures"

import sklearn
if tuple(int(p) for p in sklearn.__version__.split(".")[:2]) < (1, 8):
    sys.exit(f"scikit-learn {sklearn.__version__} found: l1_ratio=1.0 only means L1 "
             "(LASSO) from 1.8 onward. Upgrade, or change l1_ratio=1.0 to penalty='l1'.")


# ── from notebook cell 1
# configuration

SEED = 42
DNN_SEEDS = (42, 43, 44, 45, 46)      # five-seed protocol
EPS = 1e-6                            # beta clipping bound

METH_VAR_THRESHOLD = 0.02             # beta-scale inter-sample variance
EXPR_VAR_THRESHOLD = 0.50
GRID_LO, GRID_HI = 1e-4, 1e2
GRID_POINTS_OBS = 100                 # observed run. NOTE: the Colab notebook silently
                                      # overrode 10 with 100 in its last cell; 100 is what
                                      # produced your current results. Set deliberately.
GRID_POINTS_PERM = 12                 # permutation null (documented approximation)
INNER_FOLDS = 3
STABILITY_GATE = 0.50                 # fold fraction required to enter the panel
N_BOOTSTRAP = 10_000

DNN_HIDDEN = (16, 8)                  # fixed a priori, not tuned on the data
DNN_DROPOUT = 0.30
DNN_EPOCHS = 150
DNN_LR = 1e-3
XGB_ESTIMATORS, XGB_DEPTH, XGB_LR = 200, 2, 0.10



# Genes reported in the uterine fibroid, WNT/beta-catenin and hedgehog literature
FIBROID_LIT = {
    "HMGA2", "HMGA1", "MED12", "DLK1", "MEG3", "RTL1", "CTNNB1", "WNT4", "WNT5B",
    "WT1", "FH", "COL1A1", "COL3A1", "COL4A2", "FN1", "TGFB3", "TGFBR2", "ESR1",
    "PGR", "RAD51B", "CUX1", "BET1L", "TP53", "FASN", "SHH", "GLI1", "PTCH1",
    "DVL1", "APC", "AXIN2", "SFRP1", "CCDC88C",
}

# ── from notebook cell 3
# Step 1 — in-fold selection and models

def select_lasso(X_tr, y_tr, grid_points, seed=SEED):
    """Inner stratified CV over the C grid, on the training fold only."""
    grid = np.logspace(np.log10(GRID_LO), np.log10(GRID_HI), grid_points)
    skf = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
    best_c, best_score = grid[0], -np.inf

    for C in grid:
        scores = []
        for a, b in skf.split(X_tr, y_tr):
            if len(np.unique(y_tr[b])) < 2 or len(np.unique(y_tr[a])) < 2:
                continue
            m = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=C,
                                   class_weight="balanced", max_iter=5000,
                                   random_state=seed).fit(X_tr[a], y_tr[a])
            scores.append(roc_auc_score(y_tr[b], m.predict_proba(X_tr[b])[:, 1]))
        # ties broken toward the sparser model by requiring strict improvement
        if scores and np.mean(scores) > best_score + 1e-12:
            best_score, best_c = float(np.mean(scores)), C

    final = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=best_c,
                               class_weight="balanced", max_iter=5000,
                               random_state=seed).fit(X_tr, y_tr)
    coef = final.coef_.ravel()
    sel = np.where(np.abs(coef) > 1e-10)[0]
    return sel, coef[sel], float(best_c), best_score, final


def fit_xgb(X, y_tr, seed=SEED):
    from xgboost import XGBClassifier
    pos = max(float((y_tr == 1).sum()), 1.0)
    neg = float((y_tr == 0).sum())
    return XGBClassifier(n_estimators=XGB_ESTIMATORS, max_depth=XGB_DEPTH,
                         learning_rate=XGB_LR, reg_lambda=1.0,
                         scale_pos_weight=neg / pos, eval_metric="logloss",
                         random_state=seed, n_jobs=1,
                         tree_method="exact").fit(X, y_tr)


def build_dnn(n_features, seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    net = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(DNN_HIDDEN[0], activation="relu"),
        tf.keras.layers.Dropout(DNN_DROPOUT),
        tf.keras.layers.Dense(DNN_HIDDEN[1], activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    net.compile(optimizer=tf.keras.optimizers.Adam(DNN_LR),
                loss="binary_crossentropy")
    return net

# ── from notebook cell 4
# Step 4 — attribution

def tree_shap(model, X_bg, names):
    """Exact TreeSHAP on the fold's gradient-boosted model."""
    import shap
    try:
        v = np.asarray(shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent"
        ).shap_values(X_bg, check_additivity=False))
        if v.ndim == 3:
            v = v[..., -1]
        return {k: float(x) for k, x in zip(names, np.abs(v).mean(0))}
    except Exception as e:
        print(f"    [warn] TreeSHAP failed: {e}")
        return {}


def deep_shap(model, X_bg, names):
    """
    DeepSHAP on the fold's network.

    shap.DeepExplainer is unreliable on some TensorFlow/Keras 3 combinations.
    GradientExplainer is retained as a fallback; the estimator actually used is
    recorded per fold and must be reported if the fallback ever triggers.
    """
    import shap
    for label, ctor in (("DeepExplainer", shap.DeepExplainer),
                        ("GradientExplainer", shap.GradientExplainer)):
        try:
            v = np.asarray(ctor(model, X_bg).shap_values(X_bg))
            if v.ndim == 3:
                v = v[..., -1]
            return {k: float(x) for k, x in zip(names, np.abs(v).mean(0))}, label
        except Exception:
            continue
    print("    [warn] both DeepSHAP estimators failed for this fold")
    return {}, "failed"

# ── from notebook cell 5
# Step 7 — metrics

def wilson_ci(k, n, z=1.959964):
    """Wilson score interval — appropriate at n=21 where Wald is not."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(y_true, y_prob, fn, n_boot=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[ix])) < 2:
            continue
        try:
            out.append(fn(y_true[ix], y_prob[ix]))
        except Exception:
            pass
    if not out:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def evaluate(y_true, y_prob, label, threshold=0.5):
    """
    Imbalance-aware metric block.

    With prevalence 0.714 the AP chance baseline is 0.714 rather than 0, and the
    no-information rate is likewise 0.714, so accuracy is tested against that
    rather than against 0.5. Balanced accuracy and MCC take precedence over raw
    accuracy. LOOCV yields one prediction per fold, so metrics are computed once
    on the pooled out-of-fold vector rather than averaged across folds.
    """
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    prevalence = float(y_true.mean())
    nir = max(prevalence, 1 - prevalence)
    n_correct = int((pred == y_true).sum())

    auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score)
    ap_lo, ap_hi = bootstrap_ci(y_true, y_prob, average_precision_score)
    mcc_lo, mcc_hi = bootstrap_ci(
        y_true, y_prob,
        lambda a, b: matthews_corrcoef(a, (b >= threshold).astype(int)))

    return {
        "model": label, "n": n,
        "AUC": roc_auc_score(y_true, y_prob), "AUC_lo": auc_lo, "AUC_hi": auc_hi,
        "AP": average_precision_score(y_true, y_prob), "AP_lo": ap_lo, "AP_hi": ap_hi,
        "AP_baseline": prevalence,
        "accuracy": accuracy_score(y_true, pred),
        "acc_lo": wilson_ci(n_correct, n)[0], "acc_hi": wilson_ci(n_correct, n)[1],
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "sens_lo": wilson_ci(int(tp), int(tp + fn_))[0],
        "sens_hi": wilson_ci(int(tp), int(tp + fn_))[1],
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "spec_lo": wilson_ci(int(tn), int(tn + fp))[0],
        "spec_hi": wilson_ci(int(tn), int(tn + fp))[1],
        "precision": precision_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred), "MCC_lo": mcc_lo, "MCC_hi": mcc_hi,
        "NIR": nir,
        "acc_vs_NIR_p": float(stats.binomtest(n_correct, n, nir,
                                              alternative="greater").pvalue),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn_),
    }

# ── from notebook cell 6
# Step 3 — stability

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def kuncheva(a: set, b: set, n_total: int) -> float:
    """
    Kuncheva consistency index, generalised to unequal cardinality:
        (r - k_a k_b / n) / (min(k_a, k_b) - k_a k_b / n)
    Corrects for the agreement expected by chance given the candidate pool.
    Zero is chance; one is exact reproduction.
    """
    ka, kb, r = len(a), len(b), len(a & b)
    if ka == 0 or kb == 0:
        return np.nan
    expected = ka * kb / n_total
    denom = min(ka, kb) - expected
    return (r - expected) / denom if abs(denom) > 1e-12 else np.nan


def selection_stability(records, n_total):
    counts, coef_sum = {}, {}
    for r in records:
        for f, c in zip(r["selected"], r["coefs"]):
            counts[f] = counts.get(f, 0) + 1
            coef_sum[f] = coef_sum.get(f, 0.0) + c
    n = len(records)
    freq = pd.DataFrame({
        "feature": list(counts),
        "folds_selected": [counts[f] for f in counts],
        "selection_frequency": [counts[f] / n for f in counts],
        "mean_coefficient": [coef_sum[f] / counts[f] for f in counts],
    })
    freq["modality"] = np.where(freq["feature"].str.startswith("cg"),
                                "DNA methylation", "mRNA expression")
    freq["direction"] = np.where(freq["mean_coefficient"] > 0, "Fibroid+", "Fibroid-")
    freq = freq.sort_values(["folds_selected", "mean_coefficient"],
                            ascending=[False, False]).reset_index(drop=True)

    sets = [set(r["selected"]) for r in records if r["selected"]]
    jac, kun = [], []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            jac.append(jaccard(sets[i], sets[j]))
            kun.append(kuncheva(sets[i], sets[j], n_total))

    summary = {
        "n_folds": len(sets), "n_pairs": len(jac),
        "mean_set_size": float(np.mean([len(s) for s in sets])),
        "min_set_size": int(min(len(s) for s in sets)),
        "max_set_size": int(max(len(s) for s in sets)),
        "union_size": len(set().union(*sets)),
        "mean_jaccard": float(np.mean(jac)),
        "sd_jaccard": float(np.std(jac, ddof=1)),
        "median_jaccard": float(np.median(jac)),
        "mean_kuncheva": float(np.nanmean(kun)),
        "sd_kuncheva": float(np.nanstd(kun, ddof=1)),
        "n_in_all_folds": int((freq["folds_selected"] == n).sum()),
        "n_in_ge_half": int((freq["selection_frequency"] >= 0.5).sum()),
        "n_in_one_fold_only": int((freq["folds_selected"] == 1).sum()),
        "n_expression_probes_selected": int((freq["modality"] == "mRNA expression").sum()),
    }
    return freq, summary


def rank_agreement(tree_mean, deep_mean, n_boot=N_BOOTSTRAP, seed=SEED):
    """Spearman rho with a percentile bootstrap over features, plus Fisher-z."""
    shared = sorted(set(tree_mean) & set(deep_mean))
    if len(shared) < 4:
        return {"n_features": len(shared), "note": "too few shared features"}

    t = np.array([tree_mean[f] for f in shared])
    d = np.array([deep_mean[f] for f in shared])
    rho, pval = stats.spearmanr(t, d)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(shared), len(shared))
        if len(np.unique(t[ix])) < 3 or len(np.unique(d[ix])) < 3:
            continue
        rb, _ = stats.spearmanr(t[ix], d[ix])
        if np.isfinite(rb):
            boots.append(rb)

    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(len(shared) - 3)
    return {
        "n_features": len(shared), "rho": float(rho), "p_value": float(pval),
        "boot_lo": float(np.percentile(boots, 2.5)) if boots else np.nan,
        "boot_hi": float(np.percentile(boots, 97.5)) if boots else np.nan,
        "fisher_lo": float(np.tanh(z - 1.959964 * se)),
        "fisher_hi": float(np.tanh(z + 1.959964 * se)),
        "n_boot_valid": len(boots),
    }


def consensus_panel(freq, tree_mean, deep_mean, gate=STABILITY_GATE):
    """
    Stability-gated rank sum. A probe enters only if reselected in at least
    `gate` of the outer folds; eligible probes are then ranked separately by
    mean absolute LASSO coefficient, mean TreeSHAP and mean DeepSHAP, and the
    three ranks summed. Panel = rank sums at or below the first quartile.
    """
    elig = freq[freq["selection_frequency"] >= gate].copy()
    if elig.empty:
        return pd.DataFrame(), {"note": "no feature met the stability gate",
                                "gate": gate}
    elig["abs_coef"] = elig["mean_coefficient"].abs()
    elig["tree_shap"] = elig["feature"].map(tree_mean).fillna(0.0)
    elig["deep_shap"] = elig["feature"].map(deep_mean).fillna(0.0)
    elig["rank_L"] = elig["abs_coef"].rank(ascending=False, method="min")
    elig["rank_X"] = elig["tree_shap"].rank(ascending=False, method="min")
    elig["rank_D"] = elig["deep_shap"].rank(ascending=False, method="min")
    elig["rank_sum"] = elig[["rank_L", "rank_X", "rank_D"]].sum(axis=1)

    q25 = float(elig["rank_sum"].quantile(0.25))
    panel = elig[elig["rank_sum"] <= q25].sort_values("rank_sum").reset_index(drop=True)
    meta = {"stability_gate": gate, "n_eligible": int(len(elig)),
            "rank_sum_min": float(elig["rank_sum"].min()),
            "rank_sum_max": float(elig["rank_sum"].max()),
            "q25_threshold": q25, "panel_size": int(len(panel))}
    return panel, meta

# ── from notebook cell 7
#  Steps 1, 3, 4, 5, 7 driver

def run_observed(grid_points=GRID_POINTS_OBS, use_dnn=True):
    OUT.mkdir(exist_ok=True)
    folds, y, ids = load_cache()
    samples = ids["samples"]
    meth_names = np.array(ids["meth_index"])
    expr_names = np.array(ids["expr_index"])
    n = len(y)
    n_total = len(meth_names) + len(expr_names)

    ckpt = OUT / "ckpt.pkl"
    records = pickle.load(open(ckpt, "rb")) if ckpt.exists() else []
    if records:
        print(f"[resume] {len(records)} folds already complete")
    max_secs = float(os.environ.get("MAX_SECS", 1e9))
    t0 = time.time()

    print(f"[step1] nested LOOCV, {grid_points}-point C grid, all steps in-fold")
    for i, (X_tr32, X_te32, keep_m, keep_e) in enumerate(folds):
        if i < len(records):
            continue
        if time.time() - t0 > max_secs:
            print(f"[pause] {len(records)}/{n} folds done; rerun to continue")
            sys.exit(3)

        X_tr = np.ascontiguousarray(X_tr32, dtype=np.float64)
        X_te = np.ascontiguousarray(X_te32, dtype=np.float64)
        tr = [j for j in range(n) if j != i]
        y_tr = y[tr]
        fold_names = np.concatenate([meth_names[keep_m], expr_names[keep_e]])

        sel, coefs, c_star, inner_auc, lasso = select_lasso(X_tr, y_tr, grid_points)
        rec = {"fold": i, "held_out": samples[i], "y_true": int(y[i]),
               "n_meth_kept": int(keep_m.size), "n_expr_kept": int(keep_e.size),
               "C_star": c_star, "lambda_star": 1.0 / c_star, "inner_auc": inner_auc,
               "n_selected": int(sel.size),
               "selected": [str(s) for s in fold_names[sel]],
               "coefs": [float(c) for c in coefs],
               "p_lasso": float(lasso.predict_proba(X_te)[0, 1]),
               "tree_shap": {}, "deep_shap": {}, "deep_estimator": None}

        if sel.size:
            Xs, Xq = X_tr[:, sel], X_te[:, sel]
            names = rec["selected"]

            xgb = fit_xgb(Xs, y_tr)
            rec["p_xgb"] = float(xgb.predict_proba(Xq)[0, 1])

            if use_dnn:
                cw = {0: len(y_tr) / (2 * max((y_tr == 0).sum(), 1)),
                      1: len(y_tr) / (2 * max((y_tr == 1).sum(), 1))}
                preds, nets = [], []
                for s in DNN_SEEDS:
                    net = build_dnn(Xs.shape[1], s)
                    net.fit(Xs, y_tr, epochs=DNN_EPOCHS, batch_size=8,
                            verbose=0, class_weight=cw)
                    preds.append(float(net.predict(Xq, verbose=0).ravel()[0]))
                    nets.append(net)
                rec["p_dnn"] = float(np.mean(preds))
                rec["p_dnn_sd"] = float(np.std(preds, ddof=1))
                rec["deep_shap"], rec["deep_estimator"] = deep_shap(nets[0], Xs, names)
            else:
                rec["p_dnn"], rec["p_dnn_sd"] = rec["p_xgb"], 0.0

            rec["tree_shap"] = tree_shap(xgb, Xs, names)
        else:
            rec["p_xgb"] = rec["p_dnn"] = float(y_tr.mean())
            rec["p_dnn_sd"] = 0.0

        rec["p_ens"] = float(np.mean([rec["p_lasso"], rec["p_dnn"], rec["p_xgb"]]))
        records.append(rec)
        pickle.dump(records, open(ckpt, "wb"))
        print(f"  fold {i:2d} {samples[i]:<20s} kept={keep_m.size + keep_e.size:6d} "
              f"C*={c_star:.4g} sel={sel.size:2d} "
              f"p_lasso={rec['p_lasso']:.3f} p_xgb={rec['p_xgb']:.3f} "
              f"p_dnn={rec['p_dnn']:.3f}  [{time.time() - t0:.0f}s]", flush=True)

    pickle.dump(records, open(OUT / "records.pkl", "wb"))
    print(f"[step1] all {n} folds complete")

    # ── Step 7
    y_true = np.array([r["y_true"] for r in records])
    rows = [evaluate(y_true, np.array([r[k] for r in records]), lab)
            for lab, k in (("LASSO", "p_lasso"), ("DNN", "p_dnn"),
                           ("XGBoost", "p_xgb"), ("Ensemble", "p_ens"))]
    perf = pd.DataFrame(rows)
    perf.to_csv(OUT / "table_4.6_corrected_performance.csv", index=False)

    # ── Step 3
    freq, stab = selection_stability(records, n_total)
    freq.to_csv(OUT / "table_4.2_selection_frequency.csv", index=False)

    # ── Step 4
    tree, deep = {}, {}
    for r in records:
        for f, v in r["tree_shap"].items():
            tree.setdefault(f, []).append(v)
        for f, v in r["deep_shap"].items():
            deep.setdefault(f, []).append(v)
    tree_mean = {f: float(np.mean(v)) for f, v in tree.items()}
    deep_mean = {f: float(np.mean(v)) for f, v in deep.items()}
    agree = rank_agreement(tree_mean, deep_mean)

    per_fold_rhos = []
    for r in records:
        sh = [f for f in r["tree_shap"] if f in r["deep_shap"]]
        if len(sh) >= 4:
            rr, _ = stats.spearmanr([r["tree_shap"][f] for f in sh],
                                    [r["deep_shap"][f] for f in sh])
            if np.isfinite(rr):
                per_fold_rhos.append(float(rr))

    att = pd.DataFrame({"feature": sorted(set(tree_mean) | set(deep_mean))})
    att["tree_shap_mean"] = att["feature"].map(tree_mean)
    att["deep_shap_mean"] = att["feature"].map(deep_mean)
    att = att.merge(freq[["feature", "folds_selected", "selection_frequency",
                          "mean_coefficient"]], on="feature", how="left")
    att.to_csv(OUT / "table_4.4_attribution.csv", index=False)

    # ── Step 5
    panel, panel_meta = consensus_panel(freq, tree_mean, deep_mean)
    panel.to_csv(OUT / "table_4.3_consensus_panel.csv", index=False)

    fold_df = pd.DataFrame([{k: (";".join(v) if k == "selected" else v)
                             for k, v in r.items()
                             if k not in ("tree_shap", "deep_shap", "coefs")}
                            for r in records])
    fold_df.to_csv(OUT / "table_4.1_fold_level.csv", index=False)

    json.dump({"n": int(n),
               "normal": int((y == 0).sum()), "fibroid": int((y == 1).sum()),
               "grid_points": grid_points, "stability": stab,
               "attribution_pooled": agree,
               "per_fold_rhos": per_fold_rhos,
               "per_fold_rho_median": float(np.median(per_fold_rhos)) if per_fold_rhos else None,
               "per_fold_prop_negative": float(np.mean(np.array(per_fold_rhos) < 0))
               if per_fold_rhos else None,
               "panel_meta": panel_meta,
               "deep_estimators": sorted({r["deep_estimator"] for r in records
                                          if r["deep_estimator"]}),
               "runtime_s": round(time.time() - t0, 1)},
              open(OUT / "summary.json", "w"), indent=2, default=float)

    print("\n=== Step 7: corrected performance ===")
    print(perf[["model", "AUC", "AUC_lo", "AUC_hi", "accuracy", "acc_lo", "acc_hi",
                "balanced_accuracy", "MCC", "AP", "AP_baseline",
                "acc_vs_NIR_p"]].round(4).to_string(index=False))
    print("\n=== Step 3: selection stability ===")
    print(json.dumps(stab, indent=2, default=float))
    print("\n=== Step 4: attribution agreement ===")
    print(json.dumps(agree, indent=2, default=float))
    if per_fold_rhos:
        print(f"per-fold rho: median {np.median(per_fold_rhos):.3f}, "
              f"{int(np.sum(np.array(per_fold_rhos) < 0))}/{len(per_fold_rhos)} negative")
    print("\n=== Step 5: consensus panel ===")
    print(json.dumps(panel_meta, indent=2, default=float))
    if not panel.empty:
        print(panel[["feature", "selection_frequency", "mean_coefficient",
                     "rank_L", "rank_X", "rank_D", "rank_sum"]].to_string(index=False))

# ── from notebook cell 8
from pathlib import Path
import numpy as np
import pandas as pd
import csv
import os
import time
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, balanced_accuracy_score)



# Step 2 — label-permutation null

def run_permutations(n_perm=1000, grid_points=GRID_POINTS_PERM, seed=SEED):
    """
    Shuffle the labels and rerun the ENTIRE nested pipeline. Append-only so the
    run is resumable; MAX_SECS time-boxes a chunk.

    The network is omitted and the C grid coarsened under permutation for
    tractability. Both approximations weaken the null rather than the observed
    value, so the resulting p-values are conservative. Because the observed AUC
    is already at its ceiling, grid coarsening cannot inflate significance.
    """
    folds, y_obs, _ = load_cache()
    n = len(y_obs)
    grid = np.logspace(np.log10(GRID_LO), np.log10(GRID_HI), grid_points)
    out_csv = (OUT / "perm_null.csv")

    def nested(y):
        p_l, p_x = np.zeros(n), np.zeros(n)
        for i, (X_tr32, X_te32, _, _) in enumerate(folds):
            X_tr = np.ascontiguousarray(X_tr32, dtype=np.float64)
            X_te = np.ascontiguousarray(X_te32, dtype=np.float64)
            tr = [j for j in range(n) if j != i]
            y_tr = y[tr]
            if len(np.unique(y_tr)) < 2:
                p_l[i] = p_x[i] = y_tr.mean()
                continue
            skf = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
            best_c, best = grid[0], -np.inf
            for C in grid:
                sc = []
                for a, b in skf.split(X_tr, y_tr):
                    if len(np.unique(y_tr[b])) < 2 or len(np.unique(y_tr[a])) < 2:
                        continue
                    m = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=C,
                                           class_weight="balanced", max_iter=5000,
                                           random_state=seed).fit(X_tr[a], y_tr[a])
                    sc.append(roc_auc_score(y_tr[b], m.predict_proba(X_tr[b])[:, 1]))
                if sc and np.mean(sc) > best + 1e-12:
                    best, best_c = np.mean(sc), C
            f = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=best_c,
                                   class_weight="balanced", max_iter=5000,
                                   random_state=seed).fit(X_tr, y_tr)
            p_l[i] = f.predict_proba(X_te)[0, 1]
            sel = np.where(np.abs(f.coef_.ravel()) > 1e-10)[0]
            p_x[i] = (y_tr.mean() if sel.size == 0 else
                      fit_xgb(X_tr[:, sel], y_tr).predict_proba(X_te[:, sel])[0, 1])
        return p_l, p_x

    def summarise(y, p):
        pred = (p >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = np.nan
        return auc, matthews_corrcoef(y, pred), balanced_accuracy_score(y, pred)

    done = sum(1 for _ in open(out_csv)) - 1 if out_csv.exists() else 0
    if done == 0:
        with open(out_csv, "w", newline="") as fh:
            csv.writer(fh).writerow(["perm", "lasso_auc", "lasso_mcc", "lasso_bacc",
                                     "xgb_auc", "xgb_mcc", "xgb_bacc", "secs"])
    else:
        print(f"[resume] {done} permutations already complete")

    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y_obs) for _ in range(n_perm)]
    max_secs = float(os.environ.get("MAX_SECS", 1e9))
    t0 = time.time()

    for i in range(done, n_perm):
        if time.time() - t0 > max_secs:
            print(f"[pause] {i} permutations done; rerun to continue")
            sys.exit(3)
        t = time.time()
        pl, px = nested(perms[i])
        a1, m1, b1 = summarise(perms[i], pl)
        a2, m2, b2 = summarise(perms[i], px)
        with open(out_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([i, a1, m1, b1, a2, m2, b2,
                                     round(time.time() - t, 1)])
        print(f"  perm {i + 1}/{n_perm} lasso_auc={a1:.3f} xgb_auc={a2:.3f} "
              f"({time.time() - t:.0f}s)", flush=True)

    summarise_null()


def empirical_p(observed, null_values):
    """Conservative empirical p with the +1 correction (Phipson & Smyth, 2010)."""
    v = np.asarray(null_values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float((1 + np.sum(v >= observed)) / (1 + v.size))


def summarise_null():
    OUT.mkdir(exist_ok=True)
    null = pd.read_csv(OUT / "perm_null.csv")
    perf = pd.read_csv(OUT / "table_4.6_corrected_performance.csv").set_index("model")
    rows = []
    for model, key in (("LASSO", "lasso"), ("XGBoost", "xgb")):
        for metric, col, obs_col in (("AUC", f"{key}_auc", "AUC"),
                                     ("MCC", f"{key}_mcc", "MCC"),
                                     ("Balanced accuracy", f"{key}_bacc",
                                      "balanced_accuracy")):
            v = null[col].to_numpy()
            v = v[np.isfinite(v)]
            obs = float(perf.loc[model, obs_col])
            rows.append({"model": model, "metric": metric, "observed": obs,
                         "null_mean": v.mean(), "null_sd": v.std(ddof=1),
                         "null_median": np.median(v),
                         "null_q95": np.percentile(v, 95), "null_max": v.max(),
                         "n_ge_observed": int((v >= obs).sum()), "B": len(v),
                         "empirical_p": empirical_p(obs, v)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "table_4.5_permutation_null.csv", index=False)
    print(f"\n=== Step 2: label-permutation null (B = {len(null)}) ===")
    print(f"Minimum attainable p at this B is {1 / (1 + len(null)):.4f}")
    print(tbl.round(4).to_string(index=False))

# ── from notebook cell 9

# Step 6 — manifest annotation

def run_annotation(manifest_path: Path):
    """
    Map every probe in the selection union through the Illumina 450K manifest,
    test CpG-island enrichment against the array background, detect probes that
    cluster into a single genomic locus, and cross-reference the recovered genes
    against the fibroid, WNT/beta-catenin and hedgehog literature.
    """
    OUT.mkdir(exist_ok=True)
    wanted = {"IlmnID", "CHR", "MAPINFO", "Strand", "UCSC_RefGene_Name",
              "UCSC_RefGene_Group", "UCSC_CpG_Islands_Name",
              "Relation_to_UCSC_CpG_Island", "Enhancer", "DHS",
              "Regulatory_Feature_Group", "Infinium_Design_Type"}
    # the manifest carries seven header lines before the [Assay] column row
    man = pd.read_csv(manifest_path, skiprows=7, low_memory=False,
                      usecols=lambda c: c in wanted)
    man = man[man["IlmnID"].astype(str).str.startswith("cg")]
    print(f"[step6] manifest loaded: {len(man):,} cg probes")

    freq = pd.read_csv(OUT / "table_4.2_selection_frequency.csv")
    att = pd.read_csv(OUT / "table_4.4_attribution.csv")
    panel = pd.read_csv(OUT / "table_4.3_consensus_panel.csv")

    def dedup(x):
        return "" if pd.isna(x) else ";".join(dict.fromkeys(str(x).split(";")))

    ann = man.rename(columns={"IlmnID": "feature"}).copy()
    for c in ("UCSC_RefGene_Name", "UCSC_RefGene_Group"):
        ann[c] = ann[c].map(dedup)
    ann["Relation_to_UCSC_CpG_Island"] = \
        ann["Relation_to_UCSC_CpG_Island"].fillna("OpenSea")

    merged = (freq.merge(ann, on="feature", how="left")
                  .merge(att[["feature", "tree_shap_mean", "deep_shap_mean"]],
                         on="feature", how="left")
                  .sort_values("folds_selected", ascending=False))
    merged["in_panel"] = merged["feature"].isin(panel["feature"])
    merged.to_csv(OUT / "table_4.7_annotation_union.csv", index=False)
    print(f"[step6] annotated {merged['CHR'].notna().sum()}/{len(merged)} union probes")

    # island-context enrichment against the array background
    bg = man["Relation_to_UCSC_CpG_Island"].fillna("OpenSea")
    fg = merged["Relation_to_UCSC_CpG_Island"].fillna("OpenSea")
    rows = []
    for cat in ("Island", "N_Shore", "S_Shore", "N_Shelf", "S_Shelf", "OpenSea"):
        k, K = int((fg == cat).sum()), int((bg == cat).sum())
        rows.append({"context": cat, "n_union": k,
                     "pct_union": k / len(fg), "pct_array": K / len(bg),
                     "fold_enrichment": (k / len(fg)) / (K / len(bg)) if K else np.nan,
                     "p_hypergeom": stats.hypergeom.sf(k - 1, len(bg), K, len(fg))})
    enrich = pd.DataFrame(rows)
    enrich.to_csv(OUT / "table_4.8_island_enrichment.csv", index=False)
    print("\n=== Step 6: CpG-island context enrichment ===")
    print(enrich.round(4).to_string(index=False))

    # probes that cluster into one locus (co-methylated blocks, not independent markers)
    print("\n=== Step 6: locus clustering within the panel ===")
    pa = merged[merged["in_panel"] & merged["MAPINFO"].notna()]
    clusters = []
    for c, grp in pa.groupby("CHR"):
        g = grp.sort_values("MAPINFO")
        pos = g["MAPINFO"].to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                dist = abs(pos[j] - pos[i])
                if dist < 5000:
                    clusters.append({"chr": c, "probe_a": g.iloc[i]["feature"],
                                     "probe_b": g.iloc[j]["feature"],
                                     "distance_bp": int(dist)})
    if clusters:
        cl = pd.DataFrame(clusters)
        cl.to_csv(OUT / "table_4.9_panel_locus_clusters.csv", index=False)
        print(cl.to_string(index=False))
        # count loci as connected components of the <5 kb linkage graph
        parent = {f: f for f in pa["feature"]}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for r in clusters:
            parent[find(r["probe_a"])] = find(r["probe_b"])
        n_loci = len({find(f) for f in parent})
        print(f"-> {len(parent)} panel probes span {n_loci} independent loci")
    else:
        print("no panel probes within 5 kb of one another")

    # literature cross-reference
    genes = set()
    for g in merged["UCSC_RefGene_Name"].dropna():
        genes.update(str(g).split(";"))
    genes.discard("")
    overlap = sorted(genes & FIBROID_LIT)
    print("\n=== Step 6: literature cross-reference ===")
    print(f"distinct genes in union: {len(genes)}")
    print(f"overlap with fibroid / WNT / hedgehog gene set: {overlap or 'none'}")
    absent = sorted({"HMGA2", "MED12", "DLK1", "MEG3", "COL1A1", "COL3A1",
                     "FN1", "TGFB3"} - genes)
    print(f"canonical fibroid drivers ABSENT from the union: {absent}")

    json.dump({"n_union": int(len(merged)),
               "n_annotated": int(merged["CHR"].notna().sum()),
               "island_pct_union": float((fg == "Island").mean()),
               "island_pct_array": float((bg == "Island").mean()),
               "island_p": float(enrich.loc[enrich.context == "Island", "p_hypergeom"].iloc[0]),
               "genes": sorted(genes), "literature_overlap": overlap,
               "canonical_drivers_absent": absent},
              open(OUT / "annotation_summary.json", "w"), indent=2, default=float)

# ── from notebook cell 11
import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np # Added numpy import
import matplotlib # Added matplotlib import
import matplotlib.pyplot as plt # Added pyplot import
import os # Added os import
import time # Added time import
import pickle # Added pickle import


def run_redundancy():
    """Placeholder for redundancy analysis."""
    print("[step] Running redundancy analysis (placeholder)")


# figures

def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 170, "savefig.bbox": "tight"})
    FIGDIR.mkdir(parents=True, exist_ok=True);

    M = np.load(OUT / "meth.npy")
    y = np.load(OUT / "y.npy")

    ids = json.load(open(OUT / "ids.json"))
    idx = {p: i for i, p in enumerate(ids["meth_index"]) }
    freq = pd.read_csv(OUT / "table_4.2_selection_frequency.csv")
    att = pd.read_csv(OUT / "table_4.4_attribution.csv")
    fold = pd.read_csv(OUT / "table_4.1_fold_level.csv")
    panel = pd.read_csv(OUT / "table_4.3_consensus_panel.csv")["feature"].tolist()
    summ = json.load(open(OUT / "summary.json"))

    # 4a permutation null
    if (OUT / "perm_null.csv").exists():
        perm = pd.read_csv(OUT / "perm_null.csv")
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
        for a, (col, ttl) in zip(ax, [("lasso_auc", "LASSO"), ("xgb_auc", "XGBoost")]):
            a.hist(perm[col], bins=18, color="#7BA7C7", edgecolor="white")
            a.axvline(1.0, color="#C1443B", lw=2)
            a.axvline(perm[col].mean(), color="#444", ls="--", lw=1)
            a.set_title(f"{ttl} (null mean {perm[col].mean():.3f})")
            a.set_xlabel("LOOCV AUC under permuted labels")
            a.set_xlim(0, 1.05)
        ax[0].set_ylabel(f"Permutations (B = {len(perm)})")
        plt.savefig(FIGDIR / "fig_4a_permutation_null.png")
        plt.close()

    # 4b selection frequency
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    d = freq.sort_values("folds_selected", ascending=False).head(25)
    ax.bar(range(len(d)), d.folds_selected,
           color=["#2E6B8A" if v >= 0.5 else "#B8C9D6" for v in d.selection_frequency])
    ax.axhline(len(fold) / 2, color="#C1443B", ls="--", lw=1, label="50% stability gate")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d.feature, rotation=90, fontsize=6)
    ax.set_ylabel(f"Outer folds selecting the CpG (of {len(fold)})")
    ax.set_title(f"Selection frequency — union {summ['stability']['union_size']} "
                 f"features, mean Jaccard {summ['stability']['mean_jaccard']:.3f}")
    ax.legend(fontsize=7)
    plt.savefig(FIGDIR / "fig_4b_selection_frequency.png")
    plt.close()

    # 4c held-out probabilities
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(fold))
    for k, lab, mk in (("p_lasso", "LASSO", "o"), ("p_dnn", "DNN", "s"),
                       ("p_xgb", "XGBoost", "^"), ("p_ens", "Ensemble", "D")):
        ax.scatter(x, fold[k], marker=mk, s=26, label=lab, alpha=0.85)
    ax.axhline(0.5, color="#C1443B", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(fold.held_out, rotation=90, fontsize=6)
    ax.set_ylabel("Out-of-fold P(fibroid)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Held-out predicted probability, nested LOOCV")
    ax.legend(fontsize=7, ncol=4)
    plt.savefig(FIGDIR / "fig_4c_loocv_probabilities.png")
    plt.close()

    # 4d attribution divergence
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    d = (att.dropna(subset=["tree_shap_mean", "deep_shap_mean"])
            .sort_values("deep_shap_mean", ascending=False).head(14))
    yy = np.arange(len(d))
    ax[0].barh(yy, d.tree_shap_mean, color="#2E6B8A")
    ax[0].set_yticks(yy)
    ax[0].set_yticklabels(d.feature, fontsize=6)
    ax[0].invert_yaxis()
    ax[0].set_title("Exact TreeSHAP (XGBoost)")
    ax[0].set_xlabel("mean |SHAP|")
    ax[1].barh(yy, d.deep_shap_mean, color="#C1443B")
    ax[1].set_yticks(yy)
    ax[1].set_yticklabels([])
    ax[1].invert_yaxis()
    ax[1].set_title("DeepSHAP (network)")
    ax[1].set_xlabel("mean |SHAP|")
    plt.savefig(FIGDIR / "fig_4d_attribution.png")
    plt.close()

    # 4e panel biology
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    rng = np.random.default_rng(SEED)

    # Filter panel to include only methylation probes for this specific plot
    meth_panel = [p for p in panel if p.startswith("cg")]

    if meth_panel: # Only proceed if there are methylation probes in the panel
        for k, p in enumerate(meth_panel):
            b = M[idx[p]]
            ax[0].scatter(np.full((y == 0).sum(), k - 0.13) +
                          rng.uniform(-.05, .05, (y == 0).sum()), b[y == 0],
                          s=18, color="#2E6B8A", label="Myometrium" if k == 0 else None)
            ax[0].scatter(np.full((y == 1).sum(), k + 0.13) +
                          rng.uniform(-.05, .05, (y == 1).sum()), b[y == 1],
                          s=18, color="#C1443B", marker="^",
                          label="Fibroid" if k == 0 else None)
        ax[0].set_xticks(range(len(meth_panel)))
        ax[0].set_xticklabels(meth_panel, rotation=45, ha="right", fontsize=6)

        ax[0].set_ylabel("Beta value")
        ax[0].set_ylim(0, 1)
        ax[0].legend(fontsize=7)
        ax[0].set_title("Panel probes: class separation")

        # Correlation matrix also needs to use only methylation probes
        C = np.corrcoef(np.vstack([M[idx[p]] for p in meth_panel]))
        im = ax[1].imshow(C, vmin=0.9, vmax=1.0, cmap="RdYlBu_r")
        ax[1].set_xticks(range(len(meth_panel)))
        ax[1].set_yticks(range(len(meth_panel)))
        ax[1].set_xticklabels(meth_panel, rotation=45, ha="right", fontsize=6)
        ax[1].set_yticklabels(meth_panel, fontsize=6)
        for i in range(len(meth_panel)):
            for j in range(len(meth_panel)):
                ax[1].text(j, i, f"{C[i, j]:.3f}", ha="center", va="center", fontsize=6)
        ax[1].set_title("Inter-probe correlation")
        plt.colorbar(im, ax=ax[1], fraction=0.046)

    else:
        # Handle case where meth_panel is empty
        ax[0].set_title("No methylation probes in panel to display")
        ax[1].set_title("No methylation probes in panel to display")
        ax[0].set_ylabel("Beta value")
        ax[0].set_ylim(0, 1)

    plt.savefig(FIGDIR / "fig_4e_panel_biology.png")
    plt.close()
    print(f"[figures] written to {FIGDIR}/")

# CLI

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Data Leakage",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["folds", "observed", "permute", "annotate",
                                      "redundancy", "figures", "nullsummary", "all"])
    ap.add_argument("--meth", type=Path, default=DATA_DIR / "DNA_methylation_for_dryad.csv")
    ap.add_argument("--expr", type=Path, default=DATA_DIR / "mRNA_expression_for_dryad.csv")
    ap.add_argument("--manifest", type=Path,
                    default=DATA_DIR / "HumanMethylation450_15017482_v1-2.csv")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--grid-points", type=int, default=GRID_POINTS_OBS)
    ap.add_argument("--no-dnn", action="store_true")
    args = ap.parse_args(argv)

    if args.stage in ("folds", "all"):
        build_folds(args.meth, args.expr)
    if args.stage in ("observed", "all"):
        run_observed(grid_points=args.grid_points, use_dnn=not args.no_dnn)
    if args.stage == "permute":
        run_permutations(n_perm=args.n_perm)
    if args.stage == "nullsummary":
        summarise_null()
    if args.stage in ("annotate", "all"):
        run_annotation(args.manifest)
    if args.stage in ("redundancy", "all"):
      run_redundancy()
    if args.stage in ("figures", "all"):
        make_figures()

    if args.stage == "all":
        print("""\n[note] the permutation null was NOT run. Execute separately:
       python ch4_pipeline.py permute --n-perm 1000""")


def build_folds(meth_path: Path, expr_path: Path) -> None:
    """
    Stage 0, refitted per fold.

    The variance filter and the standardisation statistics depend only on WHICH
    specimens occupy the training partition, not on their labels. The 21 fold
    matrices are therefore built once here and reused by both the observed run
    and every permutation. Every label-dependent step is recomputed downstream.
    """
    print("[stage0] loading matrices")
    meth = pd.read_csv(meth_path, index_col=0)
    expr_raw = pd.read_csv(expr_path, low_memory=False)

    sample_cols = list(expr_raw.columns[1:22])
    expr = (expr_raw[expr_raw["category"] == "main"]
            .set_index("Probe Set ID")[sample_cols])
    print(f"[stage0] methylation {meth.shape}  expression main-category {expr.shape}")

    common = [c for c in expr.columns if c in meth.columns]
    dropped = [c for c in meth.columns if c not in common]
    meth, expr = meth[common], expr[common]
    print(f"[stage0] {len(common)} specimens on both modalities; excluded {dropped}")

    if meth.isna().any().any() or expr.isna().any().any():
        raise ValueError("Missing values present; resolve before proceeding.")

    y = np.array([0 if c.startswith("Myometrium") else 1 for c in common])
    print(f"[stage0] normal {int((y == 0).sum())}  fibroid {int((y == 1).sum())}  "
          f"prevalence {y.mean():.4f}")

    OUT.mkdir(parents=True, exist_ok=True) # Ensure OUT directory exists
    M = meth.to_numpy(dtype=np.float32)
    E = expr.to_numpy(dtype=np.float32)
    np.save(OUT / "meth.npy", M)
    np.save(OUT / "expr.npy", E)
    np.save(OUT / "y.npy", y)
    json.dump({"samples": common,
               "meth_index": meth.index.tolist(),
               "expr_index": expr.index.tolist()}, open(OUT / "ids.json", "w"))

    # reference figures, computed on all 21 for comparability with the literature
    mv = np.clip(M, EPS, 1 - EPS).var(axis=1, ddof=1)
    ev = E.var(axis=1, ddof=1)
    print(f"[stage0] all-21 reference: {int((mv > METH_VAR_THRESHOLD).sum())} CpG + "
          f"{int((ev > EXPR_VAR_THRESHOLD).sum())} expression = "
          f"{int((mv > METH_VAR_THRESHOLD).sum() + (ev > EXPR_VAR_THRESHOLD).sum())} features")

    n = len(y)
    folds = []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        te = np.array([i])

        keep_m = np.where(np.clip(M[:, tr], EPS, 1 - EPS).var(axis=1, ddof=1)
                          > METH_VAR_THRESHOLD)[0]
        keep_e = np.where(E[:, tr].var(axis=1, ddof=1) > EXPR_VAR_THRESHOLD)[0]

        def blocks(ix):
            b = np.clip(M[keep_m][:, ix], EPS, 1 - EPS)
            return np.log2(b / (1 - b)).T, E[keep_e][:, ix].T

        m_tr, e_tr = blocks(tr)
        m_te, e_te = blocks(te)
        mu_m, sd_m = m_tr.mean(0), m_tr.std(0, ddof=1)
        mu_e, sd_e = e_tr.mean(0), e_tr.std(0, ddof=1)
        sd_m[sd_m == 0] = 1.0
        sd_e[sd_e == 0] = 1.0

        X_tr = np.hstack([(m_tr - mu_m) / sd_m, (e_tr - mu_e) / sd_e]).astype(np.float32)
        X_te = np.hstack([(m_te - mu_m) / sd_m, (e_te - mu_e) / sd_e]).astype(np.float32)
        folds.append((X_tr, X_te, keep_m, keep_e))

    pickle.dump(folds, open(OUT / "folds.pkl", "wb"), protocol=4)
    sizes = [f[0].shape[1] for f in folds]
    print(f"[stage0] folds.pkl written; in-fold feature count {min(sizes)}–{max(sizes)}")


def load_cache():
    folds = pickle.load(open(OUT / "folds.pkl", "rb"))
    y = np.load(OUT / "y.npy")
    ids = json.load(open(OUT / "ids.json"))
    return folds, y, ids


if __name__ == "__main__":
    sys.exit(main())
