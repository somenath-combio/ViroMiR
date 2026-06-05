#!/usr/bin/env python3
"""
model_comparison.py  —  ViroMiR ML Baseline Benchmark
═════════════════════════════════════════════════════════════════
Compares 6 ML models on the SAME 699-sample feature table using
the SAME validation splits:
  1. 10-fold Stratified CV (internal stability)
  2. Leave-One-Virus-Out (unseen-virus generalization)
  3. Train ViRBase → Test VIRmiRNA (external database transfer)

Models: Logistic Regression, Random Forest, SVM (RBF),
        XGBoost, AdaBoost, KNN
═════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from xgboost import XGBClassifier

ROOT = Path("/home/somenath/ViroMiR")
DATA = ROOT / "data/processed/viromir_unified_dataset_v1.csv"

FEATURE_COLS = [
    'delta_G', 'delta_G_norm',
    'n_base_pairs', 'n_watson_crick', 'n_mismatches', 'bp_fraction',
    'seed_exact_match', 'seed_score', 'supplementary_score', 'motif_identity',
    'au_content', 'mirna_gc', 'cts_gc',
    'cts_len', 'mirna_len', 'site_position_norm',
]

# ── Model definitions ─────────────────────────────────────────────
# SVM and KNN need scaling; tree models don't
MODELS = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=3,
        random_state=42, n_jobs=4
    ),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                    probability=True, random_state=42))
    ]),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2.0,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=4
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=200, learning_rate=0.05, random_state=42
    ),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=7, weights='distance'))
    ]),
}

# ── Load data ─────────────────────────────────────────────────────
df = pd.read_csv(DATA)
X  = df[FEATURE_COLS].values
y  = df['label'].values
vb = df[df['source'] == 'virbase'].copy()
vm = df[df['source'] == 'virmirna'].copy()

print("=" * 72)
print("  ViroMiR — ML Model Comparison Benchmark")
print("=" * 72)
print(f"  Dataset: {len(df)} samples, {len(FEATURE_COLS)} features")
print(f"  Labels : pos={y.sum()}, neg={(y==0).sum()}")
print(f"  Sources: ViRBase={len(vb)}, VIRmiRNA={len(vm)}")
print(f"  Models : {len(MODELS)}")
print()

# ══════════════════════════════════════════════════════════════════
#  Test 1: 10-Fold Stratified CV
# ══════════════════════════════════════════════════════════════════
print("── Test 1: 10-Fold Stratified Cross-Validation ──────────────")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_results = {}
for name, model in MODELS.items():
    aucs, aps, mccs = [], [], []
    for tr_idx, va_idx in cv.split(X, y):
        from sklearn.base import clone
        m = clone(model)
        m.fit(X[tr_idx], y[tr_idx])
        prob = m.predict_proba(X[va_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[va_idx], prob))
        aps.append(average_precision_score(y[va_idx], prob))
        mccs.append(matthews_corrcoef(y[va_idx], pred))

    cv_results[name] = {
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
        'ap_mean':  np.mean(aps),  'ap_std':  np.std(aps),
        'mcc_mean': np.mean(mccs), 'mcc_std': np.std(mccs),
    }
    print(f"  {name:<25} AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
          f"AP={np.mean(aps):.4f}±{np.std(aps):.4f}  "
          f"MCC={np.mean(mccs):.4f}±{np.std(mccs):.4f}")

# ══════════════════════════════════════════════════════════════════
#  Test 2: Leave-One-Virus-Out
# ══════════════════════════════════════════════════════════════════
print("\n── Test 2: Leave-One-Virus-Out ──────────────────────────────")
viruses = sorted(df['virus'].unique())

lovo_results = {}
for name, model in MODELS.items():
    lovo_aucs = []
    for v in viruses:
        train = df[df['virus'] != v]
        test  = df[df['virus'] == v]
        if len(np.unique(test['label'].values)) < 2:
            continue
        if len(train) < 50:
            continue
        from sklearn.base import clone
        m = clone(model)
        m.fit(train[FEATURE_COLS].values, train['label'].values)
        prob = m.predict_proba(test[FEATURE_COLS].values)[:, 1]
        try:
            auc = roc_auc_score(test['label'].values, prob)
            lovo_aucs.append(auc)
        except:
            pass

    lovo_results[name] = {
        'auc_mean': np.mean(lovo_aucs),
        'auc_std':  np.std(lovo_aucs),
        'auc_min':  np.min(lovo_aucs),
        'n_viruses': len(lovo_aucs),
    }
    print(f"  {name:<25} LOVO AUC={np.mean(lovo_aucs):.4f}±{np.std(lovo_aucs):.4f}  "
          f"min={np.min(lovo_aucs):.4f}  (n={len(lovo_aucs)} viruses)")

# ══════════════════════════════════════════════════════════════════
#  Test 3: Train ViRBase → Test VIRmiRNA
# ══════════════════════════════════════════════════════════════════
print("\n── Test 3: Train ViRBase → Test VIRmiRNA ────────────────────")
print("  (External database transfer — strictest test)")

X_vb, y_vb = vb[FEATURE_COLS].values, vb['label'].values
X_vm, y_vm = vm[FEATURE_COLS].values, vm['label'].values

xsrc_results = {}
for name, model in MODELS.items():
    from sklearn.base import clone
    m = clone(model)
    m.fit(X_vb, y_vb)
    prob = m.predict_proba(X_vm)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y_vm, prob)
    ap  = average_precision_score(y_vm, prob)
    mcc = matthews_corrcoef(y_vm, pred)
    xsrc_results[name] = {'auc': auc, 'ap': ap, 'mcc': mcc}
    print(f"  {name:<25} AUC={auc:.4f}  AP={ap:.4f}  MCC={mcc:.4f}")

# ══════════════════════════════════════════════════════════════════
#  Publication Table
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  PUBLICATION TABLE — ViroMiR Model Comparison")
print("=" * 72)
print(f"  {'Model':<25} {'10-CV AUC':>12} {'LOVO AUC':>12} {'XSrc AUC':>12}")
print(f"  {'-'*62}")

# Sort by 10-CV AUC descending
sorted_models = sorted(cv_results.keys(),
                        key=lambda n: cv_results[n]['auc_mean'],
                        reverse=True)

for name in sorted_models:
    cv_auc  = f"{cv_results[name]['auc_mean']:.4f}±{cv_results[name]['auc_std']:.4f}"
    lo_auc  = f"{lovo_results[name]['auc_mean']:.4f}±{lovo_results[name]['auc_std']:.4f}"
    xs_auc  = f"{xsrc_results[name]['auc']:.4f}"
    marker  = " ← ViroMiR" if name == "XGBoost" else ""
    print(f"  {name:<25} {cv_auc:>12} {lo_auc:>12} {xs_auc:>12}{marker}")

print(f"  {'-'*62}")

# Determine winner
best_cv   = max(cv_results, key=lambda n: cv_results[n]['auc_mean'])
best_lovo = max(lovo_results, key=lambda n: lovo_results[n]['auc_mean'])
best_xsrc = max(xsrc_results, key=lambda n: xsrc_results[n]['auc'])

print(f"\n  Best 10-CV       : {best_cv}")
print(f"  Best LOVO        : {best_lovo}")
print(f"  Best Cross-Source: {best_xsrc}")

if best_cv == "XGBoost" and best_lovo == "XGBoost":
    print("\n  ✅ XGBoost confirmed as best model — ViroMiR choice justified")
elif best_cv == "XGBoost":
    print(f"\n  ⚠️  XGBoost wins CV but {best_lovo} wins LOVO — worth investigating")
else:
    print(f"\n  ⚠️  {best_cv} outperforms XGBoost on CV — consider model switch")

print("=" * 72)

# ── Save full results to CSV ──────────────────────────────────────
rows = []
for name in sorted_models:
    rows.append({
        'model': name,
        'cv_auc_mean': round(cv_results[name]['auc_mean'], 4),
        'cv_auc_std':  round(cv_results[name]['auc_std'], 4),
        'cv_ap_mean':  round(cv_results[name]['ap_mean'], 4),
        'cv_mcc_mean': round(cv_results[name]['mcc_mean'], 4),
        'lovo_auc_mean': round(lovo_results[name]['auc_mean'], 4),
        'lovo_auc_std':  round(lovo_results[name]['auc_std'], 4),
        'lovo_auc_min':  round(lovo_results[name]['auc_min'], 4),
        'xsrc_auc': round(xsrc_results[name]['auc'], 4),
        'xsrc_ap':  round(xsrc_results[name]['ap'], 4),
        'xsrc_mcc': round(xsrc_results[name]['mcc'], 4),
    })

out_csv = ROOT / "benchmark/results/model_comparison.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\n  Results saved → {out_csv}")
