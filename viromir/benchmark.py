#!/usr/bin/env python3
"""
viromir.benchmark  —  ViroMiR Full End-to-End Benchmark
═══════════════════════════════════════════════════════════════════
Evaluates the FULL ViroMiR pipeline (Stage 1 + Stage 2) on the
85-pair VIRmiRNA hold-out dataset.
═══════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import subprocess
import random
import pickle
import warnings
import json
import argparse
import sys
import os
warnings.filterwarnings("ignore")
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    accuracy_score, matthews_corrcoef,
    confusion_matrix
)

# ── Defaults ──────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "data" / "virmirna_85_normalized.csv"
DEFAULT_MDL  = SCRIPT_DIR / "model" / "viromir_xgb_v1.pkl"
INTARNA_BIN  = os.environ.get("INTARNA_BIN", "IntaRNA")
MIRANDA_BIN  = os.environ.get("MIRANDA_BIN", "miranda")

FEATURE_COLS = [
    'delta_G', 'delta_G_norm',
    'n_base_pairs', 'n_watson_crick', 'n_mismatches', 'bp_fraction',
    'seed_exact_match', 'seed_score', 'supplementary_score', 'motif_identity',
    'au_content', 'mirna_gc', 'cts_gc',
    'cts_len', 'mirna_len', 'site_position_norm',
]

MIN_SEQ_LEN = 40
random.seed(42); np.random.seed(42)

# ══════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════
_COMP = {'A':'T','T':'A','G':'C','C':'G','U':'A'}

def _gc(seq):
    s = seq.upper().replace('U','C')
    return sum(1 for c in s if c in 'GC') / max(len(s),1)

def _au(seq):
    s = seq.upper().replace('T','A')
    return sum(1 for c in s if c in 'AU') / max(len(s),1)

def _seed_exact_match(mi, ct):
    mi=mi.upper().replace('U','T'); ct=ct.upper().replace('U','T')
    if len(mi)<8 or len(ct)<7: return 0
    rc=''.join(_COMP.get(b,'N') for b in reversed(mi[1:8]))
    return 1 if rc in ct else 0

def _seed_score(mi, ct):
    mi=mi.upper().replace('U','T'); ct=ct.upper().replace('U','T')
    if len(mi)<8: return 0.0
    return sum(1 for b in mi[1:8] if _COMP.get(b,'N') in ct)/7.0

def _supplementary_score(mi, ct):
    mi=mi.upper().replace('U','T'); ct=ct.upper().replace('U','T')
    if len(mi)<17: return 0.0
    supp=mi[11:17]
    return sum(1 for b in supp if _COMP.get(b,'N') in ct)/len(supp)

def _parse_hybrid(h):
    if not h or '&' not in h: return 0,0
    s=h.split('&')[0]; return s.count('('), s.count('.')

def run_intarna(mirna_seq, target_seq):
    mi = mirna_seq.upper().replace('T','U')
    tgt = target_seq.upper().replace('T','U')
    cmd = [INTARNA_BIN, "-q", mi, "-t", tgt,
           "--outMode","C",
           "--outCsvCols","E,hybridDP,start1,end1,start2,end2",
           "--threads","1","--seedBP","4"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        for line in r.stdout.strip().split('\n'):
            line = line.strip()
            if not line: continue
            parts = line.split(';')
            try: dg = float(parts[0])
            except ValueError: continue
            def safe_int(s):
                s=s.strip()
                return int(s) if s.lstrip('-').isdigit() else 0
            return {
                'delta_G': dg,
                'hybridDP': parts[1] if len(parts)>1 else '',
                'tgt_start': safe_int(parts[4]) if len(parts)>4 else 0,
                'tgt_end': safe_int(parts[5]) if len(parts)>5 else 0,
            }
    except Exception as e:
        pass
    return None

def build_features(mirna_seq, target_seq, inta, target_len):
    dg = inta['delta_G']
    hyb = inta['hybridDP']
    ts = inta['tgt_start']
    m, mm = _parse_hybrid(hyb)
    sem = _seed_exact_match(mirna_seq, target_seq)
    ss = _seed_score(mirna_seq, target_seq)
    sup = _supplementary_score(mirna_seq, target_seq)
    return {
        'delta_G': dg, 'delta_G_norm': dg/max(len(mirna_seq),1),
        'n_base_pairs': m, 'n_watson_crick': m, 'n_mismatches': mm,
        'bp_fraction': m/max(len(mirna_seq),1), 'seed_exact_match': sem,
        'seed_score': ss, 'supplementary_score': sup,
        'motif_identity': sem * int(sup > 0.5), 'au_content': _au(target_seq),
        'mirna_gc': _gc(mirna_seq), 'cts_gc': _gc(target_seq),
        'cts_len': len(target_seq), 'mirna_len': len(mirna_seq),
        'site_position_norm': ts / max(target_len, 1),
    }

def run_miranda_score(mi, cts):
    with open('/tmp/mi.fa','w') as f: f.write(f">q\\n{mi}\\n")
    with open('/tmp/cts.fa','w') as f: f.write(f">t\\n{cts}\\n")
    try:
        r = subprocess.run(
            [MIRANDA_BIN,'/tmp/mi.fa','/tmp/cts.fa','-sc','50','-en','-1','-quiet'],
            capture_output=True, text=True, timeout=30)
        for line in r.stdout.split('\n'):
            if line.startswith('>>'):
                p = line.split('\t')
                if len(p) >= 5:
                    try: return float(p[4])
                    except: pass
    except: pass
    return 0.0

def confidence_label(prob):
    if prob >= 0.80: return "High"
    if prob >= 0.60: return "Medium"
    if prob >= 0.40: return "Low"
    return "Very low"

def safe_auc(y, s):
    try: return roc_auc_score(y, s)
    except: return float('nan')
def safe_ap(y, s):
    try: return average_precision_score(y, s)
    except: return float('nan')
def prec_at_k(y, s, k):
    top = np.argsort(s)[::-1][:k]
    return y[top].mean()

def main():
    parser = argparse.ArgumentParser(description='Run ViroMiR End-to-End Benchmark')
    parser.add_argument('--outdir', default='benchmark_results', help='Directory to save results (default: benchmark_results)')
    args = parser.parse_args()

    OUT_DIR = Path(args.outdir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  ViroMiR — Full End-to-End Benchmark on VIRmiRNA 85-pair Dataset")
    print("=" * 72)

    if not DEFAULT_DATA.exists():
        print(f"  ❌ Error: benchmark dataset not found at {DEFAULT_DATA}")
        sys.exit(1)
    if not DEFAULT_MDL.exists():
        print(f"  ❌ Error: model file not found at {DEFAULT_MDL}")
        sys.exit(1)

    vm = pd.read_csv(DEFAULT_DATA)
    print(f"\\n  Total VIRmiRNA records : {len(vm)}")

    vm_valid = vm[vm['target_sequence'].str.len() >= MIN_SEQ_LEN].copy().reset_index(drop=True)
    vm_short = vm[vm['target_sequence'].str.len() < MIN_SEQ_LEN]
    print(f"\\n  Excluded (seq < {MIN_SEQ_LEN} nt) : {len(vm_short)}")
    print(f"  Valid positives        : {len(vm_valid)}")

    virus_seqs = defaultdict(list)
    for _, r in vm_valid.iterrows():
        virus_seqs[r['virus']].append(str(r['target_sequence']))

    viruses = list(virus_seqs.keys())

    test_pairs = []
    for _, r in vm_valid.iterrows():
        mi_seq = str(r['miRNA_sequence'])
        cts_seq = str(r['target_sequence'])
        virus = str(r['virus'])
        test_pairs.append({'mirna_id': r['miRNA'], 'mirna_seq': mi_seq, 'target_seq': cts_seq, 'virus': virus, 'label': 1, 'pair_type': 'positive'})
        other = [v for v in viruses if v != virus]
        if other:
            neg_virus = random.choice(other)
            neg_cts = random.choice(virus_seqs[neg_virus])
            test_pairs.append({'mirna_id': r['miRNA'], 'mirna_seq': mi_seq, 'target_seq': neg_cts, 'virus': neg_virus, 'label': 0, 'pair_type': 'cross-virus negative'})

    n_pos = sum(1 for p in test_pairs if p['label']==1)
    n_neg = sum(1 for p in test_pairs if p['label']==0)
    print(f"\\n  Test pairs built       : {len(test_pairs)}")
    print(f"    Positives            : {n_pos}")
    print(f"    Cross-virus negatives: {n_neg}")

    print(f"\\n── Stage 1: IntaRNA Candidate Generation ───────────────────")
    stage1_results = []
    stage1_pass, stage1_fail, stage1_pos_pass, stage1_pos_fail = 0, 0, 0, 0

    for i, pair in enumerate(test_pairs):
        if i % 20 == 0: print(f"  [{i:3d}/{len(test_pairs)}] scanning...")
        inta = run_intarna(pair['mirna_seq'], pair['target_seq'])
        if inta is not None:
            stage1_pass += 1
            if pair['label'] == 1: stage1_pos_pass += 1
            feats = build_features(pair['mirna_seq'], pair['target_seq'], inta, len(pair['target_seq']))
            feats['mirna_id'] = pair['mirna_id']
            feats['virus'] = pair['virus']
            feats['label'] = pair['label']
            feats['pair_type'] = pair['pair_type']
            feats['start'] = inta['tgt_start']
            feats['end'] = inta['tgt_end']
            stage1_results.append(feats)
        else:
            stage1_fail += 1
            if pair['label'] == 1: stage1_pos_fail += 1

    stage1_recall = stage1_pos_pass / max(n_pos, 1)
    print(f"\\n  Stage 1 Results:")
    print(f"    Total scanned        : {len(test_pairs)}")
    print(f"    IntaRNA hits         : {stage1_pass}")
    print(f"    IntaRNA failures     : {stage1_fail}")
    print(f"    ──────────────────────────────────────")
    print(f"    Positive recall      : {stage1_pos_pass}/{n_pos} = {stage1_recall:.4f}")

    print(f"\\n── Stage 2: XGBoost Rescoring ───────────────────────────────")
    with open(DEFAULT_MDL, 'rb') as f:
        pkg = pickle.load(f)
    model = pkg['model']

    feat_matrix = [[r[col] for col in FEATURE_COLS] for r in stage1_results]
    probs = model.predict_proba(feat_matrix)[:, 1]

    for r, prob in zip(stage1_results, probs):
        r['viromir_prob'] = float(prob)
        r['confidence'] = confidence_label(prob)

    y_true = np.array([r['label'] for r in stage1_results])
    y_prob = np.array([r['viromir_prob'] for r in stage1_results])
    y_pred = (y_prob >= 0.50).astype(int)

    print(f"\\n── Full Pipeline Metrics (Stage 1 + Stage 2) ───────────────")
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)

    print(f"  Samples reaching Stage 2: {len(stage1_results)}")
    print(f"  ROC-AUC      : {auc:.4f}")
    print(f"  PR-AUC       : {ap:.4f}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")
    print(f"  F1           : {f1:.4f}")

    print(f"\\n── Baseline Tool Comparison (same test set) ─────────────────")
    sc_intarna = np.array([-r['delta_G'] for r in stage1_results])
    print("  Running miRanda on Stage 2 pairs...")
    sc_miranda = []
    for i, r in enumerate(stage1_results):
        mi = None
        for p in test_pairs:
            if p['mirna_id'] == r['mirna_id'] and p['virus'] == r['virus'] and p['label'] == r['label']:
                mi = p['mirna_seq']
                tgt = p['target_seq']
                break
        sc_miranda.append(run_miranda_score(mi, tgt) if mi else 0.0)

    sc_miranda = np.array(sc_miranda)

    print(f"\\n{'='*72}")
    print(f"  BENCHMARK SUMMARY — VIRmiRNA 85-pair External Test Set")
    print(f"{'='*72}")
    print(f"  {'Method':<30} {'AUC':>7} {'PR-AUC':>7} {'P@10':>6} {'P@20':>6}")
    print(f"  {'-'*58}")
    print(f"  {'IntaRNA (ΔG only)':<30} {safe_auc(y_true,sc_intarna):>7.4f} {safe_ap(y_true,sc_intarna):>7.4f} {prec_at_k(y_true,sc_intarna,10):>6.4f} {prec_at_k(y_true,sc_intarna,20):>6.4f}")
    print(f"  {'miRanda (alignment)':<30} {safe_auc(y_true,sc_miranda):>7.4f} {safe_ap(y_true,sc_miranda):>7.4f} {prec_at_k(y_true,sc_miranda,10):>6.4f} {prec_at_k(y_true,sc_miranda,20):>6.4f}")
    print(f"  {'ViroMiR v1 (full pipeline)':<30} {auc:>7.4f} {ap:>7.4f} {prec_at_k(y_true,y_prob,10):>6.4f} {prec_at_k(y_true,y_prob,20):>6.4f}")
    print(f"{'='*72}")

    results_df = pd.DataFrame(stage1_results)
    results_df.to_csv(OUT_DIR / "virmirna_e2e_predictions.csv", index=False)

    summary = {
        'dataset': 'VIRmiRNA 85-pair',
        'total_positives': n_pos,
        'total_negatives': n_neg,
        'excluded_short': len(vm_short),
        'stage1_recall': round(stage1_recall, 4),
        'stage1_positives_passed': stage1_pos_pass,
        'stage1_positives_lost': stage1_pos_fail,
        'pipeline_roc_auc': round(auc, 4),
        'pipeline_pr_auc': round(ap, 4),
        'pipeline_accuracy': round(acc, 4),
        'pipeline_precision': round(prec, 4),
        'pipeline_recall': round(rec, 4),
        'pipeline_f1': round(f1, 4),
        'pipeline_mcc': round(mcc, 4),
        'intarna_alone_auc': round(safe_auc(y_true, sc_intarna), 4),
        'miranda_auc': round(safe_auc(y_true, sc_miranda), 4),
    }
    with open(OUT_DIR / "virmirna_e2e_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\\n  Predictions → {OUT_DIR / 'virmirna_e2e_predictions.csv'}")
    print(f"  Summary     → {OUT_DIR / 'virmirna_e2e_summary.json'}")
    print(f"\\n{'='*72}\\n  BENCHMARK COMPLETE ✅\\n{'='*72}")

if __name__ == '__main__':
    main()
