#!/usr/bin/env python3
"""
viromir.scan  —  ViroMiR v1 Production CLI
═════════════════════════════════════════════════════════════════
Two-stage prediction of human miRNA → viral mRNA interactions:
  Stage 1: IntaRNA thermodynamic scanning (candidate generation)
  Stage 2: XGBoost rescoring with 16 biology-based features
  Stage 3: Post-processing (dedup, mode filter, dual output)
═════════════════════════════════════════════════════════════════
"""

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
DEFAULT_REF  = SCRIPT_DIR / "data" / "hsa_mature_mirbase.fa"
DEFAULT_MDL  = SCRIPT_DIR / "model" / "viromir_xgb_v1.pkl"
INTARNA_BIN  = os.environ.get("INTARNA_BIN", "IntaRNA") # Assumes IntaRNA is in PATH

MAX_THREADS  = min(os.cpu_count() or 4, 8)   # safety cap

FEATURE_COLS = [
    'delta_G', 'delta_G_norm',
    'n_base_pairs', 'n_watson_crick', 'n_mismatches', 'bp_fraction',
    'seed_exact_match', 'seed_score', 'supplementary_score', 'motif_identity',
    'au_content', 'mirna_gc', 'cts_gc',
    'cts_len', 'mirna_len', 'site_position_norm',
]

# ── Composite score weights ───────────────────────────────────────
# ViroMiR Score = weighted combination of ML probability + biology
# This breaks the probability saturation problem in genome-wide scans
SCORE_WEIGHTS = {
    'prob':      0.50,   # XGBoost probability
    'dg_norm':   0.20,   # Normalized ΔG (thermodynamic quality)
    'seed':      0.15,   # Seed region complementarity
    'structure': 0.15,   # Base-pair fraction (structural quality)
}

# ── Confidence thresholds (on composite ViroMiR score) ────────────
CONFIDENCE_THRESHOLDS = [
    (0.85, "High"),
    (0.70, "Medium"),
    (0.50, "Low"),
]

# ── Mode presets ──────────────────────────────────────────────────
MODE_PRESETS = {
    'discovery': {
        'score_cutoff': 0.50,
        'description': 'Permissive discovery — all plausible interactions',
    },
    'balanced': {
        'score_cutoff': 0.70,
        'description': 'Balanced analysis — captures all validated targets',
    },
    'strict': {
        'score_cutoff': 0.75,
        'description': 'Strict publication — compact high-confidence table',
    },
}

# ══════════════════════════════════════════════════════════════════
#  FASTA I/O
# ══════════════════════════════════════════════════════════════════

def read_fasta(path):
    records = []
    hdr, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if hdr is not None:
                    records.append((hdr, ''.join(seq_parts).upper()))
                hdr = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.replace(' ', ''))
    if hdr is not None:
        records.append((hdr, ''.join(seq_parts).upper()))
    return records

def read_fasta_full_header(path):
    records = []
    hdr, seq_parts = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if hdr is not None:
                    records.append((hdr, ''.join(seq_parts).upper()))
                hdr = line[1:].strip().split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.replace(' ', ''))
    if hdr is not None:
        records.append((hdr, ''.join(seq_parts).upper()))
    return records

# ══════════════════════════════════════════════════════════════════
#  Stage 1: IntaRNA Scanner
# ══════════════════════════════════════════════════════════════════

def run_intarna(mirna_seq, target_seq):
    mi = mirna_seq.upper().replace('T', 'U')
    tgt = target_seq.upper().replace('T', 'U')
    cmd = [
        INTARNA_BIN,
        "-q", mi, "-t", tgt,
        "--outMode", "C",
        "--outCsvCols", "E,hybridDP,start1,end1,start2,end2",
        "--threads", "1",
        "--seedBP", "4",
        "--outNumber", "5",  # Capture up to 5 suboptimal binding sites
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        hits = []
        for line in r.stdout.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('E;'): continue
            parts = line.split(';')
            try: dg = float(parts[0])
            except ValueError: continue
            def safe_int(s):
                s = s.strip()
                return int(s) if s.lstrip('-').isdigit() else 0
            hits.append({
                'delta_G':   dg,
                'hybridDP':  parts[1] if len(parts) > 1 else '',
                'tgt_start': safe_int(parts[2]) if len(parts) > 2 else 0,
                'tgt_end':   safe_int(parts[3]) if len(parts) > 3 else 0,
                'q_start':   safe_int(parts[4]) if len(parts) > 4 else 0,
                'q_end':     safe_int(parts[5]) if len(parts) > 5 else 0,
            })
        return hits if hits else None
    except FileNotFoundError:
        print(f"\\n  [!] Error: {INTARNA_BIN} not found. Please install IntaRNA or set INTARNA_BIN.", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"\\n  [!] Warning: IntaRNA timeout for {mi} vs {tgt[:20]}...", file=sys.stderr)
        return "ERROR"
    except Exception as e:
        print(f"\\n  [!] Error running IntaRNA: {e}", file=sys.stderr)
        return "ERROR"

# ══════════════════════════════════════════════════════════════════
#  Stage 2: Feature Extraction
# ══════════════════════════════════════════════════════════════════

_COMP = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'U': 'A'}

def _gc(seq):
    s = seq.upper().replace('U', 'C')
    return sum(1 for c in s if c in 'GC') / max(len(s), 1)

def _au(seq):
    s = seq.upper().replace('T', 'A')
    return sum(1 for c in s if c in 'AU') / max(len(s), 1)

def _seed_exact_match(mi, ct):
    mi = mi.upper().replace('U', 'T')
    ct = ct.upper().replace('U', 'T')
    if len(mi) < 8 or len(ct) < 7: return 0
    rc = ''.join(_COMP.get(b, 'N') for b in reversed(mi[1:8]))
    return 1 if rc in ct else 0

def _seed_score(mi, ct):
    mi = mi.upper().replace('U', 'T')
    ct = ct.upper().replace('U', 'T')
    if len(mi) < 8: return 0.0
    return sum(1 for b in mi[1:8] if _COMP.get(b, 'N') in ct) / 7.0

def _supplementary_score(mi, ct):
    mi = mi.upper().replace('U', 'T')
    ct = ct.upper().replace('U', 'T')
    if len(mi) < 17: return 0.0
    supp = mi[11:17]
    return sum(1 for b in supp if _COMP.get(b, 'N') in ct) / len(supp)

def _parse_hybrid(h):
    if not h or '&' not in h: return 0, 0
    mirna_side = h.split('&')[0]
    return mirna_side.count('('), mirna_side.count('.')

def build_features(mirna_seq, target_seq, inta_result, target_len):
    dg  = inta_result['delta_G']
    hyb = inta_result['hybridDP']
    ts  = inta_result['tgt_start']

    m, mm = _parse_hybrid(hyb)
    sem = _seed_exact_match(mirna_seq, target_seq)
    ss  = _seed_score(mirna_seq, target_seq)
    sup = _supplementary_score(mirna_seq, target_seq)

    return {
        'delta_G':             dg,
        'delta_G_norm':        dg / max(len(mirna_seq), 1),
        'n_base_pairs':        m,
        'n_watson_crick':      m,
        'n_mismatches':        mm,
        'bp_fraction':         m / max(len(mirna_seq), 1),
        'seed_exact_match':    sem,
        'seed_score':          ss,
        'supplementary_score': sup,
        'motif_identity':      sem * int(sup > 0.5),
        'au_content':          _au(target_seq),
        'mirna_gc':            _gc(mirna_seq),
        'cts_gc':              _gc(target_seq),
        'cts_len':             len(target_seq),
        'mirna_len':           len(mirna_seq),
        'site_position_norm':  ts / max(target_len, 1),
    }

def compute_viromir_score(record):
    """Composite ViroMiR score combining ML probability with biological quality.

    This score addresses the probability saturation problem: XGBoost assigns
    prob > 0.95 to most candidates in genome-wide scans. The composite score
    incorporates thermodynamic quality (ΔG), seed complementarity, and
    structural base-pairing to separate true biological interactions from
    background noise.

    Score range: 0.0 to 1.0
    """
    w = SCORE_WEIGHTS

    # ML probability component (already 0–1)
    prob_score = record['viromir_prob']

    # Thermodynamic quality: normalize ΔG to 0–1 scale
    # Typical range: -10 (weak) to -30 (very strong)
    dg = record['delta_G']
    dg_norm = min(max((dg - (-10)) / (-30 - (-10)), 0.0), 1.0)

    # Seed region complementarity (already 0–1 from build_features)
    seed = record.get('seed_score', 0.0)

    # Structural quality: base-pair fraction (already 0–1)
    bp_frac = record.get('bp_fraction', 0.0)

    score = (
        w['prob']      * prob_score +
        w['dg_norm']   * dg_norm +
        w['seed']      * seed +
        w['structure'] * bp_frac
    )
    return round(score, 4)


def confidence_label(score):
    """Assign confidence label based on composite ViroMiR score."""
    for threshold, label in CONFIDENCE_THRESHOLDS:
        if score >= threshold: return label
    return "Very low"

# ══════════════════════════════════════════════════════════════════
#  Pipeline Orchestration
# ══════════════════════════════════════════════════════════════════

def scan_pair(mirna_id, mirna_seq, target_id, target_seq):
    inta_hits = run_intarna(mirna_seq, target_seq)
    if inta_hits == "ERROR": return "ERROR"
    if not inta_hits: return None

    all_feats = []
    for inta in inta_hits:
        feats = build_features(mirna_seq, target_seq, inta, len(target_seq))
        feats['mirna_id']  = mirna_id
        feats['target_id'] = target_id
        feats['start']     = inta['tgt_start']
        feats['end']       = inta['tgt_end']
        all_feats.append(feats)
    return all_feats

def _cluster_hits(hits, window, max_hits_per_mirna):
    """Chained interval clustering / overlap-based clustering per target."""
    if not hits:
        return [], {}

    # Group by target
    by_target = {}
    for h in hits:
        by_target.setdefault(h['target_id'], []).append(h)

    clustered = []
    for tgt_id, tgt_hits in by_target.items():
        # Sort by position for deterministic clustering
        tgt_hits.sort(key=lambda r: (r['start'], r['end']))

        clusters = []
        current_cluster = []
        cluster_end = -1

        for h in tgt_hits:
            if not current_cluster:
                current_cluster.append(h)
                cluster_end = h['end'] + window
            elif h['start'] <= cluster_end:
                current_cluster.append(h)
                cluster_end = max(cluster_end, h['end'] + window)
            else:
                clusters.append(current_cluster)
                current_cluster = [h]
                cluster_end = h['end'] + window
        if current_cluster:
            clusters.append(current_cluster)

        # Process each cluster
        for c_hits in clusters:
            # Group by mirna_id within cluster
            by_mirna = {}
            for h in c_hits:
                by_mirna.setdefault(h['mirna_id'], []).append(h)
            
            # Keep best hit per miRNA per cluster
            for m_id, m_hits in by_mirna.items():
                m_hits.sort(key=lambda r: (-r['viromir_score'], r['delta_G'], -r.get('n_base_pairs', 0)))
                clustered.append(m_hits[0])

    hits_after_dedup = len(clustered)

    # Sort final result
    clustered.sort(key=lambda r: (-r['viromir_score'], r['delta_G']))

    # Global max hits
    if max_hits_per_mirna > 0:
        capped = []
        counts = {}
        for h in clustered:
            c = counts.get(h['mirna_id'], 0)
            if c < max_hits_per_mirna:
                capped.append(h)
                counts[h['mirna_id']] = c + 1
        clustered = capped
        
    stats = {
        'hits_after_dedup': hits_after_dedup,
        'hits_after_cap': len(clustered)
    }
    return clustered, stats

def run_pipeline(mirnas, targets, model, mode, show_all, threads, top_k, min_dg, cluster_window=20, max_hits_per_mirna=0):
    """
    Full 3-stage pipeline:
      Stage 1 — IntaRNA thermodynamic scanning (parallelized)
      Stage 2 — XGBoost rescoring of ALL candidates (no pre-truncation)
      Stage 3 — Cluster, dedup, mode filter, rank, dual output
    """
    mode_cfg = MODE_PRESETS[mode]
    score_cutoff = mode_cfg['score_cutoff']

    total = len(mirnas) * len(targets)
    candidates = []
    done = 0
    failed = 0
    errors = 0
    dg_filtered = 0
    t0 = time.time()

    # ── Stage 1: IntaRNA scanning ─────────────────────────────────
    futures = {}
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for mi_id, mi_seq in mirnas:
            for tgt_id, tgt_seq in targets:
                fut = executor.submit(scan_pair, mi_id, mi_seq, tgt_id, tgt_seq)
                futures[fut] = (mi_id, tgt_id)

        for fut in as_completed(futures):
            done += 1
            result_list = fut.result()
            if result_list == "ERROR":
                errors += 1
            elif result_list is not None:
                for result in result_list:
                    if result['delta_G'] > min_dg:
                        dg_filtered += 1
                    else:
                        candidates.append(result)
            else:
                failed += 1

            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.01)
                eta = (total - done) / max(rate, 0.01)
                print(f"\r  Progress: {done}/{total}  ({done*100//total}%)  "
                      f"hits={len(candidates)}  ETA={eta:.0f}s", end='', flush=True)

    elapsed = time.time() - t0
    print(f"\\n\\n  Stage 1 complete in {elapsed:.1f}s")
    print(f"    IntaRNA candidates : {len(candidates)}")
    print(f"    No interaction     : {failed}")
    if errors > 0:
        print(f"    Errors/Timeouts    : {errors}")
    print(f"    Filtered (ΔG > {min_dg}): {dg_filtered}")

    if not candidates:
        print("  ⚠️  No interactions passed thermodynamic filter. Exiting.")
        return [], [], {
            'intarna_candidates': 0, 'no_interaction': failed,
            'dg_filtered': dg_filtered, 'rescored': 0,
        }

    # ── Stage 2: XGBoost rescoring (ALL candidates, no truncation) ─
    print("\\n  Stage 2: Rescoring ALL candidates with XGBoost model...")
    feature_matrix = [[r[col] for col in FEATURE_COLS] for r in candidates]
    probabilities = model.predict_proba(feature_matrix)[:, 1]

    for r, prob in zip(candidates, probabilities):
        r['viromir_prob'] = float(prob)

    # ── Compute composite ViroMiR score ───────────────────────────
    print("  Computing composite ViroMiR scores...")
    for r in candidates:
        r['viromir_score'] = compute_viromir_score(r)
        r['confidence']    = confidence_label(r['viromir_score'])

    # Sort all candidates by composite score (primary), then ΔG (tiebreak)
    candidates.sort(key=lambda r: (-r['viromir_score'], r['delta_G']))

    # Save the full unfiltered list for supplementary output
    all_scored = list(candidates)
    for i, r in enumerate(all_scored, 1):
        r['rank_full'] = i

    print(f"  Stage 2 complete: {len(all_scored)} predictions scored")
    all_conf = _count_confidence(all_scored)
    for c in ['High', 'Medium', 'Low', 'Very low']:
        if c in all_conf:
            print(f"    {c}: {all_conf[c]}")

    # ── Stage 3: Post-processing ──────────────────────────────────
    print(f"\\n  Stage 3: Post-processing (mode={mode}, score ≥ {score_cutoff})...")

    if show_all:
        filtered = list(all_scored)
        print(f"    --show-all active: keeping all {len(filtered)} predictions")
        c_stats = {'hits_after_dedup': len(filtered), 'hits_after_cap': len(filtered)}
    else:
        # 3a-3c. Genomic clustering & per-miRNA cap
        clustered, c_stats = _cluster_hits(all_scored, cluster_window, max_hits_per_mirna)
        print(f"    Interval clustering (window={cluster_window}nt) + Dedup: "
              f"{len(all_scored)} → {c_stats['hits_after_dedup']}")
        if max_hits_per_mirna > 0:
            print(f"    Cap applied (max {max_hits_per_mirna} per miRNA): "
                  f"{c_stats['hits_after_dedup']} → {c_stats['hits_after_cap']}")

        # 3d. Mode-based composite score cutoff
        filtered = [r for r in clustered if r['viromir_score'] >= score_cutoff]
        print(f"    After {mode} filter (score ≥ {score_cutoff}): {len(filtered)}")

    hits_after_mode = len(filtered)

    # 3e. Sort and apply optional global top-k
    filtered.sort(key=lambda r: (-r['viromir_score'], r['delta_G']))
    if top_k and len(filtered) > top_k:
        filtered = filtered[:top_k]
        print(f"    Top-K retained globally: {top_k}")

    # 3f. Final ranking
    for i, r in enumerate(filtered, 1):
        r['rank'] = i

    filt_conf = _count_confidence(filtered)
    unique_mirnas = len(set(r['mirna_id'] for r in filtered))
    print(f"\\n  Final output: {len(filtered)} predictions ({unique_mirnas} unique miRNAs)")
    for c in ['High', 'Medium', 'Low', 'Very low']:
        if c in filt_conf:
            print(f"    {c}: {filt_conf[c]}")

    summary = {
        'raw_intarna_hits': len(candidates) + dg_filtered,
        'dg_passing_hits': len(candidates),
        'dg_filtered_out': dg_filtered,
        'no_interaction': failed,
        'errors': errors,
        'ml_scored_hits': len(all_scored),
        'hits_after_clustering_dedup': c_stats['hits_after_dedup'] if not show_all else len(all_scored),
        'hits_after_max_cap': c_stats['hits_after_cap'] if not show_all else len(all_scored),
        'hits_after_score_cutoff': hits_after_mode,
        'final_topk_hits': len(filtered),
        'mode': mode,
        'score_cutoff': score_cutoff,
        'cluster_window': cluster_window,
        'max_hits_per_mirna': max_hits_per_mirna,
        'unique_mirnas': unique_mirnas,
        'high': filt_conf.get('High', 0),
        'medium': filt_conf.get('Medium', 0),
        'low': filt_conf.get('Low', 0),
        'very_low': filt_conf.get('Very low', 0),
        'full_high': all_conf.get('High', 0),
        'full_medium': all_conf.get('Medium', 0),
        'full_low': all_conf.get('Low', 0),
        'full_very_low': all_conf.get('Very low', 0),
        'elapsed_seconds': round(time.time() - t0, 1),
    }
    return filtered, all_scored, summary


def _count_confidence(results):
    counts = {}
    for r in results:
        c = r['confidence']
        counts[c] = counts.get(c, 0) + 1
    return counts

# ══════════════════════════════════════════════════════════════════
#  Output Writer
# ══════════════════════════════════════════════════════════════════

OUTPUT_COLUMNS = [
    'rank', 'mirna_id', 'target_id', 'start', 'end',
    'delta_G', 'seed_score', 'supplementary_score', 'n_base_pairs',
    'viromir_prob', 'viromir_score', 'confidence',
]

FULL_OUTPUT_COLUMNS = [
    'rank_full', 'mirna_id', 'target_id', 'start', 'end',
    'delta_G', 'seed_score', 'supplementary_score', 'n_base_pairs',
    'viromir_prob', 'viromir_score', 'confidence',
]

def write_results(results, out_path, columns=OUTPUT_COLUMNS):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = dict(r)
            for k in ['delta_G', 'seed_score', 'supplementary_score', 'viromir_prob', 'viromir_score']:
                if k in row and isinstance(row[k], float):
                    row[k] = round(row[k], 4)
            writer.writerow(row)

def write_summary(summary, out_path, args_dict):
    summary_path = Path(out_path).with_suffix('.summary.json')
    summary['parameters'] = args_dict
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary  → {summary_path}")

# ══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='viromir-scan',
        description='ViroMiR v1 — Predict human miRNA → viral mRNA interactions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  discovery   Permissive (score ≥ 0.50) — highest recall for exploratory scans
  balanced    Default    (score ≥ 0.70) — practical trade-off between recall and precision
  strict      Stringent  (score ≥ 0.75) — most stringent ranked output (lowest false-positive rate)

Examples:
  # Scan against the bundled human miRNA library (default)
  viromir-scan -i virus.fa --mode balanced -o results.csv

  # Scan against a custom miRNA FASTA file
  viromir-scan -i virus.fa --mirna-file custom_mirnas.fa --mode strict -o results.csv

  # Bypass mode cutoffs to retain all ML-scored hits in the main CSV
  viromir-scan -i virus.fa --mode discovery --show-all -o results.csv
        """,
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Viral transcript FASTA or multi-FASTA file (required)')
    parser.add_argument('--out', '-o', default='viromir_predictions.csv',
                        help='Output CSV path (default: viromir_predictions.csv)')
    parser.add_argument('--mirna', '-m', default=None,
                        help='Select exactly one miRNA by name from the bundled reference library (mutually exclusive with --mirna-file)')
    parser.add_argument('--mirna-file', '-mf', default=None,
                        help='Custom FASTA file of miRNA sequences to test (mutually exclusive with --mirna)')
    parser.add_argument('--model', default=str(DEFAULT_MDL),
                        help=f'Path to ViroMiR model pickle (default bundled model)')
    parser.add_argument('--mode', choices=['discovery', 'balanced', 'strict'],
                        default='balanced',
                        help='Filtering stringency (default: balanced)')
    parser.add_argument('--show-all', action='store_true',
                        help='Bypass the mode cutoff for the main CSV; writes all clustered predictions regardless of score')
    parser.add_argument('--threads', '-t', type=int, default=4,
                        help=f'Number of parallel IntaRNA workers (default: 4, max: {MAX_THREADS})')
    parser.add_argument('--top-k', '-k', type=int, default=0,
                        help='Post-scoring ranking cap: keep only the top K predictions AFTER all filtering/clustering (default: 0 = no limit)')
    parser.add_argument('--min-dg', type=float, default=-10.0,
                        help='Discard IntaRNA hits weaker than this energy threshold (default: -10.0)')
    parser.add_argument('--cluster-window', type=int, default=20,
                        help='Cluster hits within this many nt on the same target (default: 20)')
    parser.add_argument('--max-hits-per-mirna', type=int, default=0,
                        help='Maximum allowed hits per miRNA globally (default: 0 = no limit)')

    args = parser.parse_args()
    threads = min(max(args.threads, 1), MAX_THREADS)
    top_k   = args.top_k if args.top_k > 0 else None
    mode    = args.mode
    mode_cfg = MODE_PRESETS[mode]

    print("\\n═" * 31)
    print("  ViroMiR v1 — Human miRNA → Viral mRNA Interaction Predictor")
    print("═" * 31)
    print(f"  Engine  : IntaRNA + XGBoost (16 features)")
    print(f"  Model   : {Path(args.model).name}")
    print(f"  Mode    : {mode} — {mode_cfg['description']}")
    print(f"  Threads : {threads} (max {MAX_THREADS})")
    print(f"  Top-K   : {top_k or 'no limit (post-scoring)'}")
    print(f"  Min ΔG  : {args.min_dg} kcal/mol\\n")

    if not Path(args.input).exists():
        print(f"  ❌ Error: input file not found: {args.input}"); sys.exit(1)
    if not Path(args.model).exists():
        print(f"  ❌ Error: model file not found: {args.model}"); sys.exit(1)

    targets = read_fasta(args.input)
    if not targets:
        print(f"  ❌ Error: no sequences found in {args.input}"); sys.exit(1)
    print(f"  Viral targets loaded: {len(targets)}")

    if args.mirna_file:
        mirnas = read_fasta_full_header(args.mirna_file)
        if not mirnas:
            print(f"  ❌ Error: no miRNAs found in {args.mirna_file}"); sys.exit(1)
        print(f"  miRNAs loaded from file: {len(mirnas)}")
        scan_mode = "targeted"
    elif args.mirna:
        if not DEFAULT_REF.exists():
            print(f"  ❌ Error: bundled miRNA reference not found: {DEFAULT_REF}"); sys.exit(1)
        all_mirnas = read_fasta_full_header(str(DEFAULT_REF))
        matched = [(mid, mseq) for mid, mseq in all_mirnas if mid.lower() == args.mirna.lower()]
        if not matched:
            matched = [(mid, mseq) for mid, mseq in all_mirnas if args.mirna.lower() in mid.lower()]
        if not matched:
            print(f"  ❌ Error: miRNA '{args.mirna}' not found in reference"); sys.exit(1)
        mirnas = matched
        print(f"  miRNA matched: {[m[0] for m in mirnas]}")
        scan_mode = "targeted"
    else:
        if not DEFAULT_REF.exists():
            print(f"  ❌ Error: bundled miRNA reference not found: {DEFAULT_REF}"); sys.exit(1)
        mirnas = read_fasta_full_header(str(DEFAULT_REF))
        print(f"  miRNAs loaded (auto-scan): {len(mirnas)} human miRNAs from miRBase")
        scan_mode = "auto-scan"

    total_pairs = len(mirnas) * len(targets)
    est_time = total_pairs * 0.5 / max(threads, 1)
    print(f"  Total pairs: {total_pairs:,} (estimated ~{est_time:.0f}s at {threads} threads)\\n")

    print("  Loading ViroMiR model...", end=' ')
    with open(args.model, 'rb') as f:
        pkg = pickle.load(f)
    model = pkg['model']
    print("✅\\n")

    print("── Stage 1: IntaRNA scanning ────────────────────────────────")
    cluster_window = args.cluster_window
    max_hits = args.max_hits_per_mirna
    filtered, all_scored, summary = run_pipeline(
        mirnas, targets, model, mode, args.show_all, threads, top_k, args.min_dg,
        cluster_window=cluster_window, max_hits_per_mirna=max_hits
    )

    summary['n_mirnas']  = len(mirnas)
    summary['n_targets'] = len(targets)
    summary['scan_mode'] = scan_mode

    if not filtered and not all_scored:
        print("\\n  No predictions to write.")
        write_summary(summary, args.out, vars(args))
        sys.exit(0)

    # ── Write main (filtered) output ──────────────────────────────
    write_results(filtered, args.out, OUTPUT_COLUMNS)
    print(f"\\n  Main output → {args.out}")
    print(f"  Predictions: {len(filtered)}")

    # ── Write full supplementary output ───────────────────────────
    full_path = Path(args.out).with_name(
        Path(args.out).stem + '_full' + Path(args.out).suffix
    )
    write_results(all_scored, full_path, FULL_OUTPUT_COLUMNS)
    print(f"  Full output → {full_path}  ({len(all_scored)} total)")

    write_summary(summary, args.out, vars(args))

    # ── Print top 10 from main output ─────────────────────────────
    print("\\n── Top 10 Predictions ──────────────────────────────────────")
    print(f"  {'Rank':<5} {'miRNA':<25} {'Target':<15} {'ΔG':>7} {'Score':>6} {'Conf':<8}")
    print(f"  {'-'*70}")
    for r in filtered[:10]:
        print(f"  {r['rank']:<5} {r['mirna_id']:<25} {r['target_id']:<15} "
              f"{r['delta_G']:>7.1f} {r['viromir_score']:>6.4f} {r['confidence']:<8}")

    # ── Run summary ───────────────────────────────────────────────
    print("\\n── Run Summary ─────────────────────────────────────────────")
    print(f"  Viral records scanned : {summary['n_targets']}")
    print(f"  miRNAs tested         : {summary['n_mirnas']}")
    print(f"  Raw IntaRNA hits      : {summary['raw_intarna_hits']}")
    if summary.get('errors', 0) > 0:
        print(f"  Errors/Timeouts       : {summary['errors']}")
    print(f"  ΔG passing hits       : {summary['dg_passing_hits']} (filtered {summary['dg_filtered_out']})")
    print(f"  ML scored hits        : {summary['ml_scored_hits']}")
    if not args.show_all:
        print(f"  After cluster/dedup   : {summary['hits_after_clustering_dedup']}")
        if args.max_hits_per_mirna > 0:
            print(f"  After max hits cap    : {summary['hits_after_max_cap']}")
    print(f"  Mode                  : {mode} (score ≥ {mode_cfg['score_cutoff']})")
    print(f"  After score cutoff    : {summary['hits_after_score_cutoff']}")
    print(f"  Final predictions     : {summary['final_topk_hits']}")
    print(f"  Unique miRNAs         : {summary['unique_mirnas']}")
    print(f"    High confidence     : {summary.get('high', 0)}")
    print(f"    Medium confidence   : {summary.get('medium', 0)}")
    print(f"    Low confidence      : {summary.get('low', 0)}")
    print(f"  Runtime               : {summary['elapsed_seconds']}s")
    print("\\n═" * 31)
    print("  ViroMiR prediction complete ✅")
    print("═" * 31)

if __name__ == '__main__':
    main()
