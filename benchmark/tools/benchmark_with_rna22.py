import pandas as pd
import numpy as np
import subprocess
import random
import pickle
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT     = Path("/home/somenath/ViroMiR")
DATA     = ROOT / "data/processed/viromir_unified_dataset_v1.csv"
MODEL    = ROOT / "output/models/viromir_xgb_v1.pkl"
MIRANDA  = "/home/somenath/miniconda3/envs/miranda_env/bin/miranda"
INTARNA  = "/home/somenath/miniconda3/envs/cts_env/bin/IntaRNA"
VIRMIRNA = ROOT / "data/external/virmirna_85_normalized.csv"
VIRBASE  = Path("/home/somenath/Pictures/Publication_somenath/virbase_final_dataset/virbase_cts/mirnaprotpred2.0_training_set.csv")

FEATURE_COLS = [
    'delta_G','delta_G_norm','n_base_pairs','n_watson_crick','n_mismatches',
    'bp_fraction','seed_exact_match','seed_score','supplementary_score',
    'motif_identity','au_content','mirna_gc','cts_gc',
    'cts_len','mirna_len','site_position_norm',
]

random.seed(42); np.random.seed(42)

import shutil
RNAHYBRID = shutil.which("RNAhybrid") or shutil.which("rnahybrid")

print("\n── Building cross-virus test set ───────────────────────────")

vb_raw = pd.read_csv(VIRBASE)
vm_raw = pd.read_csv(VIRMIRNA)

pos_vb = vb_raw[vb_raw['label']==1][
    ['miRNA','virus','miRNA_sequence','CTS_sequence']
].dropna().drop_duplicates()

pos_vm = vm_raw[vm_raw['label']==1][
    ['miRNA','virus','miRNA_sequence','target_sequence']
].dropna().drop_duplicates()
pos_vm = pos_vm.rename(columns={'target_sequence':'CTS_sequence'})

pos_all = pd.concat([pos_vb, pos_vm], ignore_index=True)
pos_all = pos_all[pos_all['CTS_sequence'].str.len() >= 30]

virus_seqs = {}
for _, r in pos_all.iterrows():
    v = str(r['virus'])
    if v not in virus_seqs: virus_seqs[v] = []
    virus_seqs[v].append(str(r['CTS_sequence']))

viruses = list(virus_seqs.keys())

pairs = []
for _, r in pos_all.iterrows():
    mi_seq  = str(r['miRNA_sequence'])
    cts_seq = str(r['CTS_sequence'])
    virus   = str(r['virus'])
    if len(mi_seq) < 15 or len(cts_seq) < 20: continue

    pairs.append({
        'mirna_seq': mi_seq, 'cts_seq': cts_seq,
        'label': 1, 'virus': virus, 'mirna': r['miRNA']
    })

    other_viruses = [v for v in viruses if v != virus]
    if not other_viruses: continue
    neg_virus = random.choice(other_viruses)
    neg_cts   = random.choice(virus_seqs[neg_virus])
    pairs.append({
        'mirna_seq': mi_seq, 'cts_seq': neg_cts,
        'label': 0, 'virus': neg_virus, 'mirna': r['miRNA']
    })

pos_pairs = [p for p in pairs if p['label']==1]
neg_pairs = [p for p in pairs if p['label']==0]
sample_pos = random.sample(pos_pairs, min(60, len(pos_pairs)))
sample_neg = random.sample(neg_pairs, min(60, len(neg_pairs)))
test_pairs = sample_pos + sample_neg
random.shuffle(test_pairs)

def run_intarna(mi, cts):
    cmd=[INTARNA,"-q",mi.upper().replace('T','U'),
         "-t",cts.upper().replace('T','U'),
         "--outMode","C","--outCsvCols","E",
         "--threads","1","--seedBP","4"]
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
        for l in r.stdout.strip().split('\n'):
            try: return float(l.strip())
            except: pass
    except: pass
    return 0.0

def run_miranda(mi, cts):
    with open('/tmp/mi.fa','w') as f: f.write(f">q\n{mi}\n")
    with open('/tmp/cts.fa','w') as f: f.write(f">t\n{cts}\n")
    try:
        r=subprocess.run(
            [MIRANDA,'/tmp/mi.fa','/tmp/cts.fa',
             '-sc','50','-en','-1','-quiet'],
            capture_output=True,text=True,timeout=30)
        for line in r.stdout.split('\n'):
            if line.startswith('>>'):
                p=line.split('\t')
                if len(p)>=5:
                    try: return float(p[4])
                    except: pass
    except: pass
    return 0.0

def run_rnahybrid(mi, cts):
    if not RNAHYBRID: return None
    try:
        r=subprocess.run(
            [RNAHYBRID,'-s','3utr_human',
             '-t',cts.upper().replace('T','U'),
             '-q',mi.upper().replace('T','U')],
            capture_output=True,text=True,timeout=30)
        for line in r.stdout.split('\n'):
            if 'mfe:' in line.lower():
                try: return float(line.split(':')[1].strip().split()[0])
                except: pass
    except: pass
    return None

def run_rna22(mi, cts):
    cwd = "/home/somenath/ViroMiR/benchmark/remoteRNA22v2"
    with open(f"{cwd}/myMirInputFile.txt", "w") as f:
        f.write(f">mir\n{mi}\n")
    with open(f"{cwd}/myTranscriptInputFile.txt", "w") as f:
        f.write(f">cts\n{cts}\n")
    
    # Clear output.txt before running to avoid reading old results
    try: os.remove(f"{cwd}/output.txt")
    except: pass

    try:
        subprocess.run(["java", "RNA22v2"], cwd=cwd, capture_output=True, timeout=45)
        if os.path.exists(f"{cwd}/output.txt"):
            with open(f"{cwd}/output.txt") as f:
                lines = f.readlines()
                energies = []
                for line in lines:
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        try:
                            e = float(parts[3])
                            energies.append(e)
                        except: pass
                if energies:
                    return min(energies)
    except: pass
    return 0.0

print("\n── Running tools on cross-virus test set ───────────────────")
print(f"  (This takes ~5-10 min for {len(test_pairs)} pairs)\n")

y_true = []
sc_intarna = []
sc_miranda = []
sc_rnahybrid = []
sc_rna22 = []

for i, pair in enumerate(test_pairs):
    if i % 20 == 0:
        print(f"  [{i:3d}/{len(test_pairs)}] processing...")
    mi  = pair['mirna_seq']
    cts = pair['cts_seq']

    dg  = run_intarna(mi, cts)
    sc_intarna.append(-dg)

    mr  = run_miranda(mi, cts)
    sc_miranda.append(mr)

    rh  = run_rnahybrid(mi, cts)
    if rh is not None:
        sc_rnahybrid.append(-rh)
    else:
        sc_rnahybrid.append(None)
        
    r22 = run_rna22(mi, cts)
    sc_rna22.append(-r22) # more negative = stronger

    y_true.append(pair['label'])

y_true = np.array(y_true)

# ── ViroMiR score on same pairs ───────────────────────────────────
print("\n── Scoring with ViroMiR ──────────────────────────────────────")

comp = {'A':'T','T':'A','G':'C','C':'G','U':'A'}
def gc(s): s=s.upper().replace('U','C'); return sum(1 for c in s if c in 'GC')/max(len(s),1)
def au(s): s=s.upper().replace('T','A'); return sum(1 for c in s if c in 'AU')/max(len(s),1)
def seed_em(mi,ct):
    mi=mi.upper().replace('U','T'); ct=ct.upper().replace('U','T')
    if len(mi)<8: return 0
    rc=''.join(comp.get(b,'N') for b in reversed(mi[1:8]))
    return 1 if rc in ct else 0
def seed_sc(mi,ct):
    mi=mi.upper().replace('U','T'); ct=ct.upper().replace('U','T')
    if len(mi)<8: return 0.0
    return sum(1 for b in mi[1:8] if comp.get(b,'N') in ct)/7.0
def supp_sc(mi,ct):
    mi=mi.upper().replace('U','T'); ct=ct.upper().replace('U','T')
    if len(mi)<17: return 0.0
    return sum(1 for b in mi[11:17] if comp.get(b,'N') in ct)/6.0
def parse_hyb(h):
    if not h or '&' not in h: return 0,0
    s=h.split('&')[0]; return s.count('('), s.count('.')

def make_feat(mi, cts, dg):
    cmd=[INTARNA,"-q",mi.upper().replace('T','U'),
         "-t",cts.upper().replace('T','U'),
         "--outMode","C","--outCsvCols","E,hybridDP,start2,end2",
         "--threads","1","--seedBP","4"]
    hyb=''; ts=0; te=30
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
        for l in r.stdout.strip().split('\n'):
            try:
                p=l.split(';'); float(p[0])
                hyb=p[1] if len(p)>1 else ''
                ts=int(p[2]) if len(p)>2 and p[2].strip().lstrip('-').isdigit() else 0
                te=int(p[3]) if len(p)>3 and p[3].strip().lstrip('-').isdigit() else 30
                break
            except: pass
    except: pass
    m,mm = parse_hyb(hyb)
    return [dg, dg/max(len(mi),1), m, m, mm, m/max(len(mi),1),
            seed_em(mi,cts), seed_sc(mi,cts), supp_sc(mi,cts),
            seed_em(mi,cts)*int(supp_sc(mi,cts)>0.5),
            au(cts), gc(mi), gc(cts), len(cts), len(mi),
            ts/max(len(cts),1)]

pkg   = pickle.load(open(MODEL,'rb'))
model = pkg['model']

sc_viromir = []
for i, pair in enumerate(test_pairs):
    dg = -sc_intarna[i]
    feat = make_feat(pair['mirna_seq'], pair['cts_seq'], dg)
    prob = model.predict_proba([feat])[0][1]
    sc_viromir.append(prob)
    if i % 20 == 0:
        print(f"  [{i:3d}/{len(test_pairs)}] ViroMiR scored")

# ── Final benchmark table ─────────────────────────────────────────
sc_viromir  = np.array(sc_viromir)
sc_intarna  = np.array(sc_intarna)
sc_miranda  = np.array(sc_miranda)
sc_rna22    = np.array(sc_rna22)

def safe_auc(y, s):
    try: return roc_auc_score(y, s)
    except: return float('nan')
def safe_ap(y, s):
    try: return average_precision_score(y, s)
    except: return float('nan')
def prec_at_k(y, s, k):
    top = np.argsort(s)[::-1][:k]
    return y[top].mean()

print(f"\n{'='*62}")
print(f"  FAIR BENCHMARK — Cross-Virus Negatives")
print(f"  Negatives: same miRNA tested against DIFFERENT virus mRNA")
print(f"  n={len(test_pairs)} pairs ({y_true.sum()} pos, {(y_true==0).sum()} neg)")
print(f"{'='*62}")
print(f"  {'Method':<30} {'AUC':>7} {'AUC-PR':>7} {'P@10':>6} {'P@20':>6}")
print(f"  {'-'*54}")
print(f"  {'IntaRNA (delta_G)':<30} {safe_auc(y_true,sc_intarna):>7.4f} {safe_ap(y_true,sc_intarna):>7.4f} {prec_at_k(y_true,sc_intarna,10):>6.4f} {prec_at_k(y_true,sc_intarna,20):>6.4f}")
print(f"  {'miRanda (alignment score)':<30} {safe_auc(y_true,sc_miranda):>7.4f} {safe_ap(y_true,sc_miranda):>7.4f} {prec_at_k(y_true,sc_miranda,10):>6.4f} {prec_at_k(y_true,sc_miranda,20):>6.4f}")
print(f"  {'RNA22 (folding energy)':<30} {safe_auc(y_true,sc_rna22):>7.4f} {safe_ap(y_true,sc_rna22):>7.4f} {prec_at_k(y_true,sc_rna22,10):>6.4f} {prec_at_k(y_true,sc_rna22,20):>6.4f}")

if any(x is not None for x in sc_rnahybrid):
    rh_arr = np.array([x if x is not None else 0.0 for x in sc_rnahybrid])
    print(f"  {'RNAhybrid (MFE)':<30} {safe_auc(y_true,rh_arr):>7.4f} {safe_ap(y_true,rh_arr):>7.4f} {prec_at_k(y_true,rh_arr,10):>6.4f} {prec_at_k(y_true,rh_arr,20):>6.4f}")
else:
    print(f"  {'RNAhybrid':<30} {'N/A':>7} {'N/A':>7} {'N/A':>6} {'N/A':>6}")
print(f"  {'ViroMiR v1':<30} {safe_auc(y_true,sc_viromir):>7.4f} {safe_ap(y_true,sc_viromir):>7.4f} {prec_at_k(y_true,sc_viromir,10):>6.4f} {prec_at_k(y_true,sc_viromir,20):>6.4f}")
print(f"{'='*62}")
