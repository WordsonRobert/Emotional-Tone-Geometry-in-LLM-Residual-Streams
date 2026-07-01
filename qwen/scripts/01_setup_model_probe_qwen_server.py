#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 1 (QWEN3-4B): SETUP + MODEL + DATA + ACTIVATION EXTRACTION
# ═══════════════════════════════════════════════════════════════════
# Model:   Qwen/Qwen3-4B
# Folder:  /DATA/24ms190/steering_qwen
# GPU:     cuda:1 (Quadro RTX 6000, 24 GB)
#
# What this does:
#   1. Load Qwen3-4B via TransformerLens
#   2. Load humor/melancholy/horror CSVs
#   3. Extract activations at all 36 layers + train binary probe
#      → Saves humor_acts.pt, dark_acts.pt, probe_weights.pt
#      → Saves layer_probe_accuracy.png, cell1_model_summary.json
#
# Run: python3 01_setup_model_probe_qwen.py
# Runtime: ~45 min on RTX 6000
# ═══════════════════════════════════════════════════════════════════

import os, gc, json, math, torch, warnings, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── Paths ─────────────────────────────────────────────────────────
DRIVE      = '/DATA/24ms190/steering_qwen'
MODEL_PATH = '/DATA/24ms190/steering_qwen/model'
DEVICE     = 'cuda:1'
EMOTIONS   = ['humor', 'melancholy', 'horror']

print(f'DRIVE:      {DRIVE}')
print(f'MODEL_PATH: {MODEL_PATH}')
print(f'DEVICE:     {DEVICE}')

# ── Clear any leftover GPU memory ─────────────────────────────────
try:
    del model
except NameError:
    pass
gc.collect()
torch.cuda.empty_cache()
print(f'VRAM before load: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

# ── Load Qwen3-4B ─────────────────────────────────────────────────
print(f'\nLoading Qwen/Qwen3-4B from {MODEL_PATH}...')
model = HookedTransformer.from_pretrained(
    'Qwen/Qwen3-4B',
    dtype             = torch.bfloat16,
    device            = DEVICE,
    trust_remote_code = True,
    fold_ln           = False,
    center_unembed    = False,
    move_to_device    = True,
    cache_dir         = MODEL_PATH,
)
model.eval()

# Disable Qwen3 thinking mode
model.tokenizer.chat_template = None
print('✅ chat_template cleared — thinking mode disabled')

# Speed test
print('Testing generation speed...', end='', flush=True)
t0 = time.time()
_ = model.generate('The sky is', max_new_tokens=20, temperature=0.7, verbose=False)
elapsed = time.time() - t0
print(f' {elapsed:.1f}s for 20 tokens', end='')
print(' ✅' if elapsed < 15 else ' ⚠️  SLOW')

# ── Architecture constants ────────────────────────────────────────
N_LAYERS    = model.cfg.n_layers
D_MODEL     = model.cfg.d_model
N_HEADS     = model.cfg.n_heads
N_KV_HEADS  = getattr(model.cfg, 'n_key_value_heads', N_HEADS)
D_HEAD      = model.cfg.d_head
PROBE_LAYER = N_LAYERS - 2
ALL_LAYERS  = list(range(N_LAYERS))

print(f'\nN_LAYERS={N_LAYERS}  D_MODEL={D_MODEL}  N_HEADS={N_HEADS}')
print(f'N_KV_HEADS={N_KV_HEADS}  D_HEAD={D_HEAD}  PROBE_LAYER={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

# ── Helpers ───────────────────────────────────────────────────────
def coherence_score(text: str) -> float:
    words = text.strip().split()
    if len(words) < 3: return 0.0
    counts = Counter(words)
    if counts.most_common(1)[0][1] / len(words) > 0.4: return 0.0
    chars = [c for c in text.lower() if c.isalpha()]
    if len(chars) < 10: return 0.0
    cc  = Counter(chars)
    ent = -sum((c/len(chars))*math.log2(c/len(chars)) for c in cc.values())
    if ent < 3.5: return 0.0
    BAD = ['ss-','ses','sse','sss','thed','bsp','antml']
    def ok(w):
        w = w.strip('.,!?-()').lower()
        return len(w) < 2 or (any(c in 'aeiou' for c in w)
                               and not any(p in w for p in BAD))
    return sum(ok(w) for w in words) / len(words)

def get_activation_at_layer(text: str, layer: int) -> np.ndarray:
    tokens = model.to_tokens([text], prepend_bos=False)
    if tokens.shape[1] == 0:  return np.zeros(D_MODEL, dtype=np.float32)
    hook_name = f'blocks.{layer}.hook_resid_post'
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter  = lambda n: n == hook_name,
            stop_at_layer = layer + 1,
        )
    act = cache[hook_name][0, -1, :].cpu().float().numpy()
    del cache; torch.cuda.empty_cache()
    return act

def get_activation(text: str) -> np.ndarray:
    return get_activation_at_layer(text, PROBE_LAYER)

def norm_v(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def normalize_for_probe(X: np.ndarray):
    mean = X.mean(0); std = X.std(0) + 1e-8
    Xn   = (X - mean) / std
    norms = np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-8
    return (Xn / norms).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

print(f'\n✅ Cell 1 complete — model + helpers ready')
print(f'   PROBE_LAYER = {PROBE_LAYER}  (N_LAYERS - 2 = {N_LAYERS} - 2)')

# ── Load CSVs ─────────────────────────────────────────────────────
def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = next((c for c in df.columns if c.lower() in ['text','content','post','body']),
               df.columns[0])
    if col != 'text': df.rename(columns={col:'text'}, inplace=True)
    df['text'] = df['text'].fillna('').astype(str).str.strip()
    df = df[df['text'].str.len() > 10].reset_index(drop=True)
    return df

df_humor = _load_csv(f'{DRIVE}/humor_master.csv')
df_mel   = _load_csv(f'{DRIVE}/melancholy_master.csv')
df_hor   = _load_csv(f'{DRIVE}/horror_master.csv')
dark_texts = df_mel['text'].tolist() + df_hor['text'].tolist()

print(f'\nDataset sizes:')
print(f'  humor     : {len(df_humor)}')
print(f'  melancholy: {len(df_mel)}')
print(f'  horror    : {len(df_hor)}')
print(f'  dark total: {len(dark_texts)}')
print(f'✅ Cell 2 complete')

# ── Part A: Extract activations at PROBE_LAYER ────────────────────
def _extract_acts(texts: list, label: str) -> np.ndarray:
    path = f'{DRIVE}/{label}_acts.pt'
    if os.path.exists(path):
        print(f'  {label}: loading cached {path}')
        return torch.load(path, map_location='cpu').float().numpy()
    acts = []
    for text in tqdm(texts, desc=f'  {label}'):
        acts.append(get_activation(text))
    arr = np.array(acts, dtype=np.float32)
    torch.save(torch.tensor(arr), path)
    print(f'  {label}: saved {path}  shape={arr.shape}')
    return arr

print('\nPart A: Extracting activations at PROBE_LAYER...')
humor_acts = _extract_acts(df_humor['text'].tolist(), 'humor')
dark_acts  = _extract_acts(dark_texts, 'dark')

print(f'\n  humor_acts: {humor_acts.shape}')
print(f'  dark_acts:  {dark_acts.shape}')

# ── Part B: Per-layer probe ───────────────────────────────────────
N_PROBE  = min(300, len(df_humor), len(dark_texts))
N_DARK   = len(dark_texts)
N_HUMOR  = len(df_humor)

print(f'\nPart B: Training binary probe at every layer (N_PROBE={N_PROBE} per class)...')

layer_probe_accs = {}
probe_weights    = {}

rng   = np.random.default_rng(42)
h_idx = rng.choice(N_HUMOR, size=N_PROBE, replace=False)
d_idx = rng.choice(N_DARK,  size=N_PROBE, replace=False)

PROBE_CKPT = f'{DRIVE}/cell3_probe_checkpoint.json'
if os.path.exists(PROBE_CKPT):
    with open(PROBE_CKPT) as f:
        ckpt = json.load(f)
    layer_probe_accs = {int(k): v for k, v in ckpt['accs'].items()}
    probe_weights_loaded = torch.load(f'{DRIVE}/probe_weights.pt', map_location='cpu')
    probe_weights = {int(k): v.float().numpy() for k, v in probe_weights_loaded.items()}
    print(f'  Resumed from checkpoint: {len(layer_probe_accs)} layers done')
else:
    ckpt = {'accs': {}}

for layer in tqdm(ALL_LAYERS, desc='Probe layers'):
    if layer in layer_probe_accs:
        continue

    h_sub, d_sub = [], []
    for i in h_idx:
        h_sub.append(get_activation_at_layer(df_humor['text'].iloc[i], layer))
    for i in d_idx:
        d_sub.append(get_activation_at_layer(dark_texts[i], layer))

    X = np.vstack([h_sub, d_sub]).astype(np.float32)
    y = np.array([1]*N_PROBE + [0]*N_PROBE)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(clf, X, y, cv=cv).mean()

    clf.fit(X, y)
    w = norm_v(clf.coef_[0].astype(np.float32))

    layer_probe_accs[layer] = float(acc)
    probe_weights[layer]    = w

    ckpt['accs'] = {str(k): v for k, v in layer_probe_accs.items()}
    with open(PROBE_CKPT, 'w') as f:
        json.dump(ckpt, f)
    torch.save(
        {k: torch.tensor(v) for k, v in probe_weights.items()},
        f'{DRIVE}/probe_weights.pt'
    )

# CV check at PROBE_LAYER
X_check = np.vstack([humor_acts[h_idx], dark_acts[d_idx]]).astype(np.float32)
y_check = np.array([1]*N_PROBE + [0]*N_PROBE)
Xn_check, _, _ = normalize_for_probe(X_check)
cv_check = cross_val_score(
    LogisticRegression(max_iter=1000, C=1.0),
    Xn_check, y_check,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)
print(f'\n  Probe L{PROBE_LAYER} CV accuracy: {cv_check.mean():.4f} ± {cv_check.std():.4f}')

# ── Part C: Plot ──────────────────────────────────────────────────
print('\nPart C: Plotting...')
print('  Accuracy per layer:')
best_layer = max(layer_probe_accs, key=layer_probe_accs.get)
for l in ALL_LAYERS:
    marker = ' ← BEST' if l == best_layer else (' ← PROBE_LAYER' if l == PROBE_LAYER else '')
    print(f'    L{l:2d}: {layer_probe_accs[l]:.4f}{marker}')

layers_sorted = sorted(layer_probe_accs.keys())
accs_sorted   = [layer_probe_accs[l] for l in layers_sorted]
colors = ['#C44E52' if l == PROBE_LAYER else '#DD8452' if l == best_layer else '#4C72B0'
          for l in layers_sorted]

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(layers_sorted, accs_sorted, color=colors, alpha=0.85, width=0.75)
ax.axhline(0.5, color='gray', ls='--', lw=1, label='Chance')
ax.axvline(PROBE_LAYER, color='#C44E52', lw=2,
           label=f'PROBE_LAYER={PROBE_LAYER} ({layer_probe_accs[PROBE_LAYER]:.3f})')
ax.set_xlabel('Layer'); ax.set_ylabel('CV Accuracy (5-fold)')
ax.set_title(f'Binary Probe Accuracy — Qwen/Qwen3-4B\nhumor vs dark (mel+horror) | N={N_PROBE}',
             fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.05); ax.set_xticks(range(0, N_LAYERS, 2)); ax.legend()
ax.text(PROBE_LAYER, layer_probe_accs[PROBE_LAYER] + 0.015,
        f'{layer_probe_accs[PROBE_LAYER]:.3f}', ha='center', fontsize=8,
        color='#C44E52', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{DRIVE}/layer_probe_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  layer_probe_accuracy.png saved')

# Save probe weights
torch.save({k: torch.tensor(v) for k, v in probe_weights.items()},
           f'{DRIVE}/probe_weights.pt')

# Save model summary
model_summary = {
    'model_name':         'Qwen/Qwen3-4B',
    'n_layers':           N_LAYERS,
    'd_model':            D_MODEL,
    'n_heads':            N_HEADS,
    'n_kv_heads':         N_KV_HEADS,
    'd_head':             D_HEAD,
    'probe_layer':        PROBE_LAYER,
    'probe_layer_acc':    float(layer_probe_accs[PROBE_LAYER]),
    'probe_layer_acc_std':float(cv_check.std()),
    'best_layer':         best_layer,
    'best_layer_acc':     float(layer_probe_accs[best_layer]),
    'n_probe_per_class':  N_PROBE,
    'humor_n':            N_HUMOR,
    'mel_n':              len(df_mel),
    'horror_n':           len(df_hor),
    'dark_n':             N_DARK,
    'layer_accs':         {str(k): float(v) for k, v in layer_probe_accs.items()},
}
with open(f'{DRIVE}/cell1_model_summary.json', 'w') as f:
    json.dump(model_summary, f, indent=2)

print(f'\n{"━"*55}')
print(f'SCRIPT 1 SUMMARY — Qwen/Qwen3-4B')
print(f'{"━"*55}')
print(f'  humor_acts : {humor_acts.shape}')
print(f'  dark_acts  : {dark_acts.shape}')
print(f'  probe_weights: {len(probe_weights)} layers')
print(f'  PROBE_LAYER acc: {layer_probe_accs[PROBE_LAYER]:.4f} ± {cv_check.std():.4f}')
print(f'  Best layer acc:  L{best_layer} = {layer_probe_accs[best_layer]:.4f}')
print(f'\n✅ Script 1 complete. Saved to {DRIVE}:')
print(f'   humor_acts.pt, dark_acts.pt, probe_weights.pt')
print(f'   layer_probe_accuracy.png, cell1_model_summary.json')
print(f'\nNext: python3 02_probes_steering_sweep_qwen.py')
