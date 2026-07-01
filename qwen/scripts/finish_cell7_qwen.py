#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# FINISH CELL 7 (QWEN3-4B): VECTOR COMPARISON + SUMMARY
# ═══════════════════════════════════════════════════════════════════
# Picks up after main eval is done (all 3 emotions complete)
# Runs vector comparison with live per-sample output
# ═══════════════════════════════════════════════════════════════════

import os, gc, json, math, torch, warnings, time
import numpy as np, pandas as pd
from collections import Counter
from itertools import combinations
from scipy import stats as scipy_stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DRIVE      = '/DATA/24ms190/steering_qwen'
MODEL_PATH = '/DATA/24ms190/steering_qwen/model'
DEVICE     = 'cuda:1'
EMOTIONS   = ['humor', 'melancholy', 'horror']

try: del model
except NameError: pass
gc.collect(); torch.cuda.empty_cache()

print('Loading Qwen/Qwen3-4B...')
model = HookedTransformer.from_pretrained(
    'Qwen/Qwen3-4B',
    dtype=torch.bfloat16, device=DEVICE,
    trust_remote_code=True, fold_ln=False,
    center_unembed=False, move_to_device=True,
    cache_dir=MODEL_PATH,
)
model.eval()
model.tokenizer.chat_template = None
print('✅ chat_template cleared')

N_LAYERS    = model.cfg.n_layers
D_MODEL     = model.cfg.d_model
PROBE_LAYER = N_LAYERS - 2
print(f'N_LAYERS={N_LAYERS}  D_MODEL={D_MODEL}  PROBE_LAYER={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

# Load data
df_humor = pd.read_csv(f'{DRIVE}/humor_master.csv')
df_mel   = pd.read_csv(f'{DRIVE}/melancholy_master.csv')
df_hor   = pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor, df_mel, df_hor]:
    col = next((c for c in df.columns if c.lower() in ['text','content','post','body']), df.columns[0])
    if col != 'text': df.rename(columns={col:'text'}, inplace=True)
    df['text'] = df['text'].fillna('').astype(str).str.strip()

# Load probe weights and steering vecs
w_hm = torch.load(f'{DRIVE}/probe_humor_vs_melancholy.pt').float().numpy()
w_hh = torch.load(f'{DRIVE}/probe_humor_vs_horror.pt').float().numpy()
sv_raw = torch.load(f'{DRIVE}/steering_vecs.pt')
sv = {k: v.to(DEVICE).to(torch.bfloat16) for k, v in sv_raw.items()}

with open(f'{DRIVE}/best_configs.json') as f:
    CONFIGS = json.load(f)

PROBE_FOR  = {'humor': w_hm, 'melancholy': w_hm, 'horror': w_hh}
PROBE_SIGN = {'humor': +1,   'melancholy': -1,   'horror': -1}

BEST_TEMP = float(CONFIGS.get('best_temp', 0.5))
actual_t  = BEST_TEMP
print(f'BEST_TEMP = {BEST_TEMP}')

def coherence_score(text):
    words = text.strip().split()
    if len(words) < 3: return 0.0
    counts = Counter(words)
    if counts.most_common(1)[0][1] / len(words) > 0.4: return 0.0
    chars = [c for c in text.lower() if c.isalpha()]
    if len(chars) < 10: return 0.0
    cc = Counter(chars)
    ent = -sum((c/len(chars))*math.log2(c/len(chars)) for c in cc.values())
    if ent < 3.5: return 0.0
    BAD = ['ss-','ses','sse','sss','thed','bsp','antml']
    def ok(w):
        w = w.strip('.,!?-()').lower()
        return len(w) < 2 or (any(c in 'aeiou' for c in w) and not any(p in w for p in BAD))
    return sum(ok(w) for w in words) / len(words)

def generate_with_act(prompt, max_new_tokens, temperature, extra_hooks=None):
    captured = {}
    hook_name = f'blocks.{PROBE_LAYER}.hook_resid_post'
    def capture_hook(resid_post, hook):
        captured['act'] = resid_post[0,-1,:].detach().cpu().float()
    hooks = [(hook_name, capture_hook)]
    if extra_hooks: hooks = extra_hooks + hooks
    with model.hooks(fwd_hooks=hooks):
        out_str = model.generate(prompt, max_new_tokens=max_new_tokens,
                                 temperature=temperature, verbose=False, prepend_bos=False)
    act = captured.get('act', torch.zeros(D_MODEL)).numpy()
    return out_str, act

def build_steer_hooks_vec(vec, emotion):
    cfg = CONFIGS[emotion]; layer = int(cfg['layer']); mult = float(cfg['mult'])
    def fn(resid_pre, hook):
        proj = (resid_pre @ vec).unsqueeze(-1) * vec
        return (resid_pre - proj) + vec * mult
    return [(f'blocks.{layer}.hook_resid_pre', fn)]

print(f'\n✅ Setup complete  VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

# ── Vector comparison ─────────────────────────────────────────────
print('\n' + '='*60)
print('VECTOR COMPARISON')
print('='*60)

VEC_N = 50
vec_sets = {}
for emotion in EMOTIONS:
    md  = sv[emotion].cpu().float()
    pw  = torch.tensor(PROBE_FOR[emotion]*PROBE_SIGN[emotion], dtype=torch.float32)
    pw  = pw/pw.norm().clamp(min=1e-8)
    rnd = torch.randn_like(md); rnd = rnd/rnd.norm().clamp(min=1e-8)
    vec_sets[emotion] = {
        'mean_diff': md.to(DEVICE).to(torch.bfloat16),
        'probe_w':   pw.to(DEVICE).to(torch.bfloat16),
        'random':    rnd.to(DEVICE).to(torch.bfloat16),
    }

eval_prompts_vc = {
    'humor':      df_humor.sample(VEC_N, random_state=77)['text'].tolist(),
    'melancholy': df_mel.sample(VEC_N,   random_state=77)['text'].tolist(),
    'horror':     df_hor.sample(min(VEC_N,len(df_hor)), random_state=77)['text'].tolist(),
}

vc_rows = []
vec_names = ['mean_diff', 'probe_w', 'random']

for emotion in EMOTIONS:
    for vec_name, vec in vec_sets[emotion].items():
        w = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]
        hooks = build_steer_hooks_vec(vec, emotion)
        results = []
        print(f'\n--- {emotion} | {vec_name} ---')
        t_start = time.time()

        for i, prompt in enumerate(eval_prompts_vc[emotion]):
            p = prompt[:120]
            t0 = time.time()

            _, base_act = generate_with_act(p, 60, actual_t)
            base_ps = sign * float(base_act @ w)

            ste_out, ste_act = generate_with_act(p, 60, actual_t, extra_hooks=hooks)
            ste_ps = sign * float(ste_act @ w)
            coh    = coherence_score(ste_out[len(p):].strip())
            both   = bool((ste_ps > base_ps) and (coh >= 0.6))
            shift  = float(ste_ps - base_ps)

            results.append({'both': both, 'ps_shift': shift, 'vec': vec_name, 'emotion': emotion})

            elapsed = time.time() - t0
            running_both = sum(r['both'] for r in results) / len(results)
            print(f'  [{i+1:2d}/{VEC_N}] shift={shift:+8.3f}  coh={coh:.2f}  both={both}  '
                  f'running_both={running_both:.0%}  ({elapsed:.1f}s)', flush=True)

        br = np.mean([r['both'] for r in results])
        sm = np.mean([r['ps_shift'] for r in results])
        total_t = time.time() - t_start
        print(f'  ✅ {emotion} | {vec_name}: both={br:.1%}  mean_shift={sm:+.4f}  '
              f'({total_t/60:.1f} min)', flush=True)
        vc_rows.extend(results)

df_vc = pd.DataFrame(vc_rows)
df_vc.to_csv(f'{DRIVE}/vector_comparison.csv', index=False)
print(f'\nvector_comparison.csv saved ({len(df_vc)} rows)')

# Bonferroni stats
n_comp    = len(EMOTIONS) * len(list(combinations(vec_names, 2)))
pval_rows = []
print(f'\nBonferroni Mann-Whitney (n_comp={n_comp}):')
for emotion in EMOTIONS:
    sub = df_vc[df_vc['emotion']==emotion]
    for v1, v2 in combinations(vec_names, 2):
        s1 = sub[sub['vec']==v1]['ps_shift'].values
        s2 = sub[sub['vec']==v2]['ps_shift'].values
        stat, p = scipy_stats.mannwhitneyu(s1, s2, alternative='two-sided')
        p_adj   = min(p * n_comp, 1.0)
        sig     = '***' if p_adj<0.001 else ('**' if p_adj<0.01 else ('*' if p_adj<0.05 else 'ns'))
        pval_rows.append({'emotion':emotion,'v1':v1,'v2':v2,'U':float(stat),
                         'p_raw':float(p),'p_bonf':float(p_adj),'sig':sig})
        print(f'  {emotion:12} {v1:12} vs {v2:12}  p={p_adj:.4f} {sig}')
pd.DataFrame(pval_rows).to_csv(f'{DRIVE}/vector_comparison_stats.csv', index=False)

# Plot
fig, axes_vc = plt.subplots(1, 3, figsize=(15, 5))
colors = {'mean_diff':'#2ecc71', 'probe_w':'#3498db', 'random':'#95a5a6'}
for idx, emotion in enumerate(EMOTIONS):
    ax  = axes_vc[idx]
    sub = df_vc[df_vc['emotion']==emotion]
    for vn in vec_names:
        ax.bar(vn, sub[sub['vec']==vn]['both'].mean(), color=colors[vn], alpha=0.85)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Both rate')
    ax.axhline(0.5, color='gray', ls='--', lw=1); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DRIVE}/vector_comparison.pdf', bbox_inches='tight')
plt.close()

# ── Confusion matrix from saved results ──────────────────────────
df_results = pd.read_csv(f'{DRIVE}/simplex_results_T_best.csv')
conf = np.zeros((3, 3))
for i, te in enumerate(EMOTIONS):
    sub = df_results[df_results['emotion_target']==te]
    for j, pe in enumerate(EMOTIONS):
        conf[i,j] = (sub['pred_emotion']==pe).sum() / max(len(sub), 1)
avg_diag = float(np.diag(conf).mean())

print(f'\nConfusion matrix (T={BEST_TEMP}):')
print(f'  {"":12} {"→humor":>10} {"→mel":>10} {"→horror":>10}')
for i, em in enumerate(EMOTIONS):
    print(f'  {em:12} {conf[i,0]:>10.1%} {conf[i,1]:>10.1%} {conf[i,2]:>10.1%}')
print(f'  Avg diagonal: {avg_diag:.1%}')

print(f'\nPer-emotion:')
for em in EMOTIONS:
    sub = df_results[df_results['emotion_target']==em]
    print(f'  {em:12} both={sub["both"].mean():.1%}  pred={sub["pred_correct"].mean():.1%}  '
          f'coh={sub["is_coherent"].mean():.1%}  shift={sub["ps_shift"].mean():+.4f}')

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(conf, annot=True, fmt='.1%',
            xticklabels=['→humor','→mel','→horror'],
            yticklabels=['humor','mel','horror'],
            cmap='Blues', ax=ax, vmin=0, vmax=1)
plt.tight_layout()
plt.savefig(f'{DRIVE}/probe_confusion_matrix_best.pdf', bbox_inches='tight')
plt.close()

# ── Save cell7_summary ────────────────────────────────────────────
cell7_summary = {
    'model': 'Qwen/Qwen3-4B', 'eval_temp': BEST_TEMP, 'best_temp': BEST_TEMP,
    'avg_diagonal': avg_diag,
    'per_emotion': {em: {
        'both':       float(df_results[df_results['emotion_target']==em]['both'].mean()),
        'pred':       float(df_results[df_results['emotion_target']==em]['pred_correct'].mean()),
        'coherent':   float(df_results[df_results['emotion_target']==em]['is_coherent'].mean()),
        'mean_shift': float(df_results[df_results['emotion_target']==em]['ps_shift'].mean()),
    } for em in EMOTIONS},
    'configs': CONFIGS,
    'vector_comparison': {
        em: {vn: float(df_vc[(df_vc['emotion']==em)&(df_vc['vec']==vn)]['both'].mean())
             for vn in vec_names}
        for em in EMOTIONS
    },
    'cross_model': {'llama_avg_diag': 0.900},
}
with open(f'{DRIVE}/cell7_summary.json', 'w') as f:
    json.dump(cell7_summary, f, indent=2)

print(f'\n{"━"*55}')
print(f'CELL 7 COMPLETE — Qwen/Qwen3-4B')
print(f'{"━"*55}')
print(f'  avg_diag:  {avg_diag:.1%}  (Llama: 90.0%)')
for em in EMOTIONS:
    sub = df_results[df_results['emotion_target']==em]
    print(f'  {em:12}: both={sub["both"].mean():.1%}  pred={sub["pred_correct"].mean():.1%}  '
          f'coh={sub["is_coherent"].mean():.1%}  shift={sub["ps_shift"].mean():+.2f}')

print(f'\n✅ Cell 7 complete. Saved:')
print(f'   vector_comparison.csv, vector_comparison_stats.csv')
print(f'   vector_comparison.pdf, probe_confusion_matrix_best.pdf')
print(f'   cell7_summary.json')
print(f'\nNext: python3 04_head_ablation_cpu_analyses_qwen.py')
