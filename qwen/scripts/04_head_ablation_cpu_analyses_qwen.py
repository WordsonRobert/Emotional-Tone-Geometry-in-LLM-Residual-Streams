#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 4 (QWEN3-4B): HEAD ABLATION + CPU ANALYSES
# ═══════════════════════════════════════════════════════════════════
# Model:   Qwen/Qwen3-4B
# Folder:  /DATA/24ms190/steering_qwen
# GPU:     cuda:1
# Runtime: ~1.5 hrs on RTX 6000
#
# Cell 9  (GPU): Attention head ablation at key layers
# Cell 10 (CPU): Task 3 — three-pole geometry
#                Task 4 — surface vs activation probe
#                Task 5 — OOD evaluation
# ═══════════════════════════════════════════════════════════════════

import os, gc, json, math, torch, re, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats
from transformer_lens import HookedTransformer
import warnings; warnings.filterwarnings('ignore')

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

N_LAYERS    = model.cfg.n_layers
D_MODEL     = model.cfg.d_model
N_HEADS     = model.cfg.n_heads
PROBE_LAYER = N_LAYERS - 2
print(f'N_LAYERS={N_LAYERS}  D_MODEL={D_MODEL}  N_HEADS={N_HEADS}  PROBE_LAYER={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

df_humor = pd.read_csv(f'{DRIVE}/humor_master.csv')
df_mel   = pd.read_csv(f'{DRIVE}/melancholy_master.csv')
df_hor   = pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor, df_mel, df_hor]:
    col = next((c for c in df.columns if c.lower() in ['text','content','post','body']), df.columns[0])
    if col != 'text': df.rename(columns={col:'text'}, inplace=True)
    df['text'] = df['text'].fillna('').astype(str).str.strip()

probe_weights = torch.load(f'{DRIVE}/probe_weights.pt', map_location='cpu')
with open(f'{DRIVE}/best_configs.json') as f:
    CONFIGS = json.load(f)

print(f'✅ Resume complete  VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

# ═══════════════════════════════════════════════════════════════════
# CELL 9: ATTENTION HEAD ABLATION
# ═══════════════════════════════════════════════════════════════════

steer_layers    = list(set([CONFIGS[em]['layer'] for em in EMOTIONS]))
ablation_layers = sorted(set([0, N_LAYERS//4, N_LAYERS//2, PROBE_LAYER] + steer_layers))
N_ABL  = 100; n_each = N_ABL // 2

rng    = np.random.default_rng(42)
h_idx  = rng.choice(len(df_humor), size=n_each, replace=False)
m_idx  = rng.choice(len(df_mel),   size=n_each, replace=False)
abl_texts  = df_humor['text'].iloc[h_idx].tolist() + df_mel['text'].iloc[m_idx].tolist()
abl_labels = [1]*n_each + [0]*n_each

print(f'\nAblation layers: {ablation_layers}  ({N_HEADS} heads each)')
print(f'n_texts={N_ABL}  n_each={n_each}')

def get_acts_at_layer(texts, layer, ablate_head=None):
    acts = []
    hook_name = f'blocks.{layer}.hook_resid_post'
    for text in texts:
        tokens = model.to_tokens([text], prepend_bos=False)
        # No truncation — full token limit
        if tokens.shape[1] == 0: acts.append(np.zeros(D_MODEL, dtype=np.float32)); continue
        hooks = []
        if ablate_head is not None:
            h = ablate_head
            def make_z(hi):
                def fn(z, hook):
                    z_new = z.clone(); z_new[:,:,hi,:] = 0.0; return z_new
                return fn
            hooks = [(f'blocks.{layer}.attn.hook_z', make_z(h))]
        with torch.no_grad():
            if hooks:
                with model.hooks(fwd_hooks=hooks):
                    _, cache = model.run_with_cache(tokens,
                        names_filter=lambda n: hook_name in n, stop_at_layer=layer+1)
            else:
                _, cache = model.run_with_cache(tokens,
                    names_filter=lambda n: hook_name in n, stop_at_layer=layer+1)
        acts.append(cache[hook_name][0,-1,:].cpu().float().numpy())
        del cache; torch.cuda.empty_cache()
    return np.array(acts)

def probe_accuracy(acts, labels, layer):
    w = probe_weights[layer].float().numpy()
    preds = (acts @ w > 0).astype(int)
    return float((preds == np.array(labels)).mean())

CHECKPOINT = f'{DRIVE}/task19_ablation_results.json'
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f: results = json.load(f)
    print(f'Resumed: {len(results)} entries done')
else:
    results = {}

for layer in tqdm(ablation_layers, desc='Ablation layers'):
    base_key = f'baseline_L{layer}'
    if base_key not in results:
        print(f'\n  L{layer:2d} computing baseline...', flush=True)
        t0 = time.time()
        base_acts = get_acts_at_layer(abl_texts, layer)
        base_acc  = probe_accuracy(base_acts, abl_labels, layer)
        results[base_key] = float(base_acc)
        with open(CHECKPOINT,'w') as f: json.dump(results, f)
        print(f'  L{layer:2d} baseline = {base_acc:.4f}  ({time.time()-t0:.1f}s)', flush=True)
    else:
        base_acc = results[base_key]

    for head_idx in range(N_HEADS):
        key = f'L{layer}_H{head_idx}'
        if key in results: continue
        abl_acts = get_acts_at_layer(abl_texts, layer, ablate_head=head_idx)
        abl_acc  = probe_accuracy(abl_acts, abl_labels, layer)
        drop     = float(base_acc - abl_acc)
        results[key] = {
            'layer': layer, 'head': head_idx,
            'baseline_acc': float(base_acc), 'ablated_acc': float(abl_acc),
            'drop': drop, 'load_bearing': bool(drop > 0.02)
        }
        if drop > 0.02:
            print(f'  ⚠️  L{layer:2d} H{head_idx:2d}: drop={drop:.4f}  LOAD-BEARING', flush=True)
        else:
            print(f'  L{layer:2d} H{head_idx:2d}: drop={drop:+.4f}', flush=True)
    with open(CHECKPOINT,'w') as f: json.dump(results, f)

rows   = [v for k,v in results.items() if isinstance(v,dict)]
df_abl = pd.DataFrame(rows)
df_abl.to_csv(f'{DRIVE}/task19_ablation_results.csv', index=False)

critical_heads = df_abl[df_abl['load_bearing']].sort_values('drop', ascending=False)
print(f'\nCritical heads (drop>2%): {len(critical_heads)}')
if len(critical_heads) > 0:
    print(critical_heads[['layer','head','baseline_acc','ablated_acc','drop']].to_string(index=False))

fig, axes_plt = plt.subplots(1, len(ablation_layers), figsize=(4*len(ablation_layers), 5))
if len(ablation_layers)==1: axes_plt=[axes_plt]
for idx, layer in enumerate(ablation_layers):
    ax  = axes_plt[idx]
    sub = df_abl[df_abl['layer']==layer].sort_values('head')
    drops  = sub['drop'].values
    colors = ['#C44E52' if d>0.02 else '#4C72B0' for d in drops]
    ax.bar(range(len(drops)), drops, color=colors, alpha=0.85)
    ax.axhline(0.02, color='k', lw=1, ls='--')
    ax.set_xlabel('Head')
    ax.set_xticks(range(0, N_HEADS, max(1, N_HEADS//8)))
plt.tight_layout()
plt.savefig(f'{DRIVE}/task19_head_ablation.pdf', bbox_inches='tight')
plt.close()

cell9_summary = {
    'ablation_layers': ablation_layers, 'n_heads': N_HEADS, 'n_texts': N_ABL,
    'n_critical': int(len(critical_heads)),
    'critical_heads': critical_heads[['layer','head','drop']].to_dict('records') if len(critical_heads) > 0 else [],
}
with open(f'{DRIVE}/cell9_summary.json','w') as f:
    json.dump(cell9_summary, f, indent=2)
print('\n✅ Cell 9 complete')

# ═══════════════════════════════════════════════════════════════════
# CELL 10: CPU ANALYSES — Tasks 3, 4, 5
# ═══════════════════════════════════════════════════════════════════

df_results = pd.read_csv(f'{DRIVE}/simplex_results_T_best.csv')
sv_raw     = torch.load(f'{DRIVE}/steering_vecs.pt', map_location='cpu')
sv_np      = {k: v.float().numpy() for k, v in sv_raw.items()}

def norm_v(v): return v / (np.linalg.norm(v)+1e-8)

# ── Task 3: Three-pole geometry ───────────────────────────────────
print('\n' + '='*50)
print('TASK 3: THREE-POLE GEOMETRY')
print('='*50)

conf = np.zeros((3,3))
for i,te in enumerate(EMOTIONS):
    sub = df_results[df_results['emotion_target']==te]
    for j,pe in enumerate(EMOTIONS):
        conf[i,j] = (sub['pred_emotion']==pe).sum()/max(len(sub),1)

mel_horror_bleed  = (conf[1,2]+conf[2,1])/2
humor_other_bleed = (conf[0,1]+conf[0,2]+conf[1,0]+conf[2,0])/4

print(f'Confusion matrix:')
print(f'  {"":12} {"→humor":>10} {"→mel":>10} {"→horror":>10}')
for i, em in enumerate(EMOTIONS):
    print(f'  {em:12} {conf[i,0]:>10.1%} {conf[i,1]:>10.1%} {conf[i,2]:>10.1%}')
print(f'\nmel↔horror bleed: {mel_horror_bleed:.1%}')
print(f'humor↔other bleed: {humor_other_bleed:.1%}')
print(f'ratio: {mel_horror_bleed/max(humor_other_bleed,1e-8):.2f}x')

cos_mat = np.zeros((3,3))
for i,a in enumerate(EMOTIONS):
    for j,b in enumerate(EMOTIONS):
        cos_mat[i,j] = float(norm_v(sv_np[a]) @ norm_v(sv_np[b]))

print(f'\nSV cosine matrix:')
print(f'  {"":12} {"humor":>10} {"mel":>10} {"horror":>10}')
for i, em in enumerate(EMOTIONS):
    print(f'  {em:12} {cos_mat[i,0]:>10.4f} {cos_mat[i,1]:>10.4f} {cos_mat[i,2]:>10.4f}')

task3 = {
    'mel_horror_bleed': float(mel_horror_bleed),
    'humor_other_bleed': float(humor_other_bleed),
    'bleed_ratio': float(mel_horror_bleed/max(humor_other_bleed,1e-8)),
    'sv_cosines': {f'{a}_{b}': float(cos_mat[i,j])
                   for i,a in enumerate(EMOTIONS) for j,b in enumerate(EMOTIONS)}
}
with open(f'{DRIVE}/task3_binary_vs_threepole_results.json','w') as f:
    json.dump(task3, f, indent=2)

fig, axes_p = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf, annot=True, fmt='.1%',
            xticklabels=['→humor','→mel','→horror'],
            yticklabels=['humor','mel','horror'],
            cmap='Blues', ax=axes_p[0], vmin=0, vmax=1)
sns.heatmap(cos_mat, annot=True, fmt='.3f',
            xticklabels=EMOTIONS, yticklabels=EMOTIONS,
            cmap='RdYlGn', center=0, ax=axes_p[1])
plt.tight_layout()
plt.savefig(f'{DRIVE}/task3_binary_vs_threepole.pdf', bbox_inches='tight')
plt.close()
print('✅ Task 3 done')

# ── Task 4: Surface vs activation probe ───────────────────────────
print('\n' + '='*50)
print('TASK 4: SURFACE VS ACTIVATION PROBE')
print('='*50)

def extract_surface_features(texts):
    feats = []
    for text in texts:
        words = text.split(); sents = re.split(r'[.!?]+', text)
        sents = [s.strip() for s in sents if s.strip()]; chars = list(text)
        feats.append({
            'ttr':           len(set(words))/max(len(words),1),
            'punct_density': sum(1 for c in chars if c in '.,!?;:')/max(len(chars),1),
            'sent_count':    len(sents),
            'avg_word_len':  np.mean([len(w) for w in words]) if words else 0,
            'exclaim_rate':  text.count('!')/max(len(words),1),
            'question_rate': text.count('?')/max(len(words),1),
            'upper_rate':    sum(1 for c in chars if c.isupper())/max(len(chars),1),
            'avg_sent_len':  np.mean([len(s.split()) for s in sents]) if sents else 0,
            'ellipsis_rate': text.count('...')/max(len(words),1),
            'digit_rate':    sum(1 for c in chars if c.isdigit())/max(len(chars),1),
        })
    return pd.DataFrame(feats)

all_texts  = df_humor['text'].tolist()+df_mel['text'].tolist()+df_hor['text'].tolist()
all_labels = [0]*len(df_humor)+[1]*len(df_mel)+[2]*len(df_hor)
X_surf = StandardScaler().fit_transform(extract_surface_features(all_texts).values)
y      = np.array(all_labels)
clf_surf = LogisticRegression(max_iter=1000)
surf_acc = cross_val_score(clf_surf, X_surf, y,
                           cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)).mean()
act_acc  = float(df_results['pred_correct'].mean())

print(f'Surface probe CV accuracy:    {surf_acc:.4f}')
print(f'Activation probe accuracy:    {act_acc:.4f}')
print(f'Gap (activation - surface):  {act_acc-surf_acc:+.4f}')

task4 = {'surface_probe_acc': float(surf_acc), 'activation_probe_acc': float(act_acc),
         'gap': float(act_acc-surf_acc)}
with open(f'{DRIVE}/task4_surface_probe_results.json','w') as f:
    json.dump(task4, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Surface', 'Activation'], [surf_acc, act_acc], color=['#95a5a6','#2ecc71'], alpha=0.85)
ax.set_ylim(0, 1.1); ax.set_ylabel('CV Accuracy'); ax.grid(axis='y', alpha=0.3)
ax.axhline(surf_acc, color='gray', ls='--', lw=1)
plt.tight_layout()
plt.savefig(f'{DRIVE}/task4_surface_probe.pdf', bbox_inches='tight')
plt.close()
print('✅ Task 4 done')

# ── Task 5: OOD evaluation ────────────────────────────────────────
print('\n' + '='*50)
print('TASK 5: OOD EVALUATION')
print('='*50)
try:
    from datasets import load_dataset
    clf_ood = LogisticRegression(max_iter=1000); scaler5 = StandardScaler()
    X5 = scaler5.fit_transform(extract_surface_features(all_texts).values)
    clf_ood.fit(X5, y)
    ood_results = {}

    print('Loading english_quotes...', flush=True)
    ds = load_dataset('Abirate/english_quotes', split='train')
    qt = [r['quote'] for r in ds if isinstance(r['quote'],str) and 30<len(r['quote'])<200][:200]
    preds = clf_ood.predict(scaler5.transform(extract_surface_features(qt).values))
    ood_results['humor_ood'] = {'acc': float((preds==0).mean()), 'n': len(qt)}
    print(f'  Humor OOD:     {ood_results["humor_ood"]["acc"]:.4f}  (n={len(qt)})')

    print('Loading merve/poetry...', flush=True)
    ds2 = load_dataset('merve/poetry', split='train')
    sad_kw = ['sad','sorrow','grief','loss','pain','lonely','dark','tears','death','despair']
    mel_t = [str(r.get('content',''))[:400] for r in ds2
             if any(k in str(r.get('content','')).lower() for k in sad_kw)][:200]
    if len(mel_t) < 20:
        mel_t = [str(r.get('content',''))[:400] for r in ds2 if len(str(r.get('content','')))>30][:200]
    preds2 = clf_ood.predict(scaler5.transform(extract_surface_features(mel_t).values))
    ood_results['mel_ood'] = {'acc': float((preds2==1).mean()), 'n': len(mel_t)}
    print(f'  Melancholy OOD: {ood_results["mel_ood"]["acc"]:.4f}  (n={len(mel_t)})')

    print('Loading imdb...', flush=True)
    ds3 = load_dataset('stanfordnlp/imdb', split='test')
    hor_t = [r['text'][:400] for r in ds3 if r['label']==0 and len(r['text'])>50][:200]
    preds3 = clf_ood.predict(scaler5.transform(extract_surface_features(hor_t).values))
    ood_results['horror_ood'] = {'acc': float((preds3==2).mean()), 'n': len(hor_t)}
    print(f'  Horror OOD:     {ood_results["horror_ood"]["acc"]:.4f}  (n={len(hor_t)})')

    print(f'\nIn-dist surface acc: {surf_acc:.4f}')
    for k,v in ood_results.items():
        drop = v['acc'] - surf_acc
        print(f'  {k}: acc={v["acc"]:.4f}  drop={drop:+.4f}')

    with open(f'{DRIVE}/task5_ood_results.json','w') as f:
        json.dump({'in_dist': float(surf_acc), 'ood': ood_results}, f, indent=2)
    print('✅ Task 5 done')
except Exception as e:
    print(f'Task 5 error: {e}')
    with open(f'{DRIVE}/task5_ood_results.json','w') as f:
        json.dump({'error': str(e)}, f)

print(f'\n{"━"*55}')
print(f'SCRIPT 4 SUMMARY — Qwen/Qwen3-4B')
print(f'{"━"*55}')
print(f'  Critical heads (drop>2%): {len(critical_heads)}')
print(f'  Surface probe acc:  {surf_acc:.4f}')
print(f'  Activation probe acc: {act_acc:.4f}')
print(f'  Gap: {act_acc-surf_acc:+.4f}')
print(f'  mel↔horror bleed: {mel_horror_bleed:.1%}  ratio: {mel_horror_bleed/max(humor_other_bleed,1e-8):.2f}x')
print(f'\n✅ Script 4 complete. Saved to {DRIVE}:')
for f in ['task19_ablation_results.csv','task19_head_ablation.pdf','cell9_summary.json',
          'task3_binary_vs_threepole_results.json','task3_binary_vs_threepole.pdf',
          'task4_surface_probe_results.json','task4_surface_probe.pdf',
          'task5_ood_results.json']:
    print(f'   {f}')
print(f'\nNext: python3 05_ablations_drift_roberta_qwen.py')
