#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 5 (QWEN3-4B): ABLATIONS + DRIFT + ROBERTA
# ═══════════════════════════════════════════════════════════════════
# Model:   Qwen/Qwen3-4B
# Folder:  /DATA/24ms190/steering_qwen
# GPU:     cuda:1
# Runtime: ~9 hrs on RTX 6000
#
# Cell 11: Task 2 — additive vs proj-inject hook
#          Task 6 — 2-turn tonal drift
# Cell 12: Exp A — random vector baseline
#          Exp B — data size ablation
#          Exp C — RepE PCA vs mean-diff
#          Exp D — head ablation control
#          Exp E — 4-turn tonal drift
#          Exp F — DistilRoBERTa independent eval
# ═══════════════════════════════════════════════════════════════════

import os, gc, json, math, torch, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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
PROBE_LAYER = N_LAYERS - 2
print(f'N_LAYERS={N_LAYERS}  D_MODEL={D_MODEL}  PROBE_LAYER={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

df_humor = pd.read_csv(f'{DRIVE}/humor_master.csv')
df_mel   = pd.read_csv(f'{DRIVE}/melancholy_master.csv')
df_hor   = pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor, df_mel, df_hor]:
    col = next((c for c in df.columns if c.lower() in ['text','content','post','body']), df.columns[0])
    if col != 'text': df.rename(columns={col:'text'}, inplace=True)
    df['text'] = df['text'].fillna('').astype(str).str.strip()

w_hm = torch.load(f'{DRIVE}/probe_humor_vs_melancholy.pt').float().numpy()
w_hh = torch.load(f'{DRIVE}/probe_humor_vs_horror.pt').float().numpy()
sv_raw = torch.load(f'{DRIVE}/steering_vecs.pt')
sv = {k: v.to(DEVICE).to(torch.bfloat16) for k, v in sv_raw.items()}
with open(f'{DRIVE}/best_configs.json') as f: CONFIGS = json.load(f)
with open(f'{DRIVE}/cell9_summary.json') as f: critical_heads = json.load(f)['critical_heads']

humor_acts_pt = torch.load(f'{DRIVE}/humor_acts.pt', map_location='cpu').float()
dark_acts_pt  = torch.load(f'{DRIVE}/dark_acts.pt',  map_location='cpu').float()

PROBE_FOR  = {'humor': w_hm, 'melancholy': w_hm, 'horror': w_hh}
PROBE_SIGN = {'humor': +1,   'melancholy': -1,   'horror': -1}

BEST_TEMP = float(CONFIGS.get('best_temp', 0.5))
actual_t  = BEST_TEMP
ABL_N     = 15

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

def norm_v(v):
    n = np.linalg.norm(v); return v/n if n>1e-8 else v

def build_proj_hook(svec, layer, mult):
    def fn(resid_pre, hook):
        proj = (resid_pre @ svec).unsqueeze(-1) * svec
        return (resid_pre - proj) + svec * mult
    return [(f'blocks.{layer}.hook_resid_pre', fn)]

def build_add_hook(svec, layer, mult):
    def fn(resid_pre, hook):
        return resid_pre + svec * mult
    return [(f'blocks.{layer}.hook_resid_pre', fn)]

def build_hook(emotion):
    cfg = CONFIGS[emotion]; svec = sv[emotion]
    layer = int(cfg['layer']); mult = float(cfg['mult'])
    return build_proj_hook(svec, layer, mult)

def make_hook_from_np(vec_np, emotion):
    cfg = CONFIGS[emotion]; layer = int(cfg['layer']); mult = float(cfg['mult'])
    vec = torch.tensor(vec_np, dtype=torch.bfloat16).to(DEVICE)
    vec = vec / vec.norm().clamp(min=1e-8)
    return build_proj_hook(vec, layer, mult)

def run_eval(emotion, hooks, prompts, label):
    w = PROBE_FOR[emotion]; sign = PROBE_SIGN[emotion]; rows = []
    for i, prompt in enumerate(prompts):
        p = prompt[:120]; t0 = time.time()
        _, base_act = generate_with_act(p, 20, actual_t)
        base_ps = sign * float(base_act @ w)
        ste_out, ste_act = generate_with_act(p, 20, actual_t, extra_hooks=hooks)
        ste_ps = sign * float(ste_act @ w)
        coh    = coherence_score(ste_out[len(p):].strip())
        both   = bool((ste_ps > base_ps) and (coh >= 0.6))
        rows.append({'emotion': emotion, 'label': label,
                     'ps_shift': float(ste_ps-base_ps), 'both': both,
                     'pred_correct': bool(ste_ps>0), 'coherent': bool(coh>=0.6)})
        print(f'    [{i+1:2d}/{len(prompts)}] {emotion} {label}: shift={ste_ps-base_ps:+.3f}  '
              f'coh={coh:.2f}  both={both}  ({time.time()-t0:.1f}s)', flush=True)
    return rows

def horror_score(text):
    tokens = model.to_tokens([text], prepend_bos=False)
    hook_name = f'blocks.{PROBE_LAYER}.hook_resid_post'
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n==hook_name, stop_at_layer=PROBE_LAYER+1)
    act = cache[hook_name][0,-1,:].cpu().float().numpy()
    del cache; torch.cuda.empty_cache()
    return PROBE_SIGN['horror'] * float(act @ PROBE_FOR['horror'])

print(f'✅ Resume complete  VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')
results_all = {}

# ═══════════════════════════════════════════════════════════════════
# CELL 11: TASK 2 + TASK 6
# ═══════════════════════════════════════════════════════════════════

print('\n' + '='*55)
print('CELL 11: TASK 2 (hook comparison) + TASK 6 (drift)')
print('='*55)

# Task 2
TASK2_N = 30
task2_rows = []
for emotion in EMOTIONS:
    prompts = pd.concat([df_humor,df_mel,df_hor]).sample(TASK2_N,random_state=99)['text'].tolist()
    cfg = CONFIGS[emotion]; svec = sv[emotion]
    layer = int(cfg['layer']); mult = float(cfg['mult'])
    print(f'\n--- Task 2: {emotion} proj_inject ---')
    rows_pi = run_eval(emotion, build_proj_hook(svec,layer,mult), prompts, 'proj_inject')
    task2_rows.extend(rows_pi)
    print(f'\n--- Task 2: {emotion} additive ---')
    rows_ad = run_eval(emotion, build_add_hook(svec,layer,mult), prompts, 'additive')
    task2_rows.extend(rows_ad)
    pi_both = np.mean([r['both'] for r in rows_pi])
    ad_both = np.mean([r['both'] for r in rows_ad])
    print(f'\n  {emotion}: proj_inject={pi_both:.1%}  additive={ad_both:.1%}')

df_task2 = pd.DataFrame(task2_rows)
df_task2.to_csv(f'{DRIVE}/task2_additive_vs_projinject.csv', index=False)

print('\nTask 2 Mann-Whitney:')
for emotion in EMOTIONS:
    sub = df_task2[df_task2['emotion']==emotion]
    pi  = sub[sub['label']=='proj_inject']['ps_shift'].values
    ad  = sub[sub['label']=='additive']['ps_shift'].values
    stat,p = scipy_stats.mannwhitneyu(pi, ad, alternative='two-sided')
    sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    print(f'  {emotion:12}: proj={sub[sub["label"]=="proj_inject"]["both"].mean():.1%}  '
          f'add={sub[sub["label"]=="additive"]["both"].mean():.1%}  p={p:.4f} {sig}')

fig, axes_t2 = plt.subplots(1, 3, figsize=(14, 5))
for idx, emotion in enumerate(EMOTIONS):
    ax  = axes_t2[idx]
    sub = df_task2[df_task2['emotion']==emotion]
    for label, color in [('proj_inject','#3498db'),('additive','#e74c3c')]:
        ax.bar(label, sub[sub['label']==label]['both'].mean(), color=color, alpha=0.85)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Both rate'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DRIVE}/task2_additive_vs_projinject.pdf', bbox_inches='tight')
plt.close()

# Task 6 — 2-turn drift
print('\n--- Task 6: 2-turn tonal drift ---')
DRIFT_N = 30
horror_hooks = build_hook('horror')
neutral_turns = ["What happened next?","Tell me more.","And then?","Continue."]*10
drift_rows = []

for i, prompt in enumerate(df_hor.sample(DRIFT_N,random_state=55).reset_index(drop=True)['text'].tolist()):
    p = prompt[:120]
    with model.hooks(fwd_hooks=horror_hooks):
        t1 = model.generate(p, max_new_tokens=60, temperature=actual_t, verbose=False, prepend_bos=False)
    s1 = horror_score(t1)
    t2_ctx = t1 + ' ' + neutral_turns[i%len(neutral_turns)]
    t2 = model.generate(t2_ctx[:400], max_new_tokens=60, temperature=actual_t, verbose=False, prepend_bos=False)
    s2 = horror_score(t2)
    drift_rows.append({'pair_id':i,'turn1_horror':s1,'turn2_horror':s2,'horror_drift':s1-s2})
    print(f'  [{i+1:2d}/{DRIFT_N}] T1={s1:+.3f}  T2={s2:+.3f}  drift={s1-s2:+.3f}', flush=True)

df_drift = pd.DataFrame(drift_rows)
df_drift.to_csv(f'{DRIVE}/task6_tonal_drift.csv', index=False)
t_stat, t_p = scipy_stats.ttest_rel(df_drift['turn1_horror'], df_drift['turn2_horror'])
mw_stat, mw_p = scipy_stats.mannwhitneyu(df_drift['turn1_horror'], df_drift['turn2_horror'], alternative='two-sided')
print(f'\nTask 6 results:')
print(f'  Mean T1 horror: {df_drift["turn1_horror"].mean():+.4f}')
print(f'  Mean T2 horror: {df_drift["turn2_horror"].mean():+.4f}')
print(f'  Mean drift:     {df_drift["horror_drift"].mean():+.4f}')
print(f'  t-test: t={t_stat:.3f}  p={t_p:.4f}')
print(f'  Mann-Whitney: U={mw_stat:.1f}  p={mw_p:.4f}')
print(f'  Neutralized (T2<T1): {(df_drift["turn2_horror"]<df_drift["turn1_horror"]).mean():.1%}')

task6_summary = {
    'mean_t1': float(df_drift['turn1_horror'].mean()),
    'mean_t2': float(df_drift['turn2_horror'].mean()),
    'mean_drift': float(df_drift['horror_drift'].mean()),
    't_stat': float(t_stat), 't_p': float(t_p),
    'pct_neutralized': float((df_drift['turn2_horror']<df_drift['turn1_horror']).mean()),
}
with open(f'{DRIVE}/task6_tonal_drift_summary.json','w') as f: json.dump(task6_summary, f, indent=2)

fig, axes_d = plt.subplots(1, 2, figsize=(12, 5))
axes_d[0].scatter(df_drift['turn1_horror'], df_drift['turn2_horror'], alpha=0.7, color='#3498db')
mn = min(df_drift['turn1_horror'].min(), df_drift['turn2_horror'].min())
mx = max(df_drift['turn1_horror'].max(), df_drift['turn2_horror'].max())
axes_d[0].plot([mn,mx],[mn,mx],'k--',lw=1,alpha=0.5)
axes_d[0].set_xlabel('Turn 1 horror score'); axes_d[0].set_ylabel('Turn 2 horror score')
axes_d[1].hist(df_drift['horror_drift'], bins=15, color='#e74c3c', alpha=0.8, edgecolor='white')
axes_d[1].axvline(df_drift['horror_drift'].mean(), color='k', lw=2, ls='--')
axes_d[1].set_xlabel('Drift (T1 - T2)')
plt.tight_layout()
plt.savefig(f'{DRIVE}/task6_tonal_drift.pdf', bbox_inches='tight')
plt.close()
print('✅ Cell 11 complete')

# ═══════════════════════════════════════════════════════════════════
# CELL 12: SIX ABLATION EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════

print('\n' + '='*55)
print('CELL 12: SIX ABLATION EXPERIMENTS')
print('='*55)

# Exp A — Random baseline
print('\n--- Exp A: Random vector baseline ---')
exp_a_rows = []
for emotion in EMOTIONS:
    prompts = pd.concat([df_humor,df_mel,df_hor]).sample(ABL_N,random_state=11)['text'].tolist()
    print(f'\n  {emotion} our_vec:')
    rows_ours = run_eval(emotion, build_hook(emotion), prompts, 'our_vec')
    exp_a_rows.extend(rows_ours)
    for seed in [101, 202, 303]:
        rng = np.random.default_rng(seed)
        r_vec = norm_v(rng.standard_normal(D_MODEL).astype(np.float32))
        print(f'\n  {emotion} random_{seed}:')
        exp_a_rows.extend(run_eval(emotion, make_hook_from_np(r_vec,emotion), prompts, f'random_{seed}'))

df_a = pd.DataFrame(exp_a_rows)
df_a.to_csv(f'{DRIVE}/task_random_baseline.csv', index=False)
results_all['exp_a'] = df_a.groupby(['emotion','label'])['both'].mean().to_dict()
print('\nExp A summary:')
for emotion in EMOTIONS:
    sub = df_a[df_a['emotion']==emotion]
    our = sub[sub['label']=='our_vec']['both'].mean()
    rnd = sub[sub['label'].str.startswith('random')]['both'].mean()
    print(f'  {emotion}: our={our:.1%}  random_avg={rnd:.1%}')
print('✅ Exp A done')

# Exp B — Data size ablation
print('\n--- Exp B: Data size ablation ---')
emotion = 'humor'
prompts_b = df_humor.sample(ABL_N,random_state=22).reset_index(drop=True)['text'].tolist()
min_size = min(len(humor_acts_pt), len(dark_acts_pt))
sizes = [s for s in [25,50,100,150,200,300,min_size] if s<=min_size]
exp_b_rows = []
for size in sizes:
    rng = np.random.default_rng(42)
    md_vec = norm_v(
        humor_acts_pt[rng.choice(len(humor_acts_pt),size=size,replace=False)].numpy().mean(0) -
        dark_acts_pt[rng.choice(len(dark_acts_pt), size=size,replace=False)].numpy().mean(0)
    ).astype(np.float32)
    print(f'\n  size={size}:')
    rows = run_eval(emotion, make_hook_from_np(md_vec,emotion), prompts_b, f'size_{size}')
    for r in rows: r['data_size'] = size
    exp_b_rows.extend(rows)
    print(f'  size={size}: both={np.mean([r["both"] for r in rows]):.1%}')

df_b = pd.DataFrame(exp_b_rows)
df_b.to_csv(f'{DRIVE}/task_data_ablation.csv', index=False)
results_all['exp_b'] = df_b.groupby('data_size')['both'].mean().to_dict()
print('\nExp B summary:')
for size, val in sorted(results_all['exp_b'].items()):
    print(f'  size={size}: both={val:.1%}')
print('✅ Exp B done')

# Exp C — RepE PCA vs mean-diff
print('\n--- Exp C: RepE PCA vs mean-diff ---')
stacked  = np.vstack([humor_acts_pt.numpy(), dark_acts_pt.numpy()])
pca      = PCA(n_components=1); pca.fit(stacked - stacked.mean(0))
pca_vec  = pca.components_[0].astype(np.float32)
md_vec   = norm_v(humor_acts_pt.numpy().mean(0) - dark_acts_pt.numpy().mean(0)).astype(np.float32)
w_hm_norm = norm_v(w_hm.astype(np.float32))
cos_pca_md  = float(norm_v(pca_vec) @ norm_v(md_vec))
cos_md_probe = float(norm_v(md_vec) @ w_hm_norm)
print(f'  cos(PCA, mean_diff) = {cos_pca_md:+.4f}')
print(f'  cos(mean_diff, probe_w) = {cos_md_probe:+.4f}')

exp_c_rows = []
for emotion in EMOTIONS:
    prompts_c = pd.concat([df_humor,df_mel,df_hor]).sample(ABL_N,random_state=33)['text'].tolist()
    for vn, vec in [('pca',pca_vec),('mean_diff',md_vec)]:
        print(f'\n  {emotion} {vn}:')
        exp_c_rows.extend(run_eval(emotion, make_hook_from_np(vec,emotion), prompts_c, vn))

df_c = pd.DataFrame(exp_c_rows)
df_c.to_csv(f'{DRIVE}/task_repe_comparison.csv', index=False)
results_all['exp_c'] = {'cos_pca_md': cos_pca_md, 'cos_md_probe': cos_md_probe}
print('\nExp C summary:')
for emotion in EMOTIONS:
    sub = df_c[df_c['emotion']==emotion]
    pca_b = sub[sub['label']=='pca']['both'].mean()
    md_b  = sub[sub['label']=='mean_diff']['both'].mean()
    print(f'  {emotion}: PCA={pca_b:.1%}  mean_diff={md_b:.1%}')
print('✅ Exp C done')

# Exp D — Head specificity control
print('\n--- Exp D: Head ablation control ---')
sv_pairs = [
    ("The keys to the cabinet","are","is"),("The dog near the trees","runs","run"),
    ("The manager of the stores","was","were"),("The books on the shelf","are","is"),
    ("The girl with the cats","plays","play"),("The author of the novels","writes","write"),
    ("The students in the class","study","studies"),("The bird by the windows","sings","sing"),
    ("The child near the boxes","cries","cry"),("The teacher with the markers","draws","draw"),
]*4

def sv_accuracy(ablate_layer=None, ablate_head=None):
    correct = 0
    for subj, gram, ungram in sv_pairs:
        ctx = model.to_tokens([subj], prepend_bos=False); hooks = []
        if ablate_layer is not None:
            h = ablate_head
            def make_z(hi):
                def fn(z, hook): z_new=z.clone(); z_new[:,:,hi,:]=0.0; return z_new
                return fn
            hooks = [(f'blocks.{ablate_layer}.attn.hook_z', make_z(h))]
        with torch.no_grad():
            if hooks:
                with model.hooks(fwd_hooks=hooks): logits = model(ctx)
            else: logits = model(ctx)
        last = logits[0,-1,:]
        g_id = model.to_tokens([gram],   prepend_bos=False)[0,0]
        u_id = model.to_tokens([ungram], prepend_bos=False)[0,0]
        if last[g_id] > last[u_id]: correct += 1
    return correct/len(sv_pairs)

sv_base = sv_accuracy()
print(f'  SV baseline accuracy: {sv_base:.4f}')
exp_d_rows = []
for ch in tqdm(critical_heads[:10], desc='Critical heads'):
    l, h = int(ch['layer']), int(ch['head']); emo_drop = float(ch['drop'])
    sv_abl = sv_accuracy(ablate_layer=l, ablate_head=h)
    specificity = emo_drop - (sv_base - sv_abl)
    exp_d_rows.append({'layer':l,'head':h,'emo_drop':emo_drop,
                       'sv_drop':sv_base-sv_abl,'specificity':specificity})
    print(f'  L{l:2d} H{h:2d}: emo_drop={emo_drop:.4f}  sv_drop={sv_base-sv_abl:.4f}  spec={specificity:.4f}', flush=True)

df_d = pd.DataFrame(exp_d_rows) if exp_d_rows else pd.DataFrame(columns=['layer','head','emo_drop','sv_drop','specificity'])
df_d.to_csv(f'{DRIVE}/task_head_ablation_control.csv', index=False)
results_all['exp_d'] = {'sv_baseline': sv_base, 'mean_specificity': float(df_d['specificity'].mean()) if len(df_d)>0 else 0.0}
print(f'  Mean specificity: {results_all["exp_d"]["mean_specificity"]:.4f}')
print('✅ Exp D done')

# Exp E — 4-turn drift
print('\n--- Exp E: 4-turn drift ---')
DRIFT4_N = 15
exp_e_rows = []
for i, prompt in enumerate(df_hor.sample(DRIFT4_N,random_state=66).reset_index(drop=True)['text'].tolist()):
    p = prompt[:120]
    with model.hooks(fwd_hooks=horror_hooks):
        t1 = model.generate(p, max_new_tokens=40, temperature=actual_t, verbose=False, prepend_bos=False)
    scores = [horror_score(t1)]; ctx = t1
    for turn in range(2, 5):
        nu  = neutral_turns[(i*3+turn) % len(neutral_turns)]
        ctx = (ctx + ' ' + nu)[-400:]
        out = model.generate(ctx, max_new_tokens=40, temperature=actual_t, verbose=False, prepend_bos=False)
        scores.append(horror_score(out)); ctx = out
    exp_e_rows.append({'pair_id':i,'score_t1':scores[0],'score_t2':scores[1],
                       'score_t3':scores[2],'score_t4':scores[3]})
    print(f'  [{i+1:2d}/{DRIFT4_N}] T1={scores[0]:+.3f}  T2={scores[1]:+.3f}  T3={scores[2]:+.3f}  T4={scores[3]:+.3f}', flush=True)

df_e = pd.DataFrame(exp_e_rows)
df_e.to_csv(f'{DRIVE}/task_4turn_drift.csv', index=False)
results_all['exp_e'] = {f'mean_t{t}': float(df_e[f'score_t{t}'].mean()) for t in [1,2,3,4]}
print('\nExp E summary:')
for t in [1,2,3,4]:
    print(f'  T{t}: {df_e[f"score_t{t}"].mean():+.4f}')

fig, ax = plt.subplots(figsize=(8, 5))
for _, row in df_e.iterrows():
    ax.plot([1,2,3,4],[row['score_t1'],row['score_t2'],row['score_t3'],row['score_t4']],
            color='gray', alpha=0.3, lw=1)
means = [df_e[f'score_t{t}'].mean() for t in [1,2,3,4]]
ax.plot([1,2,3,4], means, color='#e74c3c', lw=3, marker='o')
ax.set_xlabel('Turn'); ax.set_ylabel('Horror probe score'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DRIVE}/task_4turn_drift.pdf', bbox_inches='tight')
plt.close()
print('✅ Exp E done')

# Exp F — DistilRoBERTa
print('\n--- Exp F: DistilRoBERTa independent eval ---')
try:
    from transformers import pipeline
    roberta_pipe = pipeline('text-classification',
                            model='j-hartmann/emotion-english-distilroberta-base',
                            device=-1, top_k=None)
    df_rob = pd.read_csv(f'{DRIVE}/simplex_results_T_best.csv')
    LABEL_MAP = {'humor':'joy','melancholy':'sadness','horror':'fear'}
    rob_scores = []
    for i, (_, row) in enumerate(df_rob.iterrows()):
        target = LABEL_MAP.get(row['emotion_target'],'joy')
        text   = str(row.get('ste_gen',''))[:512]
        try:
            res = roberta_pipe(text, truncation=True)[0]
            sc  = {r['label'].lower(): r['score'] for r in res}
            rob_scores.append(float(sc.get(target, 0.0)))
        except: rob_scores.append(0.0)
        if (i+1) % 30 == 0:
            print(f'  [{i+1}/{len(df_rob)}] running...', flush=True)
    df_rob['roberta_target_score'] = rob_scores
    df_rob.to_csv(f'{DRIVE}/task_roberta_eval.csv', index=False)
    exp_f = {}
    print('\nExp F results:')
    for emotion in EMOTIONS:
        sub = df_rob[df_rob['emotion_target']==emotion]
        r, p = scipy_stats.spearmanr(sub['ps_shift'], sub['roberta_target_score'])
        sig  = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
        rob_acc = (sub['roberta_target_score']>0.3).mean()
        print(f'  {emotion:12}: r={r:+.4f}  p={p:.4f} {sig}  RoBERTa>0.3={rob_acc:.1%}')
        exp_f[emotion] = {'spearman_r': float(r), 'p': float(p), 'roberta_acc': float(rob_acc)}
    results_all['exp_f'] = exp_f
    print('✅ Exp F done')
except Exception as e:
    print(f'Exp F failed: {e}')
    results_all['exp_f'] = {'error': str(e)}

with open(f'{DRIVE}/cell12_summary.json','w') as f:
    json.dump({str(k): str(v) for k,v in results_all.items()}, f, indent=2)

print(f'\n{"━"*55}')
print(f'SCRIPT 5 SUMMARY — Qwen/Qwen3-4B')
print(f'{"━"*55}')
print(f'  Task 2: proj_inject vs additive completed')
print(f'  Task 6: 2-turn drift  T1={df_drift["turn1_horror"].mean():+.3f}  T2={df_drift["turn2_horror"].mean():+.3f}')
print(f'  Exp A-F: all ablations complete')
print(f'\n✅ Script 5 complete. Saved to {DRIVE}')
print(f'\nNext: python3 06_deepseek_judge_qwen.py')
