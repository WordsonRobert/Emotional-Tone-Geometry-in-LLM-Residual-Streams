#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 10 (QWEN3-4B): EMOTIONAL AXIS CAUSAL VALIDATION
# ═══════════════════════════════════════════════════════════════════
# Model:   Qwen/Qwen3-4B
# Folder:  /DATA/24ms190/steering_qwen/emotional_axes/
# GPU:     cuda:1  +  DeepSeek API
# Runtime: ~30 min GPU + ~10 min scoring
#
# Steers along depth/arousal/dark-bright axes at mult [3,6,9]
# in both directions, generates n=20 outputs per condition,
# scores with DeepSeek to validate causal effect.
# ═══════════════════════════════════════════════════════════════════

import os, json, time
import torch, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
from transformer_lens import HookedTransformer
import warnings; warnings.filterwarnings('ignore')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DRIVE_EA   = '/DATA/24ms190/steering_qwen/emotional_axes'
MODEL_PATH = '/DATA/24ms190/steering_qwen/model'
DEVICE     = 'cuda:1'

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = input('Paste DEEPSEEK_API_KEY: ').strip()

try: del model
except NameError: pass
import gc; gc.collect(); torch.cuda.empty_cache()

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
PROBE_LAYER = N_LAYERS - 2

# Best steering layer from Script 02 (humor best layer)
with open(f'{DRIVE_EA}/../best_configs.json') as f:
    configs = json.load(f)
STEER_LAYER = int(configs['humor']['layer'])
print(f'STEER_LAYER={STEER_LAYER}  (from best_configs humor layer)')
print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

def norm_t(v): return v / v.norm().clamp(min=1e-8)

depth_axis   = norm_t(torch.tensor(np.load(f'{DRIVE_EA}/axis_depth.npy'),      dtype=torch.bfloat16).to(DEVICE))
arousal_axis = norm_t(torch.tensor(np.load(f'{DRIVE_EA}/axis_arousal.npy'),    dtype=torch.bfloat16).to(DEVICE))
dark_axis    = norm_t(torch.tensor(np.load(f'{DRIVE_EA}/axis_darkbright.npy'), dtype=torch.bfloat16).to(DEVICE))
print('✅ Axes loaded (unit norm)')

EVAL_TEMP   = float(configs.get('best_temp', 0.5))
N_SAMPLES   = 20
MULTIPLIERS = [3.0, 6.0, 9.0]

def make_hook(axis_vec, mult):
    def fn(resid_pre, hook):
        proj = (resid_pre @ axis_vec).unsqueeze(-1) * axis_vec
        return (resid_pre - proj) + axis_vec * mult
    return [(f'blocks.{STEER_LAYER}.hook_resid_pre', fn)]

neutral_prompts = [
    "Tell me what you see right now.",
    "Describe what is around you.",
    "What is on your mind?",
    "Tell me something.",
    "What do you notice?",
    "Describe this moment.",
    "What are you thinking about?",
    "Tell me what you feel.",
    "Describe what is happening.",
    "What comes to mind right now?",
    "Say something.",
    "What do you observe?",
    "Describe your current state.",
    "Tell me anything.",
    "What would you like to say?",
    "Describe what you experience.",
    "What is present right now?",
    "Tell me what matters.",
    "Describe this.",
    "What do you want to say?",
]

experiments = [
    {'axis_name':'depth',      'axis_vec':depth_axis,
     'quality':'emotional depth and richness',
     'pos_desc':'deep emotional richness, resonance',
     'neg_desc':'cold clinical detachment, flatness'},
    {'axis_name':'arousal',    'axis_vec':arousal_axis,
     'quality':'energy and arousal level',
     'pos_desc':'high energy, frenzied, intense',
     'neg_desc':'calm, serene, low energy'},
    {'axis_name':'darkbright', 'axis_vec':dark_axis,
     'quality':'dark narrative tone',
     'pos_desc':'dark, horror-like, dreadful',
     'neg_desc':'bright, joyful, light'},
]

# ── Generation ────────────────────────────────────────────────────
all_results = []
print('\n' + '='*55)
print('GENERATION PHASE')
print('='*55)

for exp in experiments:
    axis_name = exp['axis_name']; axis_vec = exp['axis_vec']
    print(f'\nAXIS: {axis_name.upper()}')

    # Baseline
    print('  baseline...', end='', flush=True)
    baseline_texts = []
    for i in range(N_SAMPLES):
        prompt = neutral_prompts[i % len(neutral_prompts)]
        out = model.generate(prompt, max_new_tokens=60, temperature=EVAL_TEMP,
                             verbose=False, prepend_bos=False)
        gen = out[len(prompt):].strip()
        baseline_texts.append(gen)
        print(f'  [{i+1}] baseline: "{gen[:60]}"', flush=True)
    all_results.append({'axis':axis_name,'direction':'baseline','mult':0,
                        'texts':baseline_texts,'quality':exp['quality']})

    for direction in [+1, -1]:
        dir_label = 'positive' if direction > 0 else 'negative'
        dir_desc  = exp['pos_desc'] if direction > 0 else exp['neg_desc']
        for mult in MULTIPLIERS:
            hooks = make_hook(axis_vec, direction * mult)
            print(f'\n  {dir_label} mult={mult}  ({dir_desc[:40]})...')
            texts = []
            for i in range(N_SAMPLES):
                prompt = neutral_prompts[i % len(neutral_prompts)]
                with model.hooks(fwd_hooks=hooks):
                    out = model.generate(prompt, max_new_tokens=60,
                                         temperature=EVAL_TEMP, verbose=False, prepend_bos=False)
                gen = out[len(prompt):].strip()
                texts.append(gen)
                print(f'    [{i+1:2d}/{N_SAMPLES}] "{gen[:80]}"', flush=True)
            all_results.append({'axis':axis_name,'direction':dir_label,
                                'mult':mult,'texts':texts,'quality':exp['quality']})

print('\n✅ Generation complete')

# ── DeepSeek scoring ──────────────────────────────────────────────
def score_quality(text, quality, direction):
    prompt = f"""Rate the following text on ONE dimension only.
Dimension: {quality}
Rate how much it shows {direction} {quality} (0=not at all, 100=extremely).
Text: "{text[:400]}"
Respond with ONLY a number 0-100."""
    headers = {'Authorization': f'Bearer {DEEPSEEK_API_KEY}', 'Content-Type': 'application/json'}
    body    = {'model':'deepseek-chat',
               'messages':[{'role':'user','content':prompt}],
               'max_tokens':5,'temperature':0.0}
    for attempt in range(3):
        try:
            resp = requests.post('https://api.deepseek.com/chat/completions',
                                 headers=headers, json=body, timeout=15)
            if resp.status_code == 200:
                raw = resp.json()['choices'][0]['message']['content'].strip()
                return float(''.join(c for c in raw if c.isdigit() or c=='.'))
            elif resp.status_code == 429: time.sleep(5*(attempt+1))
        except: time.sleep(3)
    return None

print('\n' + '='*55)
print('DEEPSEEK SCORING PHASE')
print('='*55)

scored = []
for cond in tqdm(all_results, desc='Scoring conditions'):
    scores    = []
    rate_dir  = 'high' if cond['direction'] in ['positive','baseline'] else 'low'
    for j, text in enumerate(cond['texts'][:10]):
        s = score_quality(text, cond['quality'], rate_dir)
        time.sleep(0.3)
        if s is not None:
            scores.append(s)
            print(f'  {cond["axis"]} {cond["direction"]} mult={cond["mult"]} [{j+1}/10]: score={s:.1f}', flush=True)
    scored.append({
        'axis':        cond['axis'],
        'direction':   cond['direction'],
        'mult':        cond['mult'],
        'quality':     cond['quality'],
        'mean_score':  float(np.mean(scores)) if scores else 0.0,
        'std_score':   float(np.std(scores))  if scores else 0.0,
        'n_scored':    len(scores),
        'sample_text': cond['texts'][0][:150],
    })

df_scored = pd.DataFrame(scored)
df_scored.to_csv(f'{DRIVE_EA}/phase4_validation.csv', index=False)

# Results table
print(f'\n{"="*60}')
print('CAUSAL VALIDATION RESULTS')
print(f'{"="*60}')
for axis in ['depth','arousal','darkbright']:
    print(f'\n  {axis.upper()}:')
    sub = df_scored[df_scored['axis']==axis].sort_values('mult')
    for _, row in sub.iterrows():
        bar = '█' * int(row['mean_score']/10)
        print(f'  {row["direction"]:10} mult={row["mult"]:4.1f}: {row["mean_score"]:5.1f}/100  {bar}')
    pos9 = df_scored[(df_scored['axis']==axis)&(df_scored['direction']=='positive')&(df_scored['mult']==9.0)]
    if len(pos9):
        print(f'  Sample (pos×9): "{pos9.iloc[0]["sample_text"][:100]}"')

# Plot
fig, axes_plt = plt.subplots(1, 3, figsize=(15, 5))
for idx, axis in enumerate(['depth','arousal','darkbright']):
    ax  = axes_plt[idx]
    sub = df_scored[df_scored['axis']==axis]
    for direction, color, marker in [('positive','#e74c3c','o'),
                                      ('negative','#3498db','s'),
                                      ('baseline','#95a5a6','^')]:
        rows = sub[sub['direction']==direction].sort_values('mult')
        if len(rows):
            ax.plot(rows['mult'], rows['mean_score'],
                    color=color, marker=marker, lw=2, label=direction)
            ax.fill_between(rows['mult'],
                            rows['mean_score']-rows['std_score'],
                            rows['mean_score']+rows['std_score'],
                            color=color, alpha=0.15)
    ax.set_xlabel('Multiplier'); ax.set_ylabel('DeepSeek score (0-100)')
    ax.set_ylim(0, 105); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{DRIVE_EA}/phase4_validation.pdf', bbox_inches='tight')
plt.close()

# Summary
summary = {}
for axis in ['depth','arousal','darkbright']:
    sub      = df_scored[df_scored['axis']==axis]
    baseline = float(sub[sub['direction']=='baseline']['mean_score'].values[0]) if len(sub[sub['direction']=='baseline']) else 0.0
    pos_max  = float(sub[sub['direction']=='positive']['mean_score'].max())
    neg_mean = float(sub[sub['direction']=='negative']['mean_score'].mean())
    summary[axis] = {'baseline':baseline,'pos_max':pos_max,'neg_mean':neg_mean}

with open(f'{DRIVE_EA}/phase4_summary.json','w') as f:
    json.dump(summary, f, indent=2)

print(f'\n{"━"*55}')
print(f'SCRIPT 10 SUMMARY — Qwen/Qwen3-4B')
print(f'{"━"*55}')
for axis, vals in summary.items():
    effect = vals['pos_max'] - vals['baseline']
    print(f'  {axis:12}: baseline={vals["baseline"]:.1f}  pos_max={vals["pos_max"]:.1f}  effect=+{effect:.1f}')
print(f'\n✅ Script 10 complete. Saved:')
for fname in ['phase4_validation.csv','phase4_validation.pdf','phase4_summary.json']:
    print(f'   {DRIVE_EA}/{fname}')
print(f'\n🎉 ALL 10 SCRIPTS COMPLETE FOR QWEN3-4B')
