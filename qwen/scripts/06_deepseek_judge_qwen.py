#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# SCRIPT 6 (QWEN3-4B): DEEPSEEK LLM JUDGE
# ═══════════════════════════════════════════════════════════════════
# Model:   Qwen/Qwen3-4B
# Folder:  /DATA/24ms190/steering_qwen
# NO GPU needed — pure API calls
# Runtime: ~20 min
#
# Sends each steered generation to DeepSeek-V3 which rates it
# 1-5 on humor/melancholy/horror and picks dominant emotion.
# ═══════════════════════════════════════════════════════════════════

import os, json, time, re
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from scipy import stats as scipy_stats
from tqdm import tqdm

DRIVE    = '/DATA/24ms190/steering_qwen'
EMOTIONS = ['humor', 'melancholy', 'horror']

# ── API key ───────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = input('Paste DEEPSEEK_API_KEY: ').strip()
if not DEEPSEEK_API_KEY:
    raise ValueError('DEEPSEEK_API_KEY required')

DEEPSEEK_MODEL = 'deepseek-chat'

def deepseek_judge(text, max_retries=3):
    prompt = f"""Rate the following text on three emotional dimensions.
Respond ONLY with a JSON object, no explanation, no markdown, no code fences.

Text: "{text[:400]}"

Respond exactly like this:
{{"humor": 3, "melancholy": 1, "horror": 2, "dominant": "humor", "confidence": 0.8}}

Rules:
- humor: 1 (not funny) to 5 (very funny/absurd/witty)
- melancholy: 1 (not sad) to 5 (very melancholic/sorrowful/gloomy)
- horror: 1 (not scary) to 5 (very scary/dreadful/horrifying)
- dominant: whichever of humor/melancholy/horror scores highest
- confidence: 0.0 to 1.0"""

    headers = {'Authorization': f'Bearer {DEEPSEEK_API_KEY}', 'Content-Type': 'application/json'}
    body = {
        'model': DEEPSEEK_MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 80, 'temperature': 0.0,
        'response_format': {'type': 'json_object'},
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post('https://api.deepseek.com/chat/completions',
                                 headers=headers, json=body, timeout=30)
            if resp.status_code == 200:
                raw   = resp.json()['choices'][0]['message']['content'].strip()
                match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    if all(k in parsed for k in ['humor','melancholy','horror','dominant']):
                        return parsed
            elif resp.status_code == 429:
                time.sleep(5*(attempt+1))
            else:
                time.sleep(2)
        except Exception as e:
            print(f'  API error: {e}', flush=True)
            time.sleep(3)
    return None

# Test API
print(f'Testing DeepSeek API...', end='', flush=True)
test = deepseek_judge("Why did the chicken cross the road? To get to the other side!")
if test: print(f' ✅ {test}')
else: raise RuntimeError('API test failed')

# Load data
df = pd.read_csv(f'{DRIVE}/simplex_results_T_best.csv')
print(f'\n{len(df)} rows  model={DEEPSEEK_MODEL}')

CHECKPOINT = f'{DRIVE}/task_nim_eval_checkpoint.json'
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f: nim_rows = json.load(f)
    done_idx = set(r['row_idx'] for r in nim_rows)
    print(f'Resumed: {len(nim_rows)} rows done')
else:
    nim_rows = []; done_idx = set()
    print('Starting fresh')

failed = 0
for idx, row in df.iterrows():
    if idx in done_idx: continue
    text   = str(row.get('ste_gen',''))[:400]
    result = deepseek_judge(text)
    time.sleep(0.5)

    entry = {
        'row_idx':        idx,
        'emotion_target': row['emotion_target'],
        'ps_shift':       float(row['ps_shift']),
        'both':           bool(row['both']),
        'pred_correct':   bool(row['pred_correct']),
        'ste_gen':        text[:100],
    }
    if result:
        entry.update({
            'nim_humor':      int(result.get('humor',0)),
            'nim_mel':        int(result.get('melancholy',0)),
            'nim_horror':     int(result.get('horror',0)),
            'nim_dominant':   str(result.get('dominant','')),
            'nim_confidence': float(result.get('confidence',0.0)),
        })
        print(f'  [{idx+1:3d}/{len(df)}] {row["emotion_target"]:12} → dominant={result.get("dominant","")}  '
              f'h={result.get("humor",0)} m={result.get("melancholy",0)} ho={result.get("horror",0)}  '
              f'conf={result.get("confidence",0):.2f}', flush=True)
    else:
        failed += 1
        entry.update({'nim_humor':-1,'nim_mel':-1,'nim_horror':-1,
                      'nim_dominant':'failed','nim_confidence':0.0})
        print(f'  [{idx+1:3d}/{len(df)}] FAILED', flush=True)

    nim_rows.append(entry)
    if len(nim_rows) % 30 == 0:
        with open(CHECKPOINT,'w') as f: json.dump(nim_rows, f)

with open(CHECKPOINT,'w') as f: json.dump(nim_rows, f)

df_nim = pd.DataFrame(nim_rows)
df_nim = df_nim[df_nim['nim_dominant'] != 'failed'].copy()
df_nim.to_csv(f'{DRIVE}/task_nim_eval.csv', index=False)
print(f'\n✅ {len(df_nim)} valid / {len(nim_rows)} total  {failed} failed')

# Analysis
NIM_COL = {'humor':'nim_humor','melancholy':'nim_mel','horror':'nim_horror'}
print(f'\n{"="*60}')
print('DEEPSEEK JUDGE RESULTS')
print(f'{"="*60}')

print(f'\n1. Mean target scores (1-5):')
for emotion in EMOTIONS:
    sub = df_nim[df_nim['emotion_target']==emotion]
    col = NIM_COL[emotion]
    print(f'  {emotion:12}: {sub[col].mean():.2f} ± {sub[col].std():.2f}  n={len(sub)}')

print(f'\n2. Dominant emotion accuracy:')
overall_correct = []
for emotion in EMOTIONS:
    sub     = df_nim[df_nim['emotion_target']==emotion]
    correct = (sub['nim_dominant']==emotion).mean()
    overall_correct.extend((sub['nim_dominant']==emotion).tolist())
    print(f'  {emotion:12}: {correct:.1%}  ({int(correct*len(sub))}/{len(sub)})')
print(f'  Overall: {np.mean(overall_correct):.1%}')

print(f'\n3. Spearman r (ps_shift vs DeepSeek target):')
for emotion in EMOTIONS:
    sub = df_nim[df_nim['emotion_target']==emotion]
    col = NIM_COL[emotion]
    r, p = scipy_stats.spearmanr(sub['ps_shift'], sub[col])
    sig  = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
    print(f'  {emotion:12}: r={r:+.4f}  p={p:.4f}  {sig}')

print(f'\n4. Cross-validation with RoBERTa:')
rob_path = f'{DRIVE}/task_roberta_eval.csv'
if os.path.exists(rob_path):
    df_rob    = pd.read_csv(rob_path).reset_index()
    df_merged = df_nim.merge(
        df_rob[['emotion_target','roberta_target_score','ps_shift']].reset_index(),
        on=['emotion_target','ps_shift'], how='inner')
    for emotion in EMOTIONS:
        sub = df_merged[df_merged['emotion_target']==emotion]
        nim_col = NIM_COL[emotion]
        if len(sub) > 5:
            r, p = scipy_stats.spearmanr(sub[nim_col], sub['roberta_target_score'])
            sig  = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
            print(f'  {emotion:12}: DeepSeek vs RoBERTa r={r:+.4f}  p={p:.4f}  {sig}')
else:
    print('  task_roberta_eval.csv not found — run Script 5 first')

# Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
score_mat = np.zeros((3,3))
for i, em in enumerate(EMOTIONS):
    sub = df_nim[df_nim['emotion_target']==em]
    score_mat[i] = [sub['nim_humor'].mean(), sub['nim_mel'].mean(), sub['nim_horror'].mean()]
sns.heatmap(score_mat, annot=True, fmt='.2f',
            xticklabels=['humor','mel','horror'],
            yticklabels=['→humor','→mel','→horror'],
            cmap='YlOrRd', ax=axes[0], vmin=1, vmax=5)
accs = [(df_nim[df_nim['emotion_target']==em]['nim_dominant']==em).mean() for em in EMOTIONS]
axes[1].bar(EMOTIONS, accs, color=['#2196F3','#9C27B0','#F44336'], alpha=0.85)
axes[1].set_ylim(0, 1.1); axes[1].set_ylabel('Accuracy')
axes[1].axhline(1/3, color='gray', ls=':', lw=1); axes[1].grid(axis='y', alpha=0.3)
for i, acc in enumerate(accs):
    axes[1].text(i, acc+0.02, f'{acc:.1%}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{DRIVE}/task_nim_eval.pdf', bbox_inches='tight')
plt.close()

cell13_summary = {
    'judge_model':       DEEPSEEK_MODEL,
    'n_valid':           len(df_nim),
    'n_failed':          failed,
    'overall_accuracy':  float(np.mean(overall_correct)),
    'per_emotion_accuracy': {
        em: float((df_nim[df_nim['emotion_target']==em]['nim_dominant']==em).mean())
        for em in EMOTIONS
    },
    'mean_scores': {
        em: {
            'humor':      float(df_nim[df_nim['emotion_target']==em]['nim_humor'].mean()),
            'melancholy': float(df_nim[df_nim['emotion_target']==em]['nim_mel'].mean()),
            'horror':     float(df_nim[df_nim['emotion_target']==em]['nim_horror'].mean()),
        } for em in EMOTIONS
    },
}
with open(f'{DRIVE}/cell13_summary.json','w') as f:
    json.dump(cell13_summary, f, indent=2)

print(f'\n{"━"*55}')
print(f'SCRIPT 6 SUMMARY — Qwen/Qwen3-4B')
print(f'{"━"*55}')
print(f'  Overall accuracy: {np.mean(overall_correct):.1%}')
for em in EMOTIONS:
    acc = (df_nim[df_nim['emotion_target']==em]['nim_dominant']==em).mean()
    print(f'  {em:12}: {acc:.1%}')
print(f'\n✅ Script 6 complete. Saved to {DRIVE}:')
for fname in ['task_nim_eval.csv','task_nim_eval.pdf','cell13_summary.json']:
    print(f'   {fname}')
print(f'\nNext: python3 07_emotional_tone_rollouts_qwen.py')
