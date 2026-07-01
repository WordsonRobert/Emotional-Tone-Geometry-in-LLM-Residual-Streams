#!/usr/bin/env python3
# SCRIPT 03 — meta-llama/Llama-3.2-3B — cuda:1
import os,gc,json,math,torch,warnings,time
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from scipy import stats as scipy_stats
from transformer_lens import HookedTransformer
import warnings; warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
EMOTIONS=['humor','melancholy','horror']
DRIVE='/DATA/24ms190/steering_llama'; MODEL_PATH='/DATA/24ms190/steering_llama/model'; DEVICE='cuda:1'

try: del model
except NameError: pass
gc.collect(); torch.cuda.empty_cache()
model=HookedTransformer.from_pretrained('meta-llama/Llama-3.2-3B',dtype=torch.bfloat16,device=DEVICE,fold_ln=False,center_unembed=False,move_to_device=True,cache_dir=MODEL_PATH)
model.eval()
# Llama — no chat template fix needed
N_LAYERS=model.cfg.n_layers; D_MODEL=model.cfg.d_model; PROBE_LAYER=N_LAYERS-2
print(f'N_LAYERS={N_LAYERS} D_MODEL={D_MODEL} PROBE_LAYER={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

df_humor=pd.read_csv(f'{DRIVE}/humor_master.csv'); df_mel=pd.read_csv(f'{DRIVE}/melancholy_master.csv'); df_hor=pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor,df_mel,df_hor]:
    col=next((c for c in df.columns if c.lower() in ['text','content','post','body']),df.columns[0])
    if col!='text': df.rename(columns={col:'text'},inplace=True)
    df['text']=df['text'].fillna('').astype(str).str.strip()

w_hm=torch.load(f'{DRIVE}/probe_humor_vs_melancholy.pt').float().numpy()
w_hh=torch.load(f'{DRIVE}/probe_humor_vs_horror.pt').float().numpy()
sv_raw=torch.load(f'{DRIVE}/steering_vecs.pt'); sv={k:v.to(DEVICE).to(torch.bfloat16) for k,v in sv_raw.items()}
with open(f'{DRIVE}/best_configs.json') as f: CONFIGS=json.load(f)
PROBE_FOR={'humor':w_hm,'melancholy':w_hm,'horror':w_hh}; PROBE_SIGN={'humor':+1,'melancholy':-1,'horror':-1}
print('CONFIGS:')
for em in EMOTIONS: print(f'  {em:12}: L{CONFIGS[em]["layer"]} x {CONFIGS[em]["mult"]}  both={CONFIGS[em]["both"]:.1%}')

def coherence_score(text):
    words=text.strip().split()
    if len(words)<3: return 0.0
    counts=Counter(words)
    if counts.most_common(1)[0][1]/len(words)>0.4: return 0.0
    chars=[c for c in text.lower() if c.isalpha()]
    if len(chars)<10: return 0.0
    cc=Counter(chars); ent=-sum((c/len(chars))*math.log2(c/len(chars)) for c in cc.values())
    if ent<3.5: return 0.0
    BAD=['ss-','ses','sse','sss','thed','bsp','antml']
    def ok(w):
        w=w.strip('.,!?-()').lower(); return len(w)<2 or (any(c in 'aeiou' for c in w) and not any(p in w for p in BAD))
    return sum(ok(w) for w in words)/len(words)

def generate_with_act(prompt,max_new_tokens,temperature,extra_hooks=None):
    captured={}
    hn=f'blocks.{PROBE_LAYER}.hook_resid_post'
    def cap(resid_post,hook): captured['act']=resid_post[0,-1,:].detach().cpu().float()
    hooks=[(hn,cap)]
    if extra_hooks: hooks=extra_hooks+hooks
    with model.hooks(fwd_hooks=hooks):
        out=model.generate(prompt,max_new_tokens=max_new_tokens,temperature=temperature,verbose=False,prepend_bos=False)
    return out,captured.get('act',torch.zeros(D_MODEL)).numpy()

def scores_from_act(act): return {'score_humor':float(act@w_hm),'score_mel':float(act@-w_hm),'score_horror':float(act@-w_hh)}
def norm_v(v): n=np.linalg.norm(v); return v/n if n>1e-8 else v

def build_steer_hooks(emotion):
    cfg=CONFIGS[emotion]; svec=sv[emotion]; layer=int(cfg['layer']); mult=float(cfg['mult'])
    def fn(resid_pre,hook):
        proj=(resid_pre@svec).unsqueeze(-1)*svec; return (resid_pre-proj)+svec*mult
    return [(f'blocks.{layer}.hook_resid_pre',fn)]

def build_steer_hooks_vec(vec,emotion):
    cfg=CONFIGS[emotion]; layer=int(cfg['layer']); mult=float(cfg['mult'])
    def fn(resid_pre,hook):
        proj=(resid_pre@vec).unsqueeze(-1)*vec; return (resid_pre-proj)+vec*mult
    return [(f'blocks.{layer}.hook_resid_pre',fn)]

TEMPERATURES=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]; SWEEP_N=50; SWEEP_TOKENS=30
eval_sets={'humor':df_humor.sample(SWEEP_N,random_state=2026).reset_index(drop=True),
            'melancholy':df_mel.sample(SWEEP_N,random_state=2026).reset_index(drop=True),
            'horror':df_hor.sample(min(SWEEP_N,len(df_hor)),random_state=2026).reset_index(drop=True)}

SWEEP_CKPT=f'{DRIVE}/temp_sweep_checkpoint.json'; SWEEP_CSV=f'{DRIVE}/temp_sweep_results.csv'
if os.path.exists(SWEEP_CSV) and os.path.exists(SWEEP_CKPT):
    sweep_rows=pd.read_csv(SWEEP_CSV).to_dict('records')
    with open(SWEEP_CKPT) as f: done_keys=set(json.load(f)['done'])
    print(f'Resumed: {len(done_keys)}/33 done')
else:
    sweep_rows=[]; done_keys=set(); print('Starting fresh: 33 jobs')

all_jobs=[(t,em) for t in TEMPERATURES for em in EMOTIONS]
with tqdm(total=len(all_jobs),desc='Temp sweep') as pbar:
    for temp,emotion in all_jobs:
        key=f'{temp:.1f}_{emotion}'
        if key in done_keys: pbar.update(1); continue
        actual_t=1e-10 if temp==0.0 else temp
        steer_hooks=build_steer_hooks(emotion); w=PROBE_FOR[emotion]; sign=PROBE_SIGN[emotion]
        df_e=eval_sets[emotion]; t_job=time.time()
        for _,row in df_e.iterrows():
            p=str(row['text'])[:120]
            base_out,base_act=generate_with_act(p,SWEEP_TOKENS,actual_t)
            base_ps=sign*float(base_act@w)
            ste_out,ste_act=generate_with_act(p,SWEEP_TOKENS,actual_t,extra_hooks=steer_hooks)
            ste_ps=sign*float(ste_act@w); ste_gen=ste_out[len(p):].strip()
            coh=coherence_score(ste_gen); scores=scores_from_act(ste_act)
            pred_em=max(scores,key=scores.get).replace('score_','').replace('mel','melancholy')
            sweep_rows.append({'temperature':temp,'emotion_target':emotion,'prompt':p[:60],
                'base_gen':base_out[len(p):].strip()[:100],'ste_gen':ste_gen[:100],
                'ste_coh':float(coh),'is_coherent':bool(coh>=0.6),'ps_shift':float(ste_ps-base_ps),
                'ps_success':bool(ste_ps>base_ps),'both':bool((ste_ps>base_ps)and(coh>=0.6)),
                'score_humor':scores['score_humor'],'score_mel':scores['score_mel'],
                'score_horror':scores['score_horror'],'pred_emotion':pred_em,'pred_correct':bool(pred_em==emotion)})
        elapsed=time.time()-t_job
        tqdm.write(f'  {key}: {elapsed/60:.1f} min  ({elapsed/len(df_e):.1f}s/sample)')
        done_keys.add(key); pd.DataFrame(sweep_rows).to_csv(SWEEP_CSV,index=False)
        with open(SWEEP_CKPT,'w') as f: json.dump({'done':list(done_keys)},f)
        pbar.update(1)

df_sweep=pd.read_csv(SWEEP_CSV)
print(f'\nTemp sweep complete: {len(df_sweep)} rows')
print(f'\n{"T":>5} {"h_both":>8} {"m_both":>8} {"ho_both":>8} {"h_pred":>8} {"m_pred":>8} {"ho_pred":>8} {"avg_diag":>10}')
for t in TEMPERATURES:
    sub=df_sweep[df_sweep['temperature']==t]
    if len(sub)==0: continue
    v={em:{'b':sub[sub['emotion_target']==em]['both'].mean(),'p':sub[sub['emotion_target']==em]['pred_correct'].mean()} for em in EMOTIONS}
    ad=np.mean([v[em]['p'] for em in EMOTIONS])
    print(f'{t:>5.1f} {v["humor"]["b"]:>8.1%} {v["melancholy"]["b"]:>8.1%} {v["horror"]["b"]:>8.1%} {v["humor"]["p"]:>8.1%} {v["melancholy"]["p"]:>8.1%} {v["horror"]["p"]:>8.1%} {ad:>10.1%}')

BEST_TEMP=float(max(TEMPERATURES,key=lambda t:np.mean([df_sweep[(df_sweep['temperature']==t)&(df_sweep['emotion_target']==em)]['pred_correct'].mean() for em in EMOTIONS if len(df_sweep[(df_sweep['temperature']==t)&(df_sweep['emotion_target']==em)])>0])))
print(f'\nBEST_TEMP = {BEST_TEMP}')
CONFIGS['best_temp']=BEST_TEMP
with open(f'{DRIVE}/best_configs.json','w') as f: json.dump(CONFIGS,f,indent=2)
df_sweep[df_sweep['temperature']==BEST_TEMP].to_csv(f'{DRIVE}/simplex_results_final.csv',index=False)

fig,axes_grid=plt.subplots(3,4,figsize=(18,12)); axes_flat=axes_grid.flatten()
for i,t in enumerate(TEMPERATURES):
    ax=axes_flat[i]; sub=df_sweep[df_sweep['temperature']==t]
    if len(sub)==0: ax.set_visible(False); continue
    conf=np.zeros((3,3))
    for ri,te in enumerate(EMOTIONS):
        s=sub[sub['emotion_target']==te]
        for ci,pe in enumerate(EMOTIONS): conf[ri,ci]=(s['pred_emotion']==pe).sum()/max(len(s),1)
    sns.heatmap(conf,annot=True,fmt='.0%',cmap='Blues',xticklabels=['h','m','ho'],yticklabels=['h','m','ho'],ax=ax,vmin=0,vmax=1,cbar=False,annot_kws={'size':8})
    ax.set_title(f'T={t:.1f}  diag={np.diag(conf).mean():.0%}',fontsize=9)
axes_flat[-1].set_visible(False); plt.tight_layout()
plt.savefig(f'{DRIVE}/temp_sweep_confusion_grid.pdf',bbox_inches='tight'); plt.close()

fig,axes_m=plt.subplots(1,3,figsize=(16,5))
for idx,em in enumerate(EMOTIONS):
    ax=axes_m[idx]; sub=df_sweep[df_sweep['emotion_target']==em]
    for metric,color,label in [('both','#e74c3c','both'),('pred_correct','#2ecc71','pred'),('is_coherent','#3498db','coh')]:
        vals=[sub[sub['temperature']==t][metric].mean() for t in TEMPERATURES]
        ax.plot(TEMPERATURES,vals,marker='o',color=color,label=label,lw=2)
    ax.axvline(BEST_TEMP,color='gray',ls='--',lw=1); ax.set_xlabel('Temperature'); ax.set_ylim(0,1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{DRIVE}/temp_sweep_metrics_plot.pdf',bbox_inches='tight'); plt.close()
with open(f'{DRIVE}/temp_sweep_summary.json','w') as f: json.dump({'best_temp_pred':BEST_TEMP},f,indent=2)
print('✅ Cell 6 complete')

# ── CELL 7: Main eval ──────────────────────────────────────────────
eval_sets_full={'humor':df_humor.sample(100,random_state=2026).reset_index(drop=True),
                 'melancholy':df_mel.sample(100,random_state=2026).reset_index(drop=True),
                 'horror':df_hor.sample(min(100,len(df_hor)),random_state=2026).reset_index(drop=True)}
EVAL_TEMP=BEST_TEMP; actual_t=EVAL_TEMP; EVAL_TOKENS=60
print(f'\nMain eval: T={EVAL_TEMP}, n=100, max_new_tokens={EVAL_TOKENS}')
EVAL_CKPT=f'{DRIVE}/main_eval_checkpoint.json'; EVAL_CSV=f'{DRIVE}/simplex_results_T_best.csv'
if os.path.exists(EVAL_CSV) and os.path.exists(EVAL_CKPT):
    eval_rows=pd.read_csv(EVAL_CSV).to_dict('records')
    with open(EVAL_CKPT) as f: eval_done=set(json.load(f)['done'])
    print(f'  Resumed: {eval_done}')
else:
    eval_rows=[]; eval_done=set()

for emotion in EMOTIONS:
    if emotion in eval_done: continue
    steer_hooks=build_steer_hooks(emotion); w=PROBE_FOR[emotion]; sign=PROBE_SIGN[emotion]
    df_e=eval_sets_full[emotion]; t_em=time.time()
    for _,row in tqdm(df_e.iterrows(),total=len(df_e),desc=f'  {emotion}',leave=False):
        p=str(row['text'])[:120]
        base_out,base_act=generate_with_act(p,EVAL_TOKENS,actual_t)
        base_ps=sign*float(base_act@w)
        ste_out,ste_act=generate_with_act(p,EVAL_TOKENS,actual_t,extra_hooks=steer_hooks)
        ste_ps=sign*float(ste_act@w); ste_gen=ste_out[len(p):].strip()
        coh=coherence_score(ste_gen); scores=scores_from_act(ste_act)
        pred_em=max(scores,key=scores.get).replace('score_','').replace('mel','melancholy')
        eval_rows.append({'temperature':EVAL_TEMP,'emotion_target':emotion,'prompt':p[:60],
            'base_gen':base_out[len(p):].strip()[:100],'ste_gen':ste_gen[:100],
            'ste_coh':float(coh),'is_coherent':bool(coh>=0.6),'ps_shift':float(ste_ps-base_ps),
            'ps_success':bool(ste_ps>base_ps),'both':bool((ste_ps>base_ps)and(coh>=0.6)),
            'score_humor':scores['score_humor'],'score_mel':scores['score_mel'],
            'score_horror':scores['score_horror'],'pred_emotion':pred_em,'pred_correct':bool(pred_em==emotion)})
    print(f'  {emotion}: {(time.time()-t_em)/60:.1f} min')
    eval_done.add(emotion); pd.DataFrame(eval_rows).to_csv(EVAL_CSV,index=False)
    with open(EVAL_CKPT,'w') as f: json.dump({'done':list(eval_done)},f)

df_results=pd.read_csv(EVAL_CSV)
conf=np.zeros((3,3))
for i,te in enumerate(EMOTIONS):
    sub=df_results[df_results['emotion_target']==te]
    for j,pe in enumerate(EMOTIONS): conf[i,j]=(sub['pred_emotion']==pe).sum()/max(len(sub),1)
avg_diag=float(np.diag(conf).mean())
print(f'\nConfusion matrix (T={EVAL_TEMP}):')
print(f'  {"":12} {"→humor":>10} {"→mel":>10} {"→horror":>10}')
for i,em in enumerate(EMOTIONS): print(f'  {em:12} {conf[i,0]:>10.1%} {conf[i,1]:>10.1%} {conf[i,2]:>10.1%}')
print(f'  Avg diagonal: {avg_diag:.1%}')
for em in EMOTIONS:
    sub=df_results[df_results['emotion_target']==em]
    print(f'  {em:12} both={sub["both"].mean():.1%}  pred={sub["pred_correct"].mean():.1%}  coh={sub["is_coherent"].mean():.1%}  shift={sub["ps_shift"].mean():+.4f}')

fig,ax=plt.subplots(figsize=(7,5))
sns.heatmap(conf,annot=True,fmt='.1%',xticklabels=['→humor','→mel','→horror'],yticklabels=['humor','mel','horror'],cmap='Blues',ax=ax,vmin=0,vmax=1)
plt.tight_layout(); plt.savefig(f'{DRIVE}/probe_confusion_matrix_best.pdf',bbox_inches='tight'); plt.close()

VEC_N=50; vec_sets={}
for emotion in EMOTIONS:
    md=sv[emotion].cpu().float()
    pw=torch.tensor(PROBE_FOR[emotion]*PROBE_SIGN[emotion],dtype=torch.float32); pw=pw/pw.norm().clamp(min=1e-8)
    rnd=torch.randn_like(md); rnd=rnd/rnd.norm().clamp(min=1e-8)
    vec_sets[emotion]={'mean_diff':md.to(DEVICE).to(torch.bfloat16),'probe_w':pw.to(DEVICE).to(torch.bfloat16),'random':rnd.to(DEVICE).to(torch.bfloat16)}
eval_prompts_vc={'humor':df_humor.sample(VEC_N,random_state=77)['text'].tolist(),
                  'melancholy':df_mel.sample(VEC_N,random_state=77)['text'].tolist(),
                  'horror':df_hor.sample(min(VEC_N,len(df_hor)),random_state=77)['text'].tolist()}
vc_rows=[]; vec_names=['mean_diff','probe_w','random']
print(f'\n  {"emotion":12} {"vec":12} {"both":>8} {"mean_shift":>12}'); print('  '+'─'*46)
for emotion in EMOTIONS:
    for vec_name,vec in vec_sets[emotion].items():
        w=PROBE_FOR[emotion]; sign=PROBE_SIGN[emotion]; hooks=build_steer_hooks_vec(vec,emotion); results=[]
        for prompt in eval_prompts_vc[emotion]:
            p=prompt[:120]
            _,base_act=generate_with_act(p,60,actual_t); base_ps=sign*float(base_act@w)
            ste_out,ste_act=generate_with_act(p,60,actual_t,extra_hooks=hooks); ste_ps=sign*float(ste_act@w)
            coh=coherence_score(ste_out[len(p):].strip())
            results.append({'both':bool((ste_ps>base_ps)and(coh>=0.6)),'ps_shift':float(ste_ps-base_ps),'vec':vec_name,'emotion':emotion})
        br=np.mean([r['both'] for r in results]); sm=np.mean([r['ps_shift'] for r in results])
        print(f'  {emotion:12} {vec_name:12} {br:>8.1%} {sm:>+12.4f}',flush=True); vc_rows.extend(results)
df_vc=pd.DataFrame(vc_rows); df_vc.to_csv(f'{DRIVE}/vector_comparison.csv',index=False)
n_comp=len(EMOTIONS)*len(list(combinations(vec_names,2))); pval_rows=[]
for emotion in EMOTIONS:
    sub=df_vc[df_vc['emotion']==emotion]
    for v1,v2 in combinations(vec_names,2):
        s1=sub[sub['vec']==v1]['ps_shift'].values; s2=sub[sub['vec']==v2]['ps_shift'].values
        stat,p=scipy_stats.mannwhitneyu(s1,s2,alternative='two-sided'); p_adj=min(p*n_comp,1.0)
        sig='***' if p_adj<0.001 else ('**' if p_adj<0.01 else ('*' if p_adj<0.05 else 'ns'))
        pval_rows.append({'emotion':emotion,'v1':v1,'v2':v2,'U':float(stat),'p_raw':float(p),'p_bonf':float(p_adj),'sig':sig})
        print(f'  {emotion:12} {v1:12} vs {v2:12}  p={p_adj:.4f} {sig}')
pd.DataFrame(pval_rows).to_csv(f'{DRIVE}/vector_comparison_stats.csv',index=False)
fig,axes_vc=plt.subplots(1,3,figsize=(15,5)); colors={'mean_diff':'#2ecc71','probe_w':'#3498db','random':'#95a5a6'}
for idx,emotion in enumerate(EMOTIONS):
    ax=axes_vc[idx]; sub=df_vc[df_vc['emotion']==emotion]
    for vn in vec_names: ax.bar(vn,sub[sub['vec']==vn]['both'].mean(),color=colors[vn],alpha=0.85)
    ax.set_ylim(0,1.1); ax.set_ylabel('Both rate'); ax.axhline(0.5,color='gray',ls='--',lw=1); ax.grid(axis='y',alpha=0.3)
plt.tight_layout(); plt.savefig(f'{DRIVE}/vector_comparison.pdf',bbox_inches='tight'); plt.close()

cell7={'model':'meta-llama/Llama-3.2-3B','eval_temp':EVAL_TEMP,'best_temp':BEST_TEMP,'avg_diagonal':avg_diag,
       'per_emotion':{em:{'both':float(df_results[df_results['emotion_target']==em]['both'].mean()),
           'pred':float(df_results[df_results['emotion_target']==em]['pred_correct'].mean()),
           'coherent':float(df_results[df_results['emotion_target']==em]['is_coherent'].mean()),
           'mean_shift':float(df_results[df_results['emotion_target']==em]['ps_shift'].mean())}
           for em in EMOTIONS},'configs':CONFIGS}
with open(f'{DRIVE}/cell7_summary.json','w') as f: json.dump(cell7,f,indent=2)

print(f'\n{"━"*55}\nSCRIPT 3 SUMMARY — meta-llama/Llama-3.2-3B\n{"━"*55}')
print(f'  BEST_TEMP: {BEST_TEMP}  avg_diag: {avg_diag:.1%}')
for em in EMOTIONS:
    sub=df_results[df_results['emotion_target']==em]
    print(f'  {em:12}: both={sub["both"].mean():.1%}  pred={sub["pred_correct"].mean():.1%}  coh={sub["is_coherent"].mean():.1%}  shift={sub["ps_shift"].mean():+.2f}')
print(f'\n✅ Script 3 complete\nNext: python3 04_head_ablation_llama_v2.py')
