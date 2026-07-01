#!/usr/bin/env python3
# SCRIPT 04 — meta-llama/Llama-3.2-3B — cuda:1
import os,gc,json,math,torch,re,time,warnings
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats
from transformer_lens import HookedTransformer
warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
EMOTIONS=['humor','melancholy','horror']
DRIVE='/DATA/24ms190/steering_llama'; MODEL_PATH='/DATA/24ms190/steering_llama/model'; DEVICE='cuda:1'

try: del model
except NameError: pass
gc.collect(); torch.cuda.empty_cache()
model=HookedTransformer.from_pretrained('meta-llama/Llama-3.2-3B',dtype=torch.bfloat16,device=DEVICE,fold_ln=False,center_unembed=False,move_to_device=True,cache_dir=MODEL_PATH)
model.eval()
N_LAYERS=model.cfg.n_layers; D_MODEL=model.cfg.d_model; N_HEADS=model.cfg.n_heads; PROBE_LAYER=N_LAYERS-2
print(f'N_LAYERS={N_LAYERS} D_MODEL={D_MODEL} N_HEADS={N_HEADS} PROBE_LAYER={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

df_humor=pd.read_csv(f'{DRIVE}/humor_master.csv'); df_mel=pd.read_csv(f'{DRIVE}/melancholy_master.csv'); df_hor=pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor,df_mel,df_hor]:
    col=next((c for c in df.columns if c.lower() in ['text','content','post','body']),df.columns[0])
    if col!='text': df.rename(columns={col:'text'},inplace=True)
    df['text']=df['text'].fillna('').astype(str).str.strip()
probe_weights=torch.load(f'{DRIVE}/probe_weights.pt',map_location='cpu')
with open(f'{DRIVE}/best_configs.json') as f: CONFIGS=json.load(f)

steer_layers=list(set([CONFIGS[em]['layer'] for em in EMOTIONS]))
ablation_layers=sorted(set([0,N_LAYERS//4,N_LAYERS//2,PROBE_LAYER]+steer_layers))
N_ABL=100; n_each=N_ABL//2
rng=np.random.default_rng(42)
h_idx=rng.choice(len(df_humor),size=n_each,replace=False); m_idx=rng.choice(len(df_mel),size=n_each,replace=False)
abl_texts=df_humor['text'].iloc[h_idx].tolist()+df_mel['text'].iloc[m_idx].tolist()
abl_labels=[1]*n_each+[0]*n_each
print(f'Ablation layers: {ablation_layers}  ({N_HEADS} heads each)')

def get_acts_at_layer(texts,layer,ablate_head=None):
    acts=[]; hn=f'blocks.{layer}.hook_resid_post'
    for text in texts:
        tokens=model.to_tokens([text],prepend_bos=False)
        if tokens.shape[1]==0: acts.append(np.zeros(D_MODEL,dtype=np.float32)); continue
        hooks=[]
        if ablate_head is not None:
            h=ablate_head
            def make_z(hi):
                def fn(z,hook): z_new=z.clone(); z_new[:,:,hi,:]=0.0; return z_new
                return fn
            hooks=[(f'blocks.{layer}.attn.hook_z',make_z(h))]
        with torch.no_grad():
            if hooks:
                with model.hooks(fwd_hooks=hooks): _,cache=model.run_with_cache(tokens,names_filter=lambda n:hn in n,stop_at_layer=layer+1)
            else: _,cache=model.run_with_cache(tokens,names_filter=lambda n:hn in n,stop_at_layer=layer+1)
        acts.append(cache[hn][0,-1,:].cpu().float().numpy()); del cache; torch.cuda.empty_cache()
    return np.array(acts)

def probe_accuracy(acts,labels,layer):
    w=probe_weights[layer].float().numpy(); preds=(acts@w>0).astype(int)
    return float((preds==np.array(labels)).mean())

CHECKPOINT=f'{DRIVE}/task19_ablation_results.json'
results=json.load(open(CHECKPOINT)) if os.path.exists(CHECKPOINT) else dict()
for layer in tqdm(ablation_layers,desc='Ablation layers'):
    base_key=f'baseline_L{layer}'
    if base_key not in results:
        base_acts=get_acts_at_layer(abl_texts,layer); base_acc=probe_accuracy(base_acts,abl_labels,layer)
        results[base_key]=float(base_acc)
        with open(CHECKPOINT,'w') as f: json.dump(results,f)
        tqdm.write(f'  L{layer:2d} baseline = {base_acc:.4f}')
    else: base_acc=results[base_key]
    for head_idx in range(N_HEADS):
        key=f'L{layer}_H{head_idx}'
        if key in results: continue
        abl_acts=get_acts_at_layer(abl_texts,layer,ablate_head=head_idx)
        abl_acc=probe_accuracy(abl_acts,abl_labels,layer); drop=float(base_acc-abl_acc)
        results[key]={'layer':layer,'head':head_idx,'baseline_acc':float(base_acc),'ablated_acc':float(abl_acc),'drop':drop,'load_bearing':bool(drop>0.02)}
        if drop>0.02: print(f'  ⚠️  L{layer:2d} H{head_idx:2d}: drop={drop:.4f} LOAD-BEARING',flush=True)
        else: print(f'  L{layer:2d} H{head_idx:2d}: drop={drop:+.4f}',flush=True)
    with open(CHECKPOINT,'w') as f: json.dump(results,f)

rows=[v for k,v in results.items() if isinstance(v,dict)]; df_abl=pd.DataFrame(rows)
df_abl.to_csv(f'{DRIVE}/task19_ablation_results.csv',index=False)
critical_heads=df_abl[df_abl['load_bearing']].sort_values('drop',ascending=False)
print(f'\nCritical heads (drop>2%): {len(critical_heads)}')
if len(critical_heads)>0: print(critical_heads[['layer','head','baseline_acc','ablated_acc','drop']].to_string(index=False))

fig,axes_plt=plt.subplots(1,len(ablation_layers),figsize=(4*len(ablation_layers),5))
if len(ablation_layers)==1: axes_plt=[axes_plt]
for idx,layer in enumerate(ablation_layers):
    ax=axes_plt[idx]; sub=df_abl[df_abl['layer']==layer].sort_values('head')
    drops=sub['drop'].values; colors=['#C44E52' if d>0.02 else '#4C72B0' for d in drops]
    ax.bar(range(len(drops)),drops,color=colors,alpha=0.85); ax.axhline(0.02,color='k',lw=1,ls='--')
    ax.set_xlabel('Head'); ax.set_xticks(range(0,N_HEADS,max(1,N_HEADS//8)))
plt.tight_layout(); plt.savefig(f'{DRIVE}/task19_head_ablation.pdf',bbox_inches='tight'); plt.close()
cell9={'ablation_layers':ablation_layers,'n_heads':N_HEADS,'n_texts':N_ABL,'n_critical':int(len(critical_heads)),
       'critical_heads':critical_heads[['layer','head','drop']].to_dict('records') if len(critical_heads)>0 else []}
with open(f'{DRIVE}/cell9_summary.json','w') as f: json.dump(cell9,f,indent=2)
print('✅ Cell 9 complete')

# ── CELL 10: CPU analyses ──────────────────────────────────────────
df_results=pd.read_csv(f'{DRIVE}/simplex_results_T_best.csv')
sv_raw=torch.load(f'{DRIVE}/steering_vecs.pt',map_location='cpu'); sv_np={k:v.float().numpy() for k,v in sv_raw.items()}
def norm_v(v): return v/(np.linalg.norm(v)+1e-8)

conf=np.zeros((3,3))
for i,te in enumerate(EMOTIONS):
    sub=df_results[df_results['emotion_target']==te]
    for j,pe in enumerate(EMOTIONS): conf[i,j]=(sub['pred_emotion']==pe).sum()/max(len(sub),1)
mel_horror_bleed=(conf[1,2]+conf[2,1])/2; humor_other_bleed=(conf[0,1]+conf[0,2]+conf[1,0]+conf[2,0])/4
cos_mat=np.zeros((3,3))
for i,a in enumerate(EMOTIONS):
    for j,b in enumerate(EMOTIONS): cos_mat[i,j]=float(norm_v(sv_np[a])@norm_v(sv_np[b]))
print(f'\nTask 3: mel↔horror bleed={mel_horror_bleed:.1%}  humor↔other={humor_other_bleed:.1%}  ratio={mel_horror_bleed/max(humor_other_bleed,1e-8):.2f}x')
task3={'mel_horror_bleed':float(mel_horror_bleed),'humor_other_bleed':float(humor_other_bleed),
       'bleed_ratio':float(mel_horror_bleed/max(humor_other_bleed,1e-8)),
       'sv_cosines':{f'{a}_{b}':float(cos_mat[i,j]) for i,a in enumerate(EMOTIONS) for j,b in enumerate(EMOTIONS)}}
with open(f'{DRIVE}/task3_binary_vs_threepole_results.json','w') as f: json.dump(task3,f,indent=2)
fig,axes_p=plt.subplots(1,2,figsize=(12,5))
sns.heatmap(conf,annot=True,fmt='.1%',xticklabels=['→humor','→mel','→horror'],yticklabels=['humor','mel','horror'],cmap='Blues',ax=axes_p[0],vmin=0,vmax=1)
sns.heatmap(cos_mat,annot=True,fmt='.3f',xticklabels=EMOTIONS,yticklabels=EMOTIONS,cmap='RdYlGn',center=0,ax=axes_p[1])
plt.tight_layout(); plt.savefig(f'{DRIVE}/task3_binary_vs_threepole.pdf',bbox_inches='tight'); plt.close()
print('✅ Task 3 done')

def extract_surface_features(texts):
    feats=[]
    for text in texts:
        words=text.split(); sents=re.split(r'[.!?]+',text); sents=[s.strip() for s in sents if s.strip()]; chars=list(text)
        feats.append({'ttr':len(set(words))/max(len(words),1),'punct_density':sum(1 for c in chars if c in '.,!?;:')/max(len(chars),1),
            'sent_count':len(sents),'avg_word_len':np.mean([len(w) for w in words]) if words else 0,
            'exclaim_rate':text.count('!')/max(len(words),1),'question_rate':text.count('?')/max(len(words),1),
            'upper_rate':sum(1 for c in chars if c.isupper())/max(len(chars),1),
            'avg_sent_len':np.mean([len(s.split()) for s in sents]) if sents else 0,
            'ellipsis_rate':text.count('...')/max(len(words),1),'digit_rate':sum(1 for c in chars if c.isdigit())/max(len(chars),1)})
    return pd.DataFrame(feats)

all_texts=df_humor['text'].tolist()+df_mel['text'].tolist()+df_hor['text'].tolist()
all_labels=[0]*len(df_humor)+[1]*len(df_mel)+[2]*len(df_hor)
X_surf=StandardScaler().fit_transform(extract_surface_features(all_texts).values); y=np.array(all_labels)
surf_acc=cross_val_score(LogisticRegression(max_iter=1000),X_surf,y,cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)).mean()
act_acc=float(df_results['pred_correct'].mean())
print(f'\nTask 4: surface={surf_acc:.4f}  activation={act_acc:.4f}  gap={act_acc-surf_acc:+.4f}')
with open(f'{DRIVE}/task4_surface_probe_results.json','w') as f: json.dump({'surface_probe_acc':float(surf_acc),'activation_probe_acc':float(act_acc),'gap':float(act_acc-surf_acc)},f,indent=2)
print('✅ Task 4 done')

try:
    from datasets import load_dataset
    clf_ood=LogisticRegression(max_iter=1000); scaler5=StandardScaler()
    X5=scaler5.fit_transform(extract_surface_features(all_texts).values); clf_ood.fit(X5,y); ood=dict()
    ds=load_dataset('Abirate/english_quotes',split='train')
    qt=[r['quote'] for r in ds if isinstance(r['quote'],str) and 30<len(r['quote'])<200][:200]
    ood['humor_ood']={'acc':float((clf_ood.predict(scaler5.transform(extract_surface_features(qt).values))==0).mean()),'n':len(qt)}
    print(f'  Humor OOD: {ood["humor_ood"]["acc"]:.4f}')
    ds2=load_dataset('merve/poetry',split='train')
    sad_kw=['sad','sorrow','grief','loss','pain','lonely','dark','tears','death','despair']
    mel_t=[str(r.get('content',''))[:400] for r in ds2 if any(k in str(r.get('content','')).lower() for k in sad_kw)][:200]
    if len(mel_t)<20: mel_t=[str(r.get('content',''))[:400] for r in ds2 if len(str(r.get('content','')))>30][:200]
    ood['mel_ood']={'acc':float((clf_ood.predict(scaler5.transform(extract_surface_features(mel_t).values))==1).mean()),'n':len(mel_t)}
    print(f'  Mel OOD: {ood["mel_ood"]["acc"]:.4f}')
    ds3=load_dataset('stanfordnlp/imdb',split='test')
    hor_t=[r['text'][:400] for r in ds3 if r['label']==0 and len(r['text'])>50][:200]
    ood['horror_ood']={'acc':float((clf_ood.predict(scaler5.transform(extract_surface_features(hor_t).values))==2).mean()),'n':len(hor_t)}
    print(f'  Horror OOD: {ood["horror_ood"]["acc"]:.4f}')
    with open(f'{DRIVE}/task5_ood_results.json','w') as f: json.dump({'in_dist':float(surf_acc),'ood':ood},f,indent=2)
    print('✅ Task 5 done')
except Exception as e: print(f'Task 5 error: {e}'); open(f'{DRIVE}/task5_ood_results.json','w').write(json.dumps({'error':str(e)}))

print(f'\n✅ Script 4 complete — meta-llama/Llama-3.2-3B\nNext: python3 05_ablations_drift_llama_v2.py')
