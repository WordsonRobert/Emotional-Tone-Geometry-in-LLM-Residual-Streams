#!/usr/bin/env python3
# SCRIPT 02 — meta-llama/Llama-3.2-3B — cuda:1
import os,gc,json,math,torch,warnings,time
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer
import warnings; warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

DRIVE='/DATA/24ms190/steering_llama'; MODEL_PATH='/DATA/24ms190/steering_llama/model'; DEVICE='cuda:1'; EMOTIONS=['humor','melancholy','horror']

try: del model
except NameError: pass
gc.collect(); torch.cuda.empty_cache()

model=HookedTransformer.from_pretrained('meta-llama/Llama-3.2-3B',dtype=torch.bfloat16,device=DEVICE,fold_ln=False,center_unembed=False,move_to_device=True,cache_dir=MODEL_PATH)
model.eval()
# Llama — no chat template fix needed

N_LAYERS=model.cfg.n_layers; D_MODEL=model.cfg.d_model; PROBE_LAYER=N_LAYERS-2; ALL_LAYERS=list(range(N_LAYERS))
print(f'N_LAYERS={N_LAYERS} D_MODEL={D_MODEL} PROBE_LAYER={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

df_humor=pd.read_csv(f'{DRIVE}/humor_master.csv'); df_mel=pd.read_csv(f'{DRIVE}/melancholy_master.csv'); df_hor=pd.read_csv(f'{DRIVE}/horror_master.csv')
for df in [df_humor,df_mel,df_hor]:
    col=next((c for c in df.columns if c.lower() in ['text','content','post','body']),df.columns[0])
    if col!='text': df.rename(columns={col:'text'},inplace=True)
    df['text']=df['text'].fillna('').astype(str).str.strip()

humor_acts=torch.load(f'{DRIVE}/humor_acts.pt',map_location='cpu').float().numpy()
dark_acts =torch.load(f'{DRIVE}/dark_acts.pt', map_location='cpu').float().numpy()

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

def get_activation(text):
    tokens=model.to_tokens([text],prepend_bos=False)
    if tokens.shape[1]==0: return np.zeros(D_MODEL,dtype=np.float32)
    hn=f'blocks.{PROBE_LAYER}.hook_resid_post'
    with torch.no_grad():
        _,cache=model.run_with_cache(tokens,names_filter=lambda n:n==hn,stop_at_layer=PROBE_LAYER+1)
    act=cache[hn][0,-1,:].cpu().float().numpy(); del cache; torch.cuda.empty_cache(); return act

def generate_with_act(prompt,max_new_tokens,temperature,extra_hooks=None):
    captured={}
    hn=f'blocks.{PROBE_LAYER}.hook_resid_post'
    def cap(resid_post,hook): captured['act']=resid_post[0,-1,:].detach().cpu().float()
    hooks=[(hn,cap)]
    if extra_hooks: hooks=extra_hooks+hooks
    with model.hooks(fwd_hooks=hooks):
        out=model.generate(prompt,max_new_tokens=max_new_tokens,temperature=temperature,verbose=False,prepend_bos=False)
    return out,captured.get('act',torch.zeros(D_MODEL)).numpy()

def norm_v(v): n=np.linalg.norm(v); return v/n if n>1e-8 else v
def scores_from_act(act): return {'score_humor':float(act@w_hm),'score_mel':float(act@-w_hm),'score_horror':float(act@-w_hh)}

print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

# ── CELL 4: Pairwise probes + steering vectors ─────────────────────
def extract_acts(texts): return np.array([get_activation(t) for t in tqdm(texts)],dtype=np.float32)
def train_probe(acts_pos,acts_neg,label):
    n=min(len(acts_pos),len(acts_neg)); rng=np.random.default_rng(42)
    pos=acts_pos[rng.choice(len(acts_pos),n,replace=False)]; neg=acts_neg[rng.choice(len(acts_neg),n,replace=False)]
    X=np.vstack([pos,neg]).astype(np.float32); y=np.array([1]*n+[0]*n)
    mean=X.mean(0); std=X.std(0)+1e-8; Xn=(X-mean)/std; Xn=Xn/(np.linalg.norm(Xn,axis=1,keepdims=True)+1e-8)
    clf=LogisticRegression(max_iter=1000,C=1.0,random_state=42); cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    scores=cross_val_score(clf,Xn,y,cv=cv); clf.fit(Xn,y); w=norm_v(clf.coef_[0].astype(np.float32))
    print(f'  {label}: CV={scores.mean():.4f} ± {scores.std():.4f}  n={n}')
    return w,mean.astype(np.float32),std.astype(np.float32),float(scores.mean()),float(scores.std())

print('Extracting per-emotion activations...')
acts_humor=humor_acts; acts_mel=extract_acts(df_mel['text'].tolist()); acts_hor=extract_acts(df_hor['text'].tolist())

print('Training pairwise probes...')
w_hm,mean_hm,std_hm,acc_hm,std_hm_acc=train_probe(acts_humor,acts_mel,'humor vs mel')
w_hh,mean_hh,std_hh,acc_hh,std_hh_acc=train_probe(acts_humor,acts_hor,'humor vs horror')
w_mh,mean_mh,std_mh,acc_mh,std_mh_acc=train_probe(acts_mel,acts_hor,'mel vs horror')

torch.save(torch.tensor(w_hm),f'{DRIVE}/probe_humor_vs_melancholy.pt')
torch.save(torch.tensor(w_hh),f'{DRIVE}/probe_humor_vs_horror.pt')
torch.save(torch.tensor(w_mh),f'{DRIVE}/probe_melancholy_vs_horror.pt')

print('Probe cosines:')
for n1,n2,v1,v2 in [('w_hm','w_hh',w_hm,w_hh),('w_hm','w_mh',w_hm,w_mh),('w_hh','w_mh',w_hh,w_mh)]:
    print(f'  cos({n1},{n2}) = {float(v1@v2):+.4f}')

def build_sv(acts,mean,std):
    ma=acts.mean(0); normed=(ma-mean)/(std+1e-8); return norm_v(normed).astype(np.float32)

sv_raw={'humor':torch.tensor(build_sv(acts_humor,mean_hm,std_hm),dtype=torch.bfloat16),
         'melancholy':torch.tensor(build_sv(acts_mel,mean_hm,std_hm),dtype=torch.bfloat16),
         'horror':torch.tensor(build_sv(acts_hor,mean_hh,std_hh),dtype=torch.bfloat16)}
torch.save(sv_raw,f'{DRIVE}/steering_vecs.pt')
sv={k:v.to(DEVICE) for k,v in sv_raw.items()}
sv_np={k:v.cpu().float().numpy() for k,v in sv_raw.items()}

PROBE_FOR={'humor':w_hm,'melancholy':w_hm,'horror':w_hh}
PROBE_SIGN={'humor':+1,'melancholy':-1,'horror':-1}

print('SV cosines:')
for t1,t2 in [('humor','melancholy'),('humor','horror'),('melancholy','horror')]:
    print(f'  cos({t1:12},{t2:8}) = {float(norm_v(sv_np[t1])@norm_v(sv_np[t2])):+.4f}')

cell4={'probe_hm':{'cv_mean':acc_hm,'cv_std':std_hm_acc},'probe_hh':{'cv_mean':acc_hh,'cv_std':std_hh_acc},
       'probe_mh':{'cv_mean':acc_mh,'cv_std':std_mh_acc},
       'sv_cosines':{'humor_mel':float(norm_v(sv_np['humor'])@norm_v(sv_np['melancholy'])),
                       'humor_horror':float(norm_v(sv_np['humor'])@norm_v(sv_np['horror'])),
                       'mel_horror':float(norm_v(sv_np['melancholy'])@norm_v(sv_np['horror']))}}
with open(f'{DRIVE}/cell4_summary.json','w') as f: json.dump(cell4,f,indent=2)
print('✅ Cell 4 complete')

# ── CELL 5: Layer sweep + mult sweep ───────────────────────────────
def build_steer_hooks(layer,svec,mult):
    def fn(resid_pre,hook):
        proj=(resid_pre@svec).unsqueeze(-1)*svec; return (resid_pre-proj)+svec*mult
    return [(f'blocks.{layer}.hook_resid_pre',fn)]

MID_MULT=9.0; SWEEP_N=30
LAYER_CKPT=f'{DRIVE}/cell5_layer_sweep_raw.json'
if os.path.exists(LAYER_CKPT):
    with open(LAYER_CKPT) as f: intervention_results=json.load(f)
    intervention_results={em:{int(k):v for k,v in d.items()} for em,d in intervention_results.items()}
    print('Layer sweep: resumed')
else:
    intervention_results={em:{} for em in EMOTIONS}; print('Layer sweep: starting fresh')

prompts_sweep={'humor':df_humor.sample(SWEEP_N,random_state=99)['text'].tolist(),
               'melancholy':df_mel.sample(SWEEP_N,random_state=99)['text'].tolist(),
               'horror':df_hor.sample(min(SWEEP_N,len(df_hor)),random_state=99)['text'].tolist()}

for layer in tqdm(ALL_LAYERS,desc='Layer sweep'):
    if all(layer in intervention_results[em] for em in EMOTIONS): continue
    for emotion in EMOTIONS:
        if layer in intervention_results[emotion]: continue
        w=PROBE_FOR[emotion]; sign=PROBE_SIGN[emotion]; shifts=[]
        for prompt in prompts_sweep[emotion]:
            p=prompt[:120]
            _,base_act=generate_with_act(p,30,0.7)
            base_ps=sign*float(base_act@w)
            hooks=build_steer_hooks(layer,sv[emotion],MID_MULT)
            _,ste_act=generate_with_act(p,30,0.7,extra_hooks=hooks)
            ste_ps=sign*float(ste_act@w); shifts.append(ste_ps-base_ps)
        intervention_results[emotion][layer]=float(np.mean(shifts))
        print(f'  L{layer:2d} {emotion}: shift={np.mean(shifts):+.3f}',flush=True)
    with open(LAYER_CKPT,'w') as f:
        json.dump({em:{str(k):v for k,v in d.items()} for em,d in intervention_results.items()},f)

fig,axes_plt=plt.subplots(1,3,figsize=(15,4))
for idx,emotion in enumerate(EMOTIONS):
    ax=axes_plt[idx]; vals=[intervention_results[emotion].get(l,0) for l in ALL_LAYERS]
    ax.bar(ALL_LAYERS,vals,color='#3498db',alpha=0.8); ax.axhline(0,color='k',lw=0.5)
    ax.set_xlabel('Layer'); ax.set_ylabel('Mean probe shift'); ax.set_xticks(range(0,N_LAYERS,4))
plt.tight_layout(); plt.savefig(f'{DRIVE}/layer_heatmap.pdf',bbox_inches='tight'); plt.close()

top3={em:sorted(intervention_results[em],key=intervention_results[em].get,reverse=True)[:3] for em in EMOTIONS}
print('\nTop-3 layers:')
for em in EMOTIONS: print(f'  {em:12}: {top3[em]}  shifts={[f"{intervention_results[em][l]:+.3f}" for l in top3[em]]}')

MULTIPLIERS=[3.0,6.0,9.0,12.0,15.0]; MULT_N=15
MULT_CKPT=f'{DRIVE}/cell5_mult_sweep_full.json'
mult_results=json.load(open(MULT_CKPT)) if os.path.exists(MULT_CKPT) else {}

prompts_mult={'humor':df_humor.sample(MULT_N,random_state=77)['text'].tolist(),
              'melancholy':df_mel.sample(MULT_N,random_state=77)['text'].tolist(),
              'horror':df_hor.sample(min(MULT_N,len(df_hor)),random_state=77)['text'].tolist()}

for emotion in EMOTIONS:
    if emotion not in mult_results: mult_results[emotion]=dict()
    for layer in top3[emotion]:
        ls=str(layer)
        if ls not in mult_results[emotion]: mult_results[emotion][ls]=dict()
        for mult in tqdm(MULTIPLIERS,desc=f'{emotion} L{layer}'):
            ms=str(mult)
            if ms in mult_results[emotion][ls]: continue
            w=PROBE_FOR[emotion]; sign=PROBE_SIGN[emotion]; rows=[]
            for prompt in prompts_mult[emotion]:
                p=prompt[:120]
                _,base_act=generate_with_act(p,30,0.7)
                base_ps=sign*float(base_act@w)
                hooks=build_steer_hooks(layer,sv[emotion],mult)
                ste_out,ste_act=generate_with_act(p,30,0.7,extra_hooks=hooks)
                ste_ps=sign*float(ste_act@w); coh=coherence_score(ste_out[len(p):].strip())
                rows.append({'shift':ste_ps-base_ps,'coh':coh,'both':bool((ste_ps>base_ps)and(coh>=0.6))})
                tqdm.write(f'    {emotion} L{layer} x{mult}: shift={ste_ps-base_ps:+.3f} coh={coh:.2f} both={rows[-1]["both"]}')
            mult_results[emotion][ls][ms]={'both':float(np.mean([r['both'] for r in rows])),
                'mean_shift':float(np.mean([r['shift'] for r in rows])),
                'mean_coh':float(np.mean([r['coh'] for r in rows]))}
        with open(MULT_CKPT,'w') as f: json.dump(mult_results,f,indent=2)

CONFIGS=dict()
for emotion in EMOTIONS:
    best_both=-1.0; best_layer=None; best_mult=None
    for ls,mults in mult_results[emotion].items():
        for ms,metrics in mults.items():
            if metrics['both']>best_both or (metrics['both']==best_both and best_layer is not None and
               metrics['mean_shift']>mult_results[emotion][str(best_layer)][str(best_mult)]['mean_shift']):
                best_both=metrics['both']; best_layer=int(ls); best_mult=float(ms)
    CONFIGS[emotion]={'layer':best_layer,'mult':best_mult,'both':best_both}
CONFIGS['best_temp']=None
with open(f'{DRIVE}/best_configs.json','w') as f: json.dump(CONFIGS,f,indent=2)

print(f'\n{"━"*50}\nSCRIPT 2 SUMMARY — meta-llama/Llama-3.2-3B\n{"━"*50}')
print(f'  probe_hm={acc_hm:.4f}  probe_hh={acc_hh:.4f}  probe_mh={acc_mh:.4f}')
print(f'  SV: humor·mel={cell4["sv_cosines"]["humor_mel"]:+.4f}  humor·horror={cell4["sv_cosines"]["humor_horror"]:+.4f}  mel·horror={cell4["sv_cosines"]["mel_horror"]:+.4f}')
for em in EMOTIONS: print(f'  {em:12}: L{CONFIGS[em]["layer"]} x {CONFIGS[em]["mult"]}  both={CONFIGS[em]["both"]:.1%}')
print(f'\n✅ Script 2 complete\nNext: python3 03_temp_sweep_main_eval_llama_v2.py')
