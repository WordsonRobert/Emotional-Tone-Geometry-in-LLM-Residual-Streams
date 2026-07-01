#!/usr/bin/env python3
# SCRIPT 01 — meta-llama/Llama-3.2-3B — cuda:1 — cuda:1
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
warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

DRIVE='/DATA/24ms190/steering_llama'; MODEL_PATH='/DATA/24ms190/steering_llama/model'
DEVICE='cuda:1'; EMOTIONS=['humor','melancholy','horror']

try: del model
except NameError: pass
gc.collect(); torch.cuda.empty_cache()
print(f'VRAM before load: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

print('Loading meta-llama/Llama-3.2-3B...')
model=HookedTransformer.from_pretrained('meta-llama/Llama-3.2-3B',dtype=torch.bfloat16,device=DEVICE,
    fold_ln=False,center_unembed=False,move_to_device=True,cache_dir=MODEL_PATH)
model.eval()
# Llama - no chat template fix needed


t0=time.time()
_=model.generate('The sky is',max_new_tokens=20,temperature=0.7,verbose=False)
print(f'Speed: {time.time()-t0:.1f}s for 20 tokens')

N_LAYERS=model.cfg.n_layers; D_MODEL=model.cfg.d_model; PROBE_LAYER=N_LAYERS-2
ALL_LAYERS=list(range(N_LAYERS))
print(f'N_LAYERS={N_LAYERS} D_MODEL={D_MODEL} PROBE_LAYER={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

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
        w=w.strip('.,!?-()').lower()
        return len(w)<2 or (any(c in 'aeiou' for c in w) and not any(p in w for p in BAD))
    return sum(ok(w) for w in words)/len(words)

def get_activation_at_layer(text,layer):
    tokens=model.to_tokens([text],prepend_bos=False)
    if tokens.shape[1]==0: return np.zeros(D_MODEL,dtype=np.float32)
    hn=f'blocks.{layer}.hook_resid_post'
    with torch.no_grad():
        _,cache=model.run_with_cache(tokens,names_filter=lambda n:n==hn,stop_at_layer=layer+1)
    act=cache[hn][0,-1,:].cpu().float().numpy(); del cache; torch.cuda.empty_cache(); return act

def get_activation(text): return get_activation_at_layer(text,PROBE_LAYER)
def norm_v(v): n=np.linalg.norm(v); return v/n if n>1e-8 else v
def normalize_for_probe(X):
    mean=X.mean(0); std=X.std(0)+1e-8; Xn=(X-mean)/std
    norms=np.linalg.norm(Xn,axis=1,keepdims=True)+1e-8
    return (Xn/norms).astype(np.float32),mean.astype(np.float32),std.astype(np.float32)

def _load_csv(path):
    df=pd.read_csv(path)
    col=next((c for c in df.columns if c.lower() in ['text','content','post','body']),df.columns[0])
    if col!='text': df.rename(columns={col:'text'},inplace=True)
    df['text']=df['text'].fillna('').astype(str).str.strip()
    return df[df['text'].str.len()>10].reset_index(drop=True)

df_humor=_load_csv(f'/DATA/24ms190/steering_llama/humor_master.csv')
df_mel  =_load_csv(f'/DATA/24ms190/steering_llama/melancholy_master.csv')
df_hor  =_load_csv(f'/DATA/24ms190/steering_llama/horror_master.csv')
dark_texts=df_mel['text'].tolist()+df_hor['text'].tolist()
print(f'humor={len(df_humor)} mel={len(df_mel)} hor={len(df_hor)} dark={len(dark_texts)}')

def _extract_acts(texts,label):
    path=f'/DATA/24ms190/steering_llama/{label}_acts.pt'
    if os.path.exists(path):
        print(f'  {label}: loading cached'); return torch.load(path,map_location='cpu').float().numpy()
    acts=[get_activation(t) for t in tqdm(texts,desc=f'  {label}')]
    arr=np.array(acts,dtype=np.float32); torch.save(torch.tensor(arr),path)
    print(f'  {label}: saved shape={arr.shape}'); return arr

print('\nExtracting activations at PROBE_LAYER...')
humor_acts=_extract_acts(df_humor['text'].tolist(),'humor')
dark_acts =_extract_acts(dark_texts,'dark')

N_PROBE=min(300,len(df_humor),len(dark_texts))
rng=np.random.default_rng(42)
h_idx=rng.choice(len(df_humor),size=N_PROBE,replace=False)
d_idx=rng.choice(len(dark_texts),size=N_PROBE,replace=False)

PROBE_CKPT=f'/DATA/24ms190/steering_llama/cell3_probe_checkpoint.json'
if os.path.exists(PROBE_CKPT):
    with open(PROBE_CKPT) as f: ckpt=json.load(f)
    layer_probe_accs={int(k):v for k,v in ckpt['accs'].items()}
    pw_loaded=torch.load(f'/DATA/24ms190/steering_llama/probe_weights.pt',map_location='cpu')
    probe_weights={int(k):v.float().numpy() for k,v in pw_loaded.items()}
    print(f'Resumed: {len(layer_probe_accs)} layers done')
else:
    ckpt={'accs':{}}; layer_probe_accs={}; probe_weights={}

for layer in tqdm(ALL_LAYERS,desc='Probe layers'):
    if layer in layer_probe_accs: continue
    h_sub=[get_activation_at_layer(df_humor['text'].iloc[i],layer) for i in h_idx]
    d_sub=[get_activation_at_layer(dark_texts[i],layer) for i in d_idx]
    X=np.vstack([h_sub,d_sub]).astype(np.float32); y=np.array([1]*N_PROBE+[0]*N_PROBE)
    scaler=StandardScaler(); X=scaler.fit_transform(X)
    X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    clf=LogisticRegression(max_iter=1000,C=1.0,random_state=42)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    acc=cross_val_score(clf,X,y,cv=cv).mean()
    clf.fit(X,y); w=norm_v(clf.coef_[0].astype(np.float32))
    layer_probe_accs[layer]=float(acc); probe_weights[layer]=w
    ckpt['accs']={str(k):v for k,v in layer_probe_accs.items()}
    with open(PROBE_CKPT,'w') as f: json.dump(ckpt,f)
    torch.save({k:torch.tensor(v) for k,v in probe_weights.items()},f'/DATA/24ms190/steering_llama/probe_weights.pt')
    print(f'  L{layer:2d}: {acc:.4f}',flush=True)

X_check=np.vstack([humor_acts[h_idx],dark_acts[d_idx]]).astype(np.float32)
y_check=np.array([1]*N_PROBE+[0]*N_PROBE)
Xn,_,_=normalize_for_probe(X_check)
cv_check=cross_val_score(LogisticRegression(max_iter=1000,C=1.0),Xn,y_check,
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42))
print(f'\nPROBE_LAYER L{PROBE_LAYER} CV: {cv_check.mean():.4f} ± {cv_check.std():.4f}')

best_layer=max(layer_probe_accs,key=layer_probe_accs.get)
print('\nLayer accuracies:')
for l in ALL_LAYERS:
    marker=' ← BEST' if l==best_layer else (' ← PROBE_LAYER' if l==PROBE_LAYER else '')
    print(f'  L{l:2d}: {layer_probe_accs[l]:.4f}{marker}')

layers_sorted=sorted(layer_probe_accs.keys()); accs_sorted=[layer_probe_accs[l] for l in layers_sorted]
colors=['#C44E52' if l==PROBE_LAYER else '#DD8452' if l==best_layer else '#4C72B0' for l in layers_sorted]
fig,ax=plt.subplots(figsize=(14,4))
ax.bar(layers_sorted,accs_sorted,color=colors,alpha=0.85,width=0.75)
ax.axhline(0.5,color='gray',ls='--',lw=1); ax.axvline(PROBE_LAYER,color='#C44E52',lw=2)
ax.set_xlabel('Layer'); ax.set_ylabel('CV Accuracy'); ax.set_ylim(0.4,1.05)
ax.set_xticks(range(0,N_LAYERS,2))
plt.tight_layout(); plt.savefig(f'/DATA/24ms190/steering_llama/layer_probe_accuracy.pdf',bbox_inches='tight'); plt.close()

torch.save({k:torch.tensor(v) for k,v in probe_weights.items()},f'/DATA/24ms190/steering_llama/probe_weights.pt')
summary={'model_name':'meta-llama/Llama-3.2-3B','n_layers':N_LAYERS,'d_model':D_MODEL,
    'probe_layer':PROBE_LAYER,'probe_layer_acc':float(layer_probe_accs[PROBE_LAYER]),
    'probe_layer_acc_std':float(cv_check.std()),'best_layer':best_layer,
    'best_layer_acc':float(layer_probe_accs[best_layer]),'n_probe_per_class':N_PROBE,
    'humor_n':len(df_humor),'mel_n':len(df_mel),'horror_n':len(df_hor),
    'layer_accs':{str(k):float(v) for k,v in layer_probe_accs.items()}}
with open(f'/DATA/24ms190/steering_llama/cell1_model_summary.json','w') as f: json.dump(summary,f,indent=2)

print(f'\n{"━"*55}\nSCRIPT 1 SUMMARY — meta-llama/Llama-3.2-3B\n{"━"*55}')
print(f'  humor_acts: {humor_acts.shape}  dark_acts: {dark_acts.shape}')
print(f'  PROBE_LAYER acc: {layer_probe_accs[PROBE_LAYER]:.4f} ± {cv_check.std():.4f}')
print(f'  Best layer: L{best_layer} = {layer_probe_accs[best_layer]:.4f}')
print(f'\n✅ Script 1 complete\nNext: python3 02_probes_steering_sweep_llama.py')
