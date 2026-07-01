#!/usr/bin/env python3
# SCRIPT 08 — meta-llama/Llama-3.2-3B — cuda:1
import os,json,torch
import numpy as np,pandas as pd
from tqdm import tqdm
from transformer_lens import HookedTransformer
import gc,warnings; warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

DRIVE_EA='/DATA/24ms190/steering_llama/emotional_axes'; MODEL_PATH='/DATA/24ms190/steering_llama/model'; DEVICE='cuda:1'
os.makedirs(DRIVE_EA,exist_ok=True)

try: del model
except NameError: pass
gc.collect(); torch.cuda.empty_cache()
model=HookedTransformer.from_pretrained('meta-llama/Llama-3.2-3B',dtype=torch.bfloat16,device=DEVICE,fold_ln=False,center_unembed=False,move_to_device=True,cache_dir=MODEL_PATH)
model.eval()
N_LAYERS=model.cfg.n_layers; D_MODEL=model.cfg.d_model
MID_LAYER=N_LAYERS//2; PROBE_LAYER=N_LAYERS-2
print(f'N_LAYERS={N_LAYERS} D_MODEL={D_MODEL} MID={MID_LAYER} PROBE={PROBE_LAYER}')
print(f'VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB')

df=pd.read_csv(f'{DRIVE_EA}/emotional_rollouts.csv')
df['response']=df['response'].fillna('').astype(str)
df['system_prompt']=df['system_prompt'].fillna('').astype(str)
df['question']=df['question'].fillna('').astype(str)
print(f'Loaded {len(df)} rollouts across {df["tone"].nunique()} tones')

CHECKPOINT=f'{DRIVE_EA}/phase2b_checkpoint.json'
ACTS_MID  =f'{DRIVE_EA}/emotional_acts_L{MID_LAYER}.pt'
ACTS_PROBE=f'{DRIVE_EA}/emotional_acts_L{PROBE_LAYER}.pt'

if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f: ckpt=json.load(f)
    done_until=ckpt['done_until']
    acts_mid  =torch.load(ACTS_MID,  map_location='cpu')
    acts_probe=torch.load(ACTS_PROBE,map_location='cpu')
    print(f'Resumed from row {done_until}')
else:
    done_until=0
    acts_mid  =torch.zeros(len(df),D_MODEL,dtype=torch.float32)
    acts_probe=torch.zeros(len(df),D_MODEL,dtype=torch.float32)
    print('Starting fresh')

HOOK_MID  =f'blocks.{MID_LAYER}.hook_resid_post'
HOOK_PROBE=f'blocks.{PROBE_LAYER}.hook_resid_post'

def extract_response_acts(system_prompt,question,response):
    prefix=f"{system_prompt}\n\n{question}\n\n"
    prefix_toks=model.to_tokens([prefix],prepend_bos=False)
    n_prefix=prefix_toks.shape[1]
    full_text=prefix+response
    full_toks=model.to_tokens([full_text],prepend_bos=False)
    n_total=full_toks.shape[1]
    resp_start=min(n_prefix,n_total-1); resp_end=n_total
    if resp_start>=resp_end: resp_start=max(0,n_total-10)
    with torch.no_grad():
        _,cache=model.run_with_cache(full_toks,names_filter=lambda n:n in [HOOK_MID,HOOK_PROBE],stop_at_layer=PROBE_LAYER+1)
    a_mid  =cache[HOOK_MID  ][0,resp_start:resp_end].mean(0).cpu().float()
    a_probe=cache[HOOK_PROBE][0,resp_start:resp_end].mean(0).cpu().float()
    del cache; torch.cuda.empty_cache()
    return a_mid,a_probe

SAVE_EVERY=500
for idx in tqdm(range(done_until,len(df)),desc='Extracting'):
    row=df.iloc[idx]
    a_mid,a_probe=extract_response_acts(row['system_prompt'],row['question'],row['response'])
    acts_mid[idx]=a_mid; acts_probe[idx]=a_probe
    if (idx+1)%SAVE_EVERY==0:
        torch.save(acts_mid,ACTS_MID); torch.save(acts_probe,ACTS_PROBE)
        with open(CHECKPOINT,'w') as f: json.dump({'done_until':idx+1},f)
        print(f'  [{idx+1}/{len(df)}] checkpoint saved',flush=True)

torch.save(acts_mid,ACTS_MID); torch.save(acts_probe,ACTS_PROBE)
with open(CHECKPOINT,'w') as f: json.dump({'done_until':len(df)},f)
df[['tone','polarity','prompt_idx','question_idx','question']].to_csv(f'{DRIVE_EA}/emotional_acts_meta.csv',index=True)

print(f'\nSANITY CHECKS')
print(f'acts_mid   shape: {acts_mid.shape}  norm_mean={acts_mid.norm(dim=1).mean():.3f}')
print(f'acts_probe shape: {acts_probe.shape} norm_mean={acts_probe.norm(dim=1).mean():.3f}')
global_mean=acts_probe[df['polarity']=='positive'].mean(0)
print('\nCentered cosines at PROBE_LAYER:')
for t1,t2 in [('humor','melancholy'),('humor','horror'),('melancholy','horror')]:
    i1=df[(df['tone']==t1)&(df['polarity']=='positive')].index
    i2=df[(df['tone']==t2)&(df['polarity']=='positive')].index
    v1=acts_probe[i1].mean(0)-global_mean; v2=acts_probe[i2].mean(0)-global_mean
    v1=v1/(v1.norm()+1e-8); v2=v2/(v2.norm()+1e-8)
    print(f'  cos({t1:12},{t2:12}) = {float(v1@v2):+.4f}')

print(f'\n✅ Script 8 complete — meta-llama/Llama-3.2-3B')
print(f'   {ACTS_MID}'); print(f'   {ACTS_PROBE}')
print(f'\nNext: python3 09_emotional_pca_axes_llama_v2.py')
