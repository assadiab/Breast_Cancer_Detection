#!/usr/bin/env python3
"""Generate the pF1 calibration notebook: load the best checkpoint, run inference on val/test,
then grid-search temperature scaling x breast aggregation (mean/max) to maximize breast pF1.
No retraining - inference + post-processing only."""
import json, sys

cells = []
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s})
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s})

md("""# RSNA - pF1 calibration (no retraining)

Load the best Mammo-CLIP multi-head checkpoint, run inference on validation and test, then
**grid-search temperature scaling x breast-level aggregation (mean/max)** to maximize the
official RSNA metric (probabilistic F1). Temperature is fit on validation, then applied to test.""")

code("""import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','efficientnet_pytorch==0.7.1'], check=False)
import os, json, glob, time
import numpy as np, pandas as pd, cv2
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import roc_auc_score
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type=='cuda'
MC_MEAN, MC_STD = 0.3089279, 0.25053555408335154
IMG_SIZE, N_BIRADS, N_DENSITY = 1024, 3, 4
print('device', DEVICE)""")

# paths
code("""# Cache (extracted .jpg folder) + split CSVs + checkpoint
_best=(0,None)
for dp,dn,fn in os.walk('/kaggle/input'):
    n=sum(1 for f in fn if f.endswith('.jpg'))
    if n>_best[0]: _best=(n,dp)
CACHE_DIR=_best[1]; print('CACHE_DIR', CACHE_DIR, _best[0])
_csv=None
for dp,dn,fn in os.walk('/kaggle/input'):
    if 'X_val.csv' in fn: _csv=dp; break
CKPT=None
for dp,dn,fn in os.walk('/kaggle/input'):
    for f in fn:
        if f.endswith('.pth'): CKPT=os.path.join(dp,f)
print('CSV', _csv, '| CKPT', CKPT)

X_val=pd.read_csv(f'{_csv}/X_val.csv'); Y_val=pd.read_csv(f'{_csv}/Y_val.csv')
X_test=pd.read_csv(f'{_csv}/X_test.csv'); Y_test=pd.read_csv(f'{_csv}/Y_test.csv')
def with_lab(X,Y):
    d=X[['patient_id','image_id','laterality']].copy(); d['cancer']=Y['cancer'].values.astype('float32'); return d
L_val, L_test = with_lab(X_val,Y_val), with_lab(X_test,Y_test)""")

# dataset + model
code('''class EvalDS(Dataset):
    def __init__(self, L, cache_dir):
        paths=L.apply(lambda r:f"{cache_dir}/{r['patient_id']}_{r['image_id']}.jpg",axis=1).values
        m=np.array([os.path.exists(p) for p in paths])
        self.df=L[m].reset_index(drop=True); self.paths=paths[m]
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        img=cv2.imread(self.paths[i],cv2.IMREAD_GRAYSCALE)
        if img is None: img=np.zeros((IMG_SIZE,IMG_SIZE),np.float32)
        else:
            img=img.astype(np.float32)/255.0
            if img.shape[0]!=IMG_SIZE: img=cv2.resize(img,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
        img=(np.clip(img,0,1)-MC_MEAN)/MC_STD
        return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).repeat(3,1,1)

def gem(x,p=3,eps=1e-6): return F.avg_pool2d(x.clamp(min=eps).pow(p),(x.size(-2),x.size(-1))).pow(1/p)
class GeM(nn.Module):
    def __init__(s): super().__init__()
    def forward(s,x): return gem(x).flatten(1)
def _head(nf,n,drop=0.5):
    return nn.Sequential(nn.Dropout(drop),nn.Linear(nf,512),nn.BatchNorm1d(512),nn.SiLU(),nn.Dropout(drop),nn.Linear(512,n))
class MammoMultiHead(nn.Module):
    def __init__(s):
        super().__init__(); s.encoder=EfficientNet.from_name('efficientnet-b5',num_classes=1); nf=2048; s.pool=GeM()
        s.head_cancer=_head(nf,1); s.head_biopsy=_head(nf,1); s.head_invasive=_head(nf,1)
        s.head_birads=_head(nf,N_BIRADS); s.head_density=_head(nf,N_DENSITY)
    def forward(s,x):
        f=s.pool(s.encoder.extract_features(x)); return s.head_cancer(f).squeeze(-1)

model=MammoMultiHead().to(DEVICE)
model.load_state_dict(torch.load(CKPT,map_location=DEVICE)); model.eval()
print('checkpoint loaded')''')

# inference -> raw logits
code('''def infer_logits(L):
    ds=EvalDS(L,CACHE_DIR); dl=DataLoader(ds,batch_size=8,shuffle=False,num_workers=4,pin_memory=True)
    logits=[]
    with torch.no_grad():
        for X in tqdm(dl,desc='infer'):
            X=X.to(DEVICE)
            with torch.amp.autocast('cuda',enabled=USE_AMP):
                logits.append(model(X).float().cpu())
    return torch.cat(logits).numpy(), ds.df

val_logits, val_df = infer_logits(L_val)
test_logits, test_df = infer_logits(L_test)
print('val', val_logits.shape, '| test', test_logits.shape)''')

# calibration grid-search
code('''def pfbeta(labels,preds,beta=1.0):
    labels=np.asarray(labels,np.float64); preds=np.asarray(preds,np.float64)
    ctp=(preds*labels).sum(); cfp=(preds*(1-labels)).sum()
    p=ctp/(ctp+cfp+1e-9); r=ctp/(labels.sum()+1e-9)
    return 0.0 if p+r==0 else (1+beta**2)*p*r/(beta**2*p+r+1e-9)

def breast(df,probs,how):
    g=df.copy(); g['p']=probs
    a=g.groupby(['patient_id','laterality']).agg(p=('p',how),lab=('cancer','max'))
    return a['p'].values, a['lab'].values

def best_f1(lab,prob):
    b=0;bt=0.5
    for t in np.arange(0.02,0.95,0.01):
        pr=(prob>=t).astype(int); tp=((pr==1)&(lab==1)).sum();fp=((pr==1)&(lab==0)).sum();fn=((pr==0)&(lab==1)).sum()
        f=2*tp/(2*tp+fp+fn+1e-9)
        if f>b:b,bt=f,t
    return b,bt

# Baseline (no calibration): T=1, mean aggregation
bp,bl=breast(val_df, 1/(1+np.exp(-val_logits)), 'mean')
print(f"VAL baseline pF1 (T=1, mean) = {pfbeta(bl,bp):.4f}")

# Grid-search temperature x aggregation on VAL
best=(-1,None,None)
for how in ['mean','max']:
    for T in np.arange(0.4,3.01,0.1):
        probs=1/(1+np.exp(-val_logits/T))
        bp,bl=breast(val_df,probs,how)
        s=pfbeta(bl,bp)
        if s>best[0]: best=(s,round(float(T),2),how)
val_pf1,bestT,bestHow=best
print(f"VAL best pF1 = {val_pf1:.4f}  (T={bestT}, agg={bestHow})")

# Apply to TEST
def evaltest(T,how):
    probs=1/(1+np.exp(-test_logits/T))
    bp,bl=breast(test_df,probs,how)
    f1,thr=best_f1(bl,bp)
    return {'pf1':round(float(pfbeta(bl,bp)),4),'f1':round(float(f1),4),'thr':round(float(thr),3),
            'auroc':round(float(roc_auc_score(bl,bp)),4)}

base=evaltest(1.0,'mean'); cal=evaltest(bestT,bestHow)
print("\\nTEST baseline (T=1, mean):", base)
print(f"TEST calibrated (T={bestT}, agg={bestHow}):", cal)
res={'val_best':{'pf1':round(float(val_pf1),4),'T':bestT,'agg':bestHow},
     'test_baseline':base,'test_calibrated':cal}
os.makedirs('/kaggle/working/results',exist_ok=True)
json.dump(res, open('/kaggle/working/results/calibration.json','w'), indent=2)
print("\\nsaved /kaggle/working/results/calibration.json")''')

nb={"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.12.0"},
    "kaggle":{"accelerator":"nvidiaTeslaT4","isGpuEnabled":True,"isInternetEnabled":True,
              "language":"python","sourceType":"notebook"}},
    "nbformat":4,"nbformat_minor":4,"cells":cells}
import os as _os
out=sys.argv[1] if len(sys.argv)>1 else "kaggle/calibration/rsna-mammoclip-calibration.ipynb"
_os.makedirs(_os.path.dirname(out),exist_ok=True)
json.dump(nb, open(out,'w'), ensure_ascii=False, indent=1)
print(f"Notebook generated: {out} ({len(cells)} cells)")
