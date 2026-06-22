#!/usr/bin/env python3
"""Generate the RSNA Mammo-CLIP MULTI-HEAD training notebook (1024 cache, 5 heads, multi-task loss)."""
import json, sys

cells = []
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s})
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s})

# ════════════════════════════════════════════════════════════════════════════
md("""# RSNA Breast Cancer — Mammo-CLIP B5 multi-head (1024px)

Shared EfficientNet-B5 encoder (Mammo-CLIP, pretrained on mammograms) + 5 heads:
**cancer** (main) + **biopsy** + **invasive** + **BIRADS** + **density** (auxiliary, regularization).

- **1024px + ROI crop** cache (dataset `rsna-cache-1024-assa`, JPEG mounted/unzipped at start)
- Multi-task loss: cancer (1.0) + aux (0.1-0.15), aux masked when the label is missing
- 2 stages (freeze -> gentle fine-tuning), selection on breast-level AUROC (cancer head), TTA""")

# ── Cell 1 : GPU ─────────────────────────────────────────────────────────────
code("""import subprocess
_smi = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                      capture_output=True, text=True)
print("GPU:", _smi.stdout.strip())
if 'T4' not in _smi.stdout:
    print("⚠️  GPU != T4 — check the accelerator in the Kaggle UI")""")

# ── Cell 2 : installs + imports ──────────────────────────────────────────────
code("""import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','efficientnet_pytorch==0.7.1'], check=False)

import os, json, time, gc, math, random, zipfile, glob
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import roc_auc_score

SEED=42; random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type=='cuda'
MC_MEAN, MC_STD = 0.3089279, 0.25053555408335154
print(f"torch {torch.__version__} | device {DEVICE} | AMP={USE_AMP}")""")

# ── Cell 3: 1024 cache (direct mount or unzip) + config ─────────────
code("""WORK='/kaggle/working'
# 1) If the dataset is already extracted (Kaggle unzips archives) -> use the .jpg folder directly
_best=(0,None)
for dp,dn,fn in os.walk('/kaggle/input'):
    n=sum(1 for f in fn if f.endswith('.jpg'))
    if n>_best[0]: _best=(n,dp)
if _best[0]>1000:
    CACHE_DIR=_best[1]
    print(f"✅ cache extracted (direct mount) : {_best[0]} JPEG → {CACHE_DIR}")
else:
    # 2) Otherwise unzip cache_1024.zip into /kaggle/working
    CACHE_DIR=f'{WORK}/cache_1024'
    _zip=None
    for dp,dn,fn in os.walk('/kaggle/input'):
        if 'cache_1024.zip' in fn: _zip=os.path.join(dp,'cache_1024.zip'); break
    assert _zip, "no .jpg folder nor cache_1024.zip found in /kaggle/input"
    print(f"Unzipping {_zip} ...")
    t0=time.time()
    with zipfile.ZipFile(_zip) as z: z.extractall(WORK)
    for dp,dn,fn in os.walk(WORK):
        if sum(1 for f in fn if f.endswith('.jpg'))>1000: CACHE_DIR=dp; break
    print(f"✅ {len(glob.glob(CACHE_DIR+'/*.jpg'))} JPEG in {(time.time()-t0)/60:.1f} min → {CACHE_DIR}")

# Split CSVs
_csv=None
for dp,dn,fn in os.walk('/kaggle/input'):
    if 'X_train.csv' in fn: _csv=dp; break
assert _csv, "X_train.csv not found"
print("CSV dir:", _csv)
# train.csv (auxiliary labels) - preferably from the competition
_train_csv=None
for dp,dn,fn in os.walk('/kaggle/input'):
    if 'train.csv' in fn and 'rsna-breast-cancer-detection' in dp: _train_csv=os.path.join(dp,'train.csv')
if _train_csv is None:
    for dp,dn,fn in os.walk('/kaggle/input'):
        if 'train.csv' in fn: _train_csv=os.path.join(dp,'train.csv'); break
print("train.csv:", _train_csv)

# ── Hyper-parameters (1024 is heavy -> small batch + grad accumulation) ────────────────
IMG_SIZE     = 1024
BATCH_SIZE   = 4
ACCUM_STEPS  = 8        # batch effectif = 32
NUM_WORKERS  = 4
P1_EPOCHS    = 2        # frozen backbone: just warm up the heads (the gain is in P2)
P2_EPOCHS    = 8        # gentle fine-tuning: productive stage (capped by the 8h safety guard)
PATIENCE     = 4
LR_HEAD_P1   = 1e-3
LR_BACKBONE  = 1e-5
LR_HEAD_P2   = 1e-4
WEIGHT_DECAY = 1e-3
WARMUP_RATIO = 0.1
W = {'cancer':1.0, 'biopsy':0.15, 'invasive':0.15, 'birads':0.10, 'density':0.10}
N_BIRADS, N_DENSITY = 3, 4

CKPT_DIR=f'{WORK}/checkpoints'; os.makedirs(CKPT_DIR,exist_ok=True)
RES_DIR =f'{WORK}/results';     os.makedirs(RES_DIR,exist_ok=True)
FIG_DIR =f'{WORK}/figures';     os.makedirs(FIG_DIR,exist_ok=True)
print("✅ Config ready")""")

# ── Cell 4: splits + merge auxiliary labels ───────────────────────────────
code("""X_train=pd.read_csv(f'{_csv}/X_train.csv'); Y_train=pd.read_csv(f'{_csv}/Y_train.csv')
X_val  =pd.read_csv(f'{_csv}/X_val.csv');   Y_val  =pd.read_csv(f'{_csv}/Y_val.csv')
X_test =pd.read_csv(f'{_csv}/X_test.csv');  Y_test =pd.read_csv(f'{_csv}/Y_test.csv')

_aux = pd.read_csv(_train_csv)[['patient_id','image_id','biopsy','invasive','BIRADS','density']]
_dens_map={'A':0,'B':1,'C':2,'D':3}
def add_labels(X, Y):
    d = X[['patient_id','image_id','laterality']].merge(_aux, on=['patient_id','image_id'], how='left')
    d['cancer']   = Y['cancer'].values.astype('float32')
    d['biopsy']   = d['biopsy'].fillna(0).astype('float32')
    d['invasive'] = d['invasive'].fillna(0).astype('float32')
    d['birads']   = pd.to_numeric(d['BIRADS'], errors='coerce').fillna(-1).astype('int64').clip(-1,2)
    d['density']  = d['density'].map(_dens_map).fillna(-1).astype('int64')
    return d
L_train, L_val, L_test = add_labels(X_train,Y_train), add_labels(X_val,Y_val), add_labels(X_test,Y_test)
for n,L in [('train',L_train),('val',L_val),('test',L_test)]:
    print(f"{n}: {len(L)} | cancer+={int(L['cancer'].sum())} biopsy+={int(L['biopsy'].sum())} "
          f"invasive+={int(L['invasive'].sum())} birads_ok={(L['birads']>=0).sum()} dens_ok={(L['density']>=0).sum()}")""")

# ── Cell 5: multi-label dataset (1024 cache already ROI-cropped) ────────────────────
code('''class MammoMultiDataset(Dataset):
    def __init__(self, L_df, cache_dir, img_size=1024, augment=False):
        d = L_df.reset_index(drop=True)
        paths = d.apply(lambda r: f"{cache_dir}/{r['patient_id']}_{r['image_id']}.jpg", axis=1).values
        mask = np.array([os.path.exists(p) for p in paths])
        if not mask.all():
            print(f"  ⚠️ {(~mask).sum()}/{len(mask)} images missing from cache (filtered out)")
        self.df = d[mask].reset_index(drop=True)
        self.paths = paths[mask]
        self.img_size = img_size; self.augment = augment

    def __len__(self): return len(self.df)

    def _aug(self, img):
        if random.random()<0.5: img=np.fliplr(img).copy()
        if random.random()<0.5: img=np.flipud(img).copy()
        if random.random()<0.4:
            a=random.uniform(-12,12); h,w=img.shape
            M=cv2.getRotationMatrix2D((w/2,h/2),a,1.0)
            img=cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_REFLECT_101)
        if random.random()<0.4: img=np.clip(img*random.uniform(0.85,1.15),0,1)
        if random.random()<0.3: img=np.power(np.clip(img,0,1),random.uniform(0.8,1.25))
        return img

    def __getitem__(self, i):
        img=cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None: img=np.zeros((self.img_size,self.img_size),np.float32)
        else:
            img=img.astype(np.float32)/255.0
            if img.shape[0]!=self.img_size: img=cv2.resize(img,(self.img_size,self.img_size),interpolation=cv2.INTER_AREA)
        if self.augment: img=self._aug(img)
        img=(np.clip(img,0,1)-MC_MEAN)/MC_STD
        t=torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).repeat(3,1,1)
        r=self.df.iloc[i]
        y={'cancer':torch.tensor(r['cancer'],dtype=torch.float32),
           'biopsy':torch.tensor(r['biopsy'],dtype=torch.float32),
           'invasive':torch.tensor(r['invasive'],dtype=torch.float32),
           'birads':torch.tensor(int(r['birads']),dtype=torch.long),
           'density':torch.tensor(int(r['density']),dtype=torch.long)}
        return t, y

print("✅ MammoMultiDataset defined")''')

# ── Cell 6: multi-head model ──────────────────────────────────────────────
code('''def gem(x,p=3,eps=1e-6): return F.avg_pool2d(x.clamp(min=eps).pow(p),(x.size(-2),x.size(-1))).pow(1.0/p)
class GeM(nn.Module):
    def __init__(self,p=3,eps=1e-6): super().__init__(); self.p=p; self.eps=eps
    def forward(self,x): return gem(x,self.p,self.eps).flatten(1)

def _head(nf, nout, drop=0.5):
    return nn.Sequential(nn.Dropout(drop), nn.Linear(nf,512), nn.BatchNorm1d(512),
                         nn.SiLU(), nn.Dropout(drop), nn.Linear(512,nout))

class MammoMultiHead(nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        self.encoder = EfficientNet.from_name('efficientnet-b5', num_classes=1)
        nf=2048; self.pool=GeM()
        if ckpt_path and os.path.exists(ckpt_path): self._load(ckpt_path)
        else: print("⚠️  pas de checkpoint Mammo-CLIP !")
        self.head_cancer   = _head(nf,1)
        self.head_biopsy   = _head(nf,1)
        self.head_invasive = _head(nf,1)
        self.head_birads   = _head(nf,N_BIRADS)
        self.head_density  = _head(nf,N_DENSITY)

    def _load(self,path):
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        raw=ckpt.get('model',ckpt); sd={}
        for k,v in raw.items():
            if k.startswith('image_encoder.'): sd[k[len('image_encoder.'):]]=v
        miss,unexp=self.encoder.load_state_dict(sd,strict=False)
        print(f"✅ Mammo-CLIP : {len(sd)-len(unexp)}/{len(sd)} tenseurs | manquants={len(miss)}")

    def freeze_backbone(self):
        for p in self.encoder.parameters(): p.requires_grad=False
        self.encoder.eval()
    def unfreeze_backbone(self):
        for p in self.encoder.parameters(): p.requires_grad=True
        self.encoder.train()

    def forward(self,x):
        f=self.pool(self.encoder.extract_features(x))
        return {'cancer':self.head_cancer(f).squeeze(-1),
                'biopsy':self.head_biopsy(f).squeeze(-1),
                'invasive':self.head_invasive(f).squeeze(-1),
                'birads':self.head_birads(f),
                'density':self.head_density(f)}

MAMMOCLIP_CKPT=None
for dp,dn,fn in os.walk('/kaggle/input'):
    for f in fn:
        if f.endswith('.tar') and 'b5' in f.lower(): MAMMOCLIP_CKPT=os.path.join(dp,f)
if MAMMOCLIP_CKPT is None:
    try:
        from huggingface_hub import hf_hub_download
        MAMMOCLIP_CKPT=hf_hub_download(repo_id="shawn24/Mammo-CLIP",
            filename="Pre-trained-checkpoints/b5-model-best-epoch-7.tar", cache_dir=f"{WORK}/hf")
    except Exception as e: print("⚠️ HF download failed:", e)
model=MammoMultiHead(MAMMOCLIP_CKPT).to(DEVICE)
print(f"✅ Multi-head model : {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")''')

# ── Cell 7: multi-task loss + metrics ───────────────────────────────────
code('''_pos=L_train['cancer'].sum(); _neg=len(L_train)-_pos
_pw=torch.tensor([min(_neg/_pos,5.0)],device=DEVICE)
bce_cancer=nn.BCEWithLogitsLoss(pos_weight=_pw)
bce=nn.BCEWithLogitsLoss()
ce=nn.CrossEntropyLoss(ignore_index=-1)   # masks -1 labels (missing BIRADS/density)

def multitask_loss(out, y):
    l = W['cancer']*bce_cancer(out['cancer'], y['cancer'])
    l = l + W['biopsy']*bce(out['biopsy'], y['biopsy'])
    l = l + W['invasive']*bce(out['invasive'], y['invasive'])
    if (y['birads']>=0).any():  l = l + W['birads']*ce(out['birads'], y['birads'])
    if (y['density']>=0).any(): l = l + W['density']*ce(out['density'], y['density'])
    return l

def pfbeta(labels,preds,beta=1.0):
    labels=np.asarray(labels,np.float64); preds=np.asarray(preds,np.float64)
    ctp=(preds*labels).sum(); cfp=(preds*(1-labels)).sum()
    p=ctp/(ctp+cfp+1e-9); r=ctp/(labels.sum()+1e-9)
    return 0.0 if p+r==0 else (1+beta**2)*p*r/(beta**2*p+r+1e-9)

def breast_agg(df,probs,labels):
    g=df.copy(); g['prob']=probs; g['lab']=labels
    a=g.groupby(['patient_id','laterality']).agg(prob=('prob','mean'),lab=('lab','max'))
    return a['prob'].values, a['lab'].values

def best_f1(labels,probs):
    b=0;bt=0.5
    for t in np.arange(0.02,0.9,0.02):
        pr=(probs>=t).astype(int)
        tp=((pr==1)&(labels==1)).sum();fp=((pr==1)&(labels==0)).sum();fn=((pr==0)&(labels==1)).sum()
        f=2*tp/(2*tp+fp+fn+1e-9)
        if f>b:b,bt=f,t
    return b,bt

def compute_metrics(df,probs,labels):
    probs=np.asarray(probs);labels=np.asarray(labels)
    auroc=roc_auc_score(labels,probs) if len(np.unique(labels))>1 else 0.5
    bp,bl=breast_agg(df,probs,labels)
    f1,thr=best_f1(bl,bp)
    return {'auroc':round(float(auroc),4),'pf1_breast':round(float(pfbeta(bl,bp)),4),
            'f1_breast':round(float(f1),4),'threshold':round(float(thr),3),
            'auroc_breast':round(float(roc_auc_score(bl,bp)) if len(np.unique(bl))>1 else 0.5,4)}
print("✅ loss + metrics defined")''')

# ── Cell 8 : loaders ─────────────────────────────────────────────────────────
code("""ds_tr=MammoMultiDataset(L_train,CACHE_DIR,IMG_SIZE,augment=True)
ds_va=MammoMultiDataset(L_val,  CACHE_DIR,IMG_SIZE,augment=False)
ds_te=MammoMultiDataset(L_test, CACHE_DIR,IMG_SIZE,augment=False)
_w=np.where(ds_tr.df['cancer'].values==1, 3.0, 1.0)
sampler=WeightedRandomSampler(torch.tensor(_w,dtype=torch.float), len(_w), replacement=True)
_pf={'prefetch_factor':2} if NUM_WORKERS>0 else {}
train_loader=DataLoader(ds_tr,batch_size=BATCH_SIZE,sampler=sampler,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True,**_pf)
val_loader  =DataLoader(ds_va,batch_size=BATCH_SIZE*2,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True)
test_loader =DataLoader(ds_te,batch_size=BATCH_SIZE*2,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True)
print(f"✅ loaders train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}")""")

# ── Cell 9: 2-stage training ───────────────────────────────────────────
code('''def run_eval(loader, L_df):
    model.eval(); probs=[]; labs=[]
    with torch.no_grad():
        for X,y in tqdm(loader,desc='eval',leave=False):
            X=X.to(DEVICE,non_blocking=True)
            with torch.amp.autocast('cuda',enabled=USE_AMP):
                out=model(X)
            probs.extend(torch.sigmoid(out['cancer']).float().cpu().numpy().tolist())
            labs.extend(y['cancer'].numpy().tolist())
    return compute_metrics(L_df, probs, labs)

history=[]; best_auroc=-1; best_state=None; best_tag=""
T_DEADLINE = time.time() + 8.0*3600   # safety guard: clean stop before the 9h GPU limit
def to_dev(y): return {k:v.to(DEVICE,non_blocking=True) for k,v in y.items()}

def train_phase(tag, n_epochs, optimizer, frozen, patience=PATIENCE):
    global best_auroc,best_state,best_tag
    if time.time() > T_DEADLINE:
        print(f"⏰ Deadline reached before {tag} - stage skipped"); return
    spe=max(1,len(train_loader)//ACCUM_STEPS); total=spe*n_epochs; warm=max(1,int(total*WARMUP_RATIO))
    sched=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: s/warm if s<warm else 0.5*(1+math.cos(math.pi*(s-warm)/max(1,total-warm))))
    scaler=torch.amp.GradScaler('cuda',enabled=USE_AMP); no_imp=0
    print(f"\\n━━━ {tag} : {n_epochs} ep (backbone {'FROZEN' if frozen else 'unfrozen'}) ━━━")
    for epoch in range(1,n_epochs+1):
        model.train()
        if frozen: model.encoder.eval()
        optimizer.zero_grad(); rl=0.0; nb=0
        for i,(X,y) in enumerate(tqdm(train_loader,desc=f'{tag} E{epoch}',leave=False)):
            X=X.to(DEVICE,non_blocking=True); y=to_dev(y)
            with torch.amp.autocast('cuda',enabled=USE_AMP):
                loss=multitask_loss(model(X), y)/ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i+1)%ACCUM_STEPS==0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,model.parameters()),5.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); sched.step()
            rl+=loss.item()*ACCUM_STEPS; nb+=1
            if time.time() > T_DEADLINE:
                print("⏰ Deadline reached mid-epoch - stopping"); break
        m=run_eval(val_loader, L_val)
        history.append({'phase':tag,'epoch':epoch,'train_loss':round(rl/max(nb,1),4),
                        'lr':round(optimizer.param_groups[0]['lr'],6),**m})
        print(f"{tag} E{epoch} | loss={rl/max(nb,1):.4f} | AUROC={m['auroc']:.4f} "
              f"| AUROC_sein={m['auroc_breast']:.4f} | pF1={m['pf1_breast']:.4f}")
        if m['auroc_breast']>best_auroc:
            best_auroc=m['auroc_breast']; best_tag=f"{tag} E{epoch}"
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            torch.save(best_state,f"{CKPT_DIR}/mammoclip_mh_best.pth"); no_imp=0
        else:
            no_imp+=1
            if no_imp>=patience: print(f"⏹ early stop {tag} E{epoch}"); break
        gc.collect(); torch.cuda.empty_cache()
        if time.time() > T_DEADLINE:
            print(f"⏰ Deadline - early end of {tag}"); break

t0=time.time()
model.freeze_backbone()
_heads=[model.head_cancer,model.head_biopsy,model.head_invasive,model.head_birads,model.head_density]
opt1=torch.optim.AdamW([p for h in _heads for p in h.parameters()], lr=LR_HEAD_P1, weight_decay=WEIGHT_DECAY)
train_phase("P1-frozen", P1_EPOCHS, opt1, frozen=True)

if best_state: model.load_state_dict(best_state)
gc.collect(); torch.cuda.empty_cache()
model.unfreeze_backbone()
opt2=torch.optim.AdamW([
    {'params':model.encoder.parameters(),'lr':LR_BACKBONE},
    {'params':[p for h in _heads for p in h.parameters()],'lr':LR_HEAD_P2},
], weight_decay=WEIGHT_DECAY)
train_phase("P2-ft", P2_EPOCHS, opt2, frozen=False)

if best_state: model.load_state_dict(best_state)
print(f"\\n✅ Training done in {(time.time()-t0)/60:.1f} min — best {best_tag} breast AUROC val={best_auroc:.4f}")''')

# ── Cell 10 : test + TTA ─────────────────────────────────────────────────────
code('''def run_test_tta(loader, L_df):
    model.eval(); probs=[]; labs=[]
    with torch.no_grad():
        for X,y in tqdm(loader,desc='test+TTA',leave=False):
            X=X.to(DEVICE,non_blocking=True)
            with torch.amp.autocast('cuda',enabled=USE_AMP):
                p1=torch.sigmoid(model(X)['cancer'])
                p2=torch.sigmoid(model(torch.flip(X,dims=[3]))['cancer'])
            p=((p1+p2)/2).float().cpu().numpy()
            probs.extend(p.tolist()); labs.extend(y['cancer'].numpy().tolist())
    return compute_metrics(L_df,probs,labs), probs, labs

test_metrics, test_probs, test_labs = run_test_tta(test_loader, L_test)
print("📊 TEST - Mammo-CLIP B5 multi-head (1024)")
for k,v in test_metrics.items(): print(f"  {k:14s}: {v}")

results={'model':'mammoclip_b5_multihead_1024','best_auroc_breast_val':best_auroc,'best_tag':best_tag,
         'history':history,'test_metrics':test_metrics,
         'config':{'img_size':IMG_SIZE,'batch':BATCH_SIZE*ACCUM_STEPS,'heads':list(W.keys()),'weights':W,
                   'p1_epochs':P1_EPOCHS,'p2_epochs':P2_EPOCHS}}
json.dump(results, open(f"{RES_DIR}/mammoclip_mh.json",'w'), indent=2)
print(f"✅ results -> {RES_DIR}/mammoclip_mh.json")''')

# ── Cell 11 : courbes ────────────────────────────────────────────────────────
code('''ep=list(range(1,len(history)+1)); _p1=sum(1 for h in history if h['phase'].startswith('P1'))
fig,ax=plt.subplots(1,2,figsize=(13,4))
ax[0].plot(ep,[h['train_loss'] for h in history],'o-'); ax[0].axvline(_p1+0.5,color='r',ls='--',alpha=.5)
ax[0].set_title('Multi-task loss'); ax[0].set_xlabel('epoch'); ax[0].grid(alpha=.3)
ax[1].plot(ep,[h['auroc'] for h in history],'o-',label='AUROC img')
ax[1].plot(ep,[h['auroc_breast'] for h in history],'s-',label='AUROC sein')
ax[1].plot(ep,[h['pf1_breast'] for h in history],'^-',label='pF1 sein')
ax[1].axvline(_p1+0.5,color='r',ls='--',alpha=.5); ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title('Val')
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/mammoclip_mh_curves.png",dpi=110,bbox_inches='tight'); plt.close()

from sklearn.metrics import roc_curve
bp,bl=breast_agg(L_test,test_probs,test_labs); fpr,tpr,_=roc_curve(bl,bp)
plt.figure(figsize=(5,5)); plt.plot(fpr,tpr,label=f"AUROC={test_metrics['auroc_breast']:.3f}")
plt.plot([0,1],[0,1],'k--',alpha=.3); plt.legend(); plt.grid(alpha=.3); plt.title('ROC test (sein)')
plt.savefig(f"{FIG_DIR}/mammoclip_mh_roc.png",dpi=110,bbox_inches='tight'); plt.close()
print("✅ figures saved\\n🎉 MULTI-HEAD NOTEBOOK DONE")''')

# ── Mode --resume : charger un checkpoint et reprendre directement en P2 ──────
RESUME = '--resume' in sys.argv
if RESUME:
    _load_ckpt = (
        "\n# RESUME : load the full v6 checkpoint (encoder + heads) on top of the Mammo-CLIP init\n"
        "for _dp,_dn,_fn in os.walk('/kaggle/input'):\n"
        "    for _f in _fn:\n"
        "        if _f.endswith('.pth'):\n"
        "            _ck=os.path.join(_dp,_f)\n"
        "            model.load_state_dict(torch.load(_ck,map_location=DEVICE)); print('checkpoint repris :',_ck)\n"
    )
    for c in cells:
        s=''.join(c['source'])
        # 1) inject checkpoint loading right after model creation
        if 'print(f"✅ Multi-head model' in s:
            c['source'] = s + _load_ckpt
        # 2) skip stage 1 (already done in the previous run) -> go straight to P2
        if 'train_phase("P1-frozen"' in s:
            s2 = s.replace(
                "model.freeze_backbone()\n"
                "_heads=[model.head_cancer,model.head_biopsy,model.head_invasive,model.head_birads,model.head_density]\n"
                "opt1=torch.optim.AdamW([p for h in _heads for p in h.parameters()], lr=LR_HEAD_P1, weight_decay=WEIGHT_DECAY)\n"
                'train_phase("P1-frozen", P1_EPOCHS, opt1, frozen=True)\n\n'
                "if best_state: model.load_state_dict(best_state)\n",
                "# RESUME : P1 skipped (checkpoint already fine-tuned)\n"
                "_heads=[model.head_cancer,model.head_biopsy,model.head_invasive,model.head_birads,model.head_density]\n")
            c['source'] = s2

# ── Ablation --cancer-only : set auxiliary head weights to 0 (same arch, cancer loss only) ──
if '--cancer-only' in sys.argv:
    for c in cells:
        s=''.join(c['source'])
        if "W = {'cancer':1.0" in s:
            c['source']=s.replace(
                "W = {'cancer':1.0, 'biopsy':0.15, 'invasive':0.15, 'birads':0.10, 'density':0.10}",
                "W = {'cancer':1.0, 'biopsy':0.0, 'invasive':0.0, 'birads':0.0, 'density':0.0}  # ablation: cancer-only")

# ════════════════════════════════════════════════════════════════════════════
nb={"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.12.0"},
    "kaggle":{"accelerator":"nvidiaTeslaT4","isGpuEnabled":True,"isInternetEnabled":True,
              "language":"python","sourceType":"notebook"}},
    "nbformat":4,"nbformat_minor":4,"cells":cells}
_args=[a for a in sys.argv[1:] if not a.startswith('--')]
out=_args[0] if _args else "notebooks_multihead/rsna-mammoclip-multihead.ipynb"
import os as _os; _os.makedirs(_os.path.dirname(out),exist_ok=True)
json.dump(nb, open(out,'w'), ensure_ascii=False, indent=1)
_tags = (' [RESUME P2]' if RESUME else '') + (' [CANCER-ONLY]' if '--cancer-only' in sys.argv else '')
print(f"✅ Notebook generated : {out} ({len(cells)} cells){_tags}")
