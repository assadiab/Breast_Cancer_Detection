#!/usr/bin/env python3
"""Generate the Kaggle preprocessing kernel: 1024px cache + ROI crop, zip-based resume."""
import json, sys

cells = []
def code(s): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s})
def md(s):   cells.append({"cell_type":"markdown","metadata":{},"source":s})

md("""# RSNA — Construction cache 1024px + crop ROI

Rebuilds the high-resolution cache from DICOM (windowing + Otsu breast crop + resize 1024).
**Zip-based resume** : mounts the previous partial cache (`rsna-cache-1024-assa` = un seul `cache_1024.zip`),
unzips it, builds the missing images, re-zips. Several runs until all 47004 images are built.""")

# ── Cell 1 : installs + imports ──────────────────────────────────────────────
code("""import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q',
    'pydicom','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg','python-gdcm'], check=False)
import os, time, zipfile, glob
import numpy as np, pandas as pd, cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
print("✅ imports OK")""")

# ── Cell 2 : chemins + reprise zip ───────────────────────────────────────────
code("""IMG_SIZE = 1024
WORK      = '/kaggle/working'
CACHE_OUT = f'{WORK}/cache_1024'
os.makedirs(CACHE_OUT, exist_ok=True)

# Competition DICOMs
DICOM_DIR = '/kaggle/input/rsna-breast-cancer-detection/train_images'
if not os.path.isdir(DICOM_DIR):
    for dp,dn,fn in os.walk('/kaggle/input'):
        if dp.endswith('train_images'): DICOM_DIR = dp; break
print("DICOM_DIR =", DICOM_DIR)

# List of images to build: to_build.csv (partial list of missing ones) if present,
# otherwise the full df_final (47004)
_todo_path = None
for dp,dn,fn in os.walk('/kaggle/input'):
    if 'to_build.csv' in fn: _todo_path = os.path.join(dp,'to_build.csv'); break
if _todo_path:
    df = pd.read_csv(_todo_path)
    print(f"📋 Partial list : {len(df)} images to build (from {_todo_path})")
else:
    _df_path = None
    for dp,dn,fn in os.walk('/kaggle/input'):
        if 'df_final.csv' in fn: _df_path = os.path.join(dp,'df_final.csv'); break
    df = pd.read_csv(_df_path)
    print(f"df_final : {len(df)} images to build (from {_df_path})")

# Resume: unzip the previous partial zip if it exists
_prev_zip = None
for dp,dn,fn in os.walk('/kaggle/input'):
    for f in fn:
        if f == 'cache_1024.zip': _prev_zip = os.path.join(dp,f)
if _prev_zip:
    print(f"♻️  Resuming from {_prev_zip} ...")
    with zipfile.ZipFile(_prev_zip) as z:
        z.extractall(WORK)   # restores cache_1024/
    print(f"   {len(glob.glob(CACHE_OUT+'/*.jpg'))} images already present")
else:
    print("No partial cache - starting from scratch")""")

# ── Cell 3: preprocessing function (windowing + ROI crop + resize) ───────
code('''def preprocess_one(args):
    pid, iid = args
    out = f"{CACHE_OUT}/{pid}_{iid}.jpg"
    if os.path.exists(out):
        return 'skip'
    dcm = f"{DICOM_DIR}/{pid}/{iid}.dcm"
    if not os.path.exists(dcm):
        return 'missing'
    try:
        ds = pydicom.dcmread(dcm)
        img = ds.pixel_array.astype(np.float32)
        # VOI-LUT windowing if available
        try:
            img = apply_voi_lut(img, ds).astype(np.float32)
        except Exception:
            pass
        # MONOCHROME1 → invert
        if str(getattr(ds,'PhotometricInterpretation','')).upper() == 'MONOCHROME1':
            img = img.max() - img
        # normalize to [0,1]
        mn, mx = float(img.min()), float(img.max())
        img = (img - mn) / (mx - mn + 1e-6)

        # ── breast ROI crop (Otsu + largest connected component) ──
        u8 = (np.clip(img,0,1)*255).astype(np.uint8)
        thr_val, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thr = max(thr_val/255.0, 0.05)
        mask = (img > thr).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num > 1:
            idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            if stats[idx, cv2.CC_STAT_AREA] > 8000:
                ys, xs = np.where(lbl == idx)
                y0,y1,x0,x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
                img = img[y0:y1, x0:x1]
        # square resize to 1024
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        # JPEG q95: ~110KB/img (vs PNG ~446KB) -> fits within the 20GB /kaggle/working limit
        cv2.imwrite(out, (np.clip(img,0,1)*255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        return 'ok'
    except Exception as e:
        return f'err:{e}'

print("✅ preprocess_one defined")''')

# ── Cell 4: parallel build ──────────────────────────────────────────
code("""todo = [(r['patient_id'], r['image_id']) for _,r in df.iterrows()
        if not os.path.exists(f"{CACHE_OUT}/{r['patient_id']}_{r['image_id']}.jpg")]
print(f"To build this run: {len(todo)} / {len(df)}")

# CPU kernel = 12h on Kaggle -> stop at 11h to leave time to zip
DEADLINE = time.time() + 11.0*3600
ok=skip=miss=err=0
t0=time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(preprocess_one, a): a for a in todo}
    for fu in tqdm(as_completed(futs), total=len(futs)):
        r = fu.result()
        if r=='ok': ok+=1
        elif r=='skip': skip+=1
        elif r=='missing': miss+=1
        else: err+=1
        if time.time() > DEADLINE:
            print("⏰ Deadline reached - clean stop to zip");
            for f in futs: f.cancel()
            break
print(f"✅ run done : ok={ok} skip={skip} missing={miss} err={err} en {(time.time()-t0)/60:.1f} min")
_n_total = len(glob.glob(CACHE_OUT+'/*.jpg'))
print(f"📦 Total cache : {_n_total} / {len(df)} images")""")

# ── Cell 5: output zip (delete-as-you-go -> peak disk = 1x) ─
code("""print("Zipping the cache (deleting as we go to save disk)...")
t0=time.time()
zip_path = f"{WORK}/cache_1024.zip"
files = glob.glob(CACHE_OUT+'/*.jpg')
# ZIP_STORED (JPEG already compressed) + remove each jpg after adding -> peak disk ~= 1x total size
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
    for p in files:
        z.write(p, arcname=f"cache_1024/{os.path.basename(p)}")
        os.remove(p)
print(f"✅ {zip_path} : {(time.time()-t0)/60:.1f} min ({os.path.getsize(zip_path)/1e9:.2f} GB)")

import shutil
shutil.rmtree(CACHE_OUT, ignore_errors=True)  # the folder is already empty
_done = 0
with zipfile.ZipFile(zip_path) as z:
    _done = sum(1 for n in z.namelist() if n.endswith('.jpg'))
print(f"🎯 {_done} images in the zip (target this run : {len(df)}) — "
      f"{'✅ run complete' if _done >= len(df)*0.99 else '⏳ relaunch for the rest'}")""")

# Ablation: --nowin -> crop-only cache (no VOI-LUT windowing)
NOWIN = '--nowin' in sys.argv
if NOWIN:
    _win_block = ("        # VOI-LUT windowing if available\n"
                  "        try:\n"
                  "            img = apply_voi_lut(img, ds).astype(np.float32)\n"
                  "        except Exception:\n"
                  "            pass\n")
    for c in cells:
        s = ''.join(c['source'])
        if 'apply_voi_lut' in s:
            c['source'] = s.replace(_win_block, "        # (crop-only ablation: NO VOI-LUT windowing)\n")

nb = {"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python","version":"3.12.0"},
      "kaggle":{"accelerator":"none","isGpuEnabled":False,"isInternetEnabled":True,
                "language":"python","sourceType":"notebook"}},
      "nbformat":4,"nbformat_minor":4,"cells":cells}
_args = [a for a in sys.argv[1:] if not a.startswith('--')]
out = _args[0] if _args else "cache_builder/rsna-build-cache-1024.ipynb"
with open(out,'w') as f: json.dump(nb,f,ensure_ascii=False,indent=1)
print(f"✅ Kernel generated : {out} ({len(cells)} cells){' [CROP-ONLY, no windowing]' if NOWIN else ' [windowing+crop]'}")
