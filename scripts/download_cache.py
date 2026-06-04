#!/usr/bin/env python3
"""Télécharge TOUT l'output du kernel v42 (47k PNGs) en paginant via l'API bas-niveau."""
import os, sys, time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

KERNEL = "testlolll/rsna-breast-cancer-detection-assa"
OWNER, SLUG = KERNEL.split("/")
DEST = sys.argv[1] if len(sys.argv) > 1 else "./cache_dataset"
os.makedirs(DEST, exist_ok=True)

api = KaggleApi(); api.authenticate()

def list_page(kaggle, token):
    """Liste une page avec backoff exponentiel sur 429/erreurs réseau."""
    delay = 5
    for attempt in range(8):
        try:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = OWNER
            req.kernel_slug = SLUG
            req.page_size = 200
            if token:
                req.page_token = token
            return kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        except Exception as e:
            msg = str(e)
            if '429' in msg or 'Too Many' in msg or 'Connection' in msg or 'timeout' in msg.lower():
                print(f"  ⏳ rate-limit/réseau (tentative {attempt+1}) — pause {delay}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                raise
    raise RuntimeError("Échec liste page après 8 tentatives")

token = None
total = 0
page = 0
sess = requests.Session()
with api.build_kaggle_client() as kaggle:
    while True:
        resp = list_page(kaggle, token)
        files = resp.files or []
        page += 1
        for item in files:
            outfile = os.path.join(DEST, item.file_name)
            if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
                total += 1
                continue
            os.makedirs(os.path.split(outfile)[0], exist_ok=True)
            for attempt in range(3):
                try:
                    dr = sess.get(item.url, stream=True, timeout=60)
                    with open(outfile, "wb") as out:
                        out.write(dr.content)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  ⚠️ échec {item.file_name}: {e}")
                    time.sleep(1)
            total += 1
        token = resp.next_page_token
        print(f"[page {page}] cumul={total} fichiers — token={'oui' if token else 'FIN'}", flush=True)
        if not token or not files:
            break
        time.sleep(0.5)  # petit délai pour éviter le rate-limit

print(f"\n✅ Terminé : {total} fichiers dans {DEST}")
