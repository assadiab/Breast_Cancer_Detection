#!/usr/bin/env python3
"""Deterministically rebuild the train/val/test split from the competition train.csv.

This is the single source of truth for the data split, so anyone with the competition data can
reproduce the exact patient-wise split used for training (no competition labels are redistributed).

Pipeline: drop implants -> keep one image per (patient, laterality, view) -> median-impute age ->
patient-level stratified 70/15/15 split (seed 42). Writes df_final.csv + X/Y_{train,val,test}.csv.

Usage:  python scripts/make_splits.py path/to/train.csv  [output_dir]
"""
import sys, os
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
FEATURES = ["site_id", "patient_id", "image_id", "laterality", "view", "age", "density", "machine_id"]


def build(train_csv: str, out_dir: str = "data") -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(train_csv)

    # 1) drop implants
    df = df[df["implant"] != 1].copy()

    # 2) one image per (patient, laterality, view) - deterministic
    df = (df.sort_values(["patient_id", "laterality", "view", "image_id"])
            .drop_duplicates(subset=["patient_id", "laterality", "view"], keep="first")
            .reset_index(drop=True))

    # 3) median-impute age
    df["age"] = df["age"].fillna(df["age"].median())

    # 4) missing density -> "N"
    df["density"] = df["density"].astype("object").where(df["density"].notna(), "N")

    df_final = df[FEATURES + ["cancer"]].copy()

    # 5) patient-level stratified split (70 / 15 / 15)
    pat = df_final.groupby("patient_id", sort=True).agg(cancer=("cancer", "max")).reset_index()
    train_pat, temp_pat = train_test_split(pat["patient_id"], test_size=0.30,
                                           random_state=SEED, stratify=pat["cancer"])
    temp = pat[pat["patient_id"].isin(temp_pat)]
    val_pat, test_pat = train_test_split(temp["patient_id"], test_size=0.50,
                                         random_state=SEED, stratify=temp["cancer"])
    df_final["split"] = "train"
    df_final.loc[df_final["patient_id"].isin(val_pat), "split"] = "val"
    df_final.loc[df_final["patient_id"].isin(test_pat), "split"] = "test"

    # 6) write df_final + X/Y splits
    df_final.to_csv(f"{out_dir}/df_final.csv", index=False)
    for name in ("train", "val", "test"):
        sub = df_final[df_final["split"] == name]
        sub[FEATURES].to_csv(f"{out_dir}/X_{name}.csv", index=False)
        sub[["cancer"]].to_csv(f"{out_dir}/Y_{name}.csv", index=False)

    for name in ("train", "val", "test"):
        n = (df_final["split"] == name).sum()
        print(f"{name:5s}: {n:6d} images")
    print(f"total: {len(df_final)} images, {df_final['cancer'].sum()} cancer  -> {out_dir}/")
    return df_final


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/make_splits.py path/to/train.csv [output_dir]")
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data")
