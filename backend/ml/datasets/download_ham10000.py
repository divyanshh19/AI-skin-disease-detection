"""
Download and organize the HAM10000 dataset via the Kaggle API.

Prerequisites:
  1. Create a Kaggle account, go to Account -> Create New API Token.
     This downloads kaggle.json.
  2. Place it at ~/.kaggle/kaggle.json  (chmod 600 ~/.kaggle/kaggle.json)
  3. pip install kaggle

Usage:
  python download_ham10000.py --out ../../../data/ham10000
"""
import argparse
import os
import shutil
import zipfile
import pandas as pd

KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

# HAM10000 7-class label map (dx column -> human readable name)
LABEL_MAP = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def download(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise SystemExit(
            "kaggle package not installed. Run: pip install kaggle"
        ) from e

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json
    print(f"Downloading {KAGGLE_DATASET} to {out_dir} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=out_dir, unzip=True)
    print("Download complete.")


def organize(out_dir: str):
    """
    Kaggle version of HAM10000 ships:
      HAM10000_metadata.csv
      HAM10000_images_part_1/ , HAM10000_images_part_2/  (or ham10000_images_part_1/2)
    We merge images into a single images/ folder and keep metadata.csv at the root,
    matching what ml/datasets/dataset.py expects.
    """
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for entry in os.listdir(out_dir):
        full = os.path.join(out_dir, entry)
        if os.path.isdir(full) and "images_part" in entry.lower():
            for fname in os.listdir(full):
                src = os.path.join(full, fname)
                dst = os.path.join(images_dir, fname)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(full, ignore_errors=True)

    # normalize metadata filename
    for entry in os.listdir(out_dir):
        if entry.lower() == "hmnist_28_28_l.csv" or entry.lower().startswith("hmnist"):
            continue  # not needed, this is a flattened-pixel variant, ignore
        if entry.lower() in ("ham10000_metadata.csv", "metadata.csv"):
            src = os.path.join(out_dir, entry)
            dst = os.path.join(out_dir, "metadata.csv")
            if src != dst:
                shutil.move(src, dst)

    meta_path = os.path.join(out_dir, "metadata.csv")
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        df["label_name"] = df["dx"].map(LABEL_MAP)
        df["image_path"] = df["image_id"].apply(lambda x: os.path.join("images", f"{x}.jpg"))
        df.to_csv(meta_path, index=False)
        print(f"Metadata organized: {len(df)} rows, {df['dx'].nunique()} classes.")
        print(df["dx"].value_counts())
    else:
        print("WARNING: metadata.csv not found after organization, check the raw download.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="../../../data/ham10000")
    parser.add_argument("--skip-download", action="store_true",
                         help="If you already manually placed the raw kaggle zip contents in --out")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    if not args.skip_download:
        download(out_dir)
    organize(out_dir)
