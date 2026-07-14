"""
Pull a curated subset of images from the ISIC Archive to augment
underrepresented HAM10000 classes (e.g. dermatofibroma, vascular lesions).

Uses the public ISIC Archive API (no key required for metadata/download of
public images): https://api.isic-archive.com/api/v2/

Usage:
  python download_isic_subset.py --diagnosis "dermatofibroma" --limit 200 --out ../../../data/isic_subset
"""
import argparse
import os
import time
import requests
from tqdm import tqdm

API_BASE = "https://api.isic-archive.com/api/v2"


def search_images(diagnosis: str, limit: int):
    """Query ISIC's public search API for images with a given diagnosis label."""
    collected = []
    query = f'diagnosis:"{diagnosis}"'
    cursor = None
    with tqdm(total=limit, desc=f"Querying ISIC for '{diagnosis}'") as pbar:
        while len(collected) < limit:
            params = {"query": query, "limit": min(100, limit - len(collected))}
            if cursor:
                params["cursor"] = cursor
            resp = requests.get(f"{API_BASE}/images/search/", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            collected.extend(results)
            pbar.update(len(results))
            cursor = data.get("next")
            if not cursor:
                break
            time.sleep(0.2)  # be polite to the API
    return collected[:limit]


def download_images(records, out_dir):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest = []
    for rec in tqdm(records, desc="Downloading images"):
        isic_id = rec.get("isic_id")
        files = rec.get("files", {})
        url = files.get("full", {}).get("url") or files.get("thumbnail_256", {}).get("url")
        if not url or not isic_id:
            continue
        dest = os.path.join(img_dir, f"{isic_id}.jpg")
        if os.path.exists(dest):
            manifest.append((isic_id, rec.get("metadata", {}).get("clinical", {}).get("diagnosis")))
            continue
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(dest, "wb") as f:
                f.write(r.content)
            manifest.append((isic_id, rec.get("metadata", {}).get("clinical", {}).get("diagnosis")))
        except requests.RequestException as e:
            print(f"Failed {isic_id}: {e}")
        time.sleep(0.1)
    return manifest


def write_manifest(manifest, out_dir):
    import csv
    path = os.path.join(out_dir, "isic_subset_metadata.csv")
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["isic_id", "diagnosis"])
        for row in manifest:
            writer.writerow(row)
    print(f"Wrote manifest to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis", type=str, required=True,
                         help='ISIC diagnosis label, e.g. "dermatofibroma", "vascular lesion"')
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", type=str, default="../../../data/isic_subset")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    records = search_images(args.diagnosis, args.limit)
    print(f"Found {len(records)} candidate images.")
    manifest = download_images(records, out_dir)
    write_manifest(manifest, out_dir)
    print(f"Done. {len(manifest)} images saved under {out_dir}/images")
