"""Downloads the three datasets into data/, in the exact layout process.py expects.
This is step zero of NutritionLLM: data/ is too big for git, so this script rebuilds it on any
machine (like a cloud GPU). It fetches only what process.py reads: Nutrition5k overhead rgb.png
photos plus its labels, the MM-Food CSV plus its first 6,000 photos, and the SNAPMe archive. SNAPMe
is unpacked whole because its "before" photos are links in snapme_cs_db that point at the real
files in snapme_nut_db. Files that already exist are skipped, so it is safe to re-run after a crash.
Look at the printed "failed" counts: about 1,500 Nutrition5k dishes have no overhead photo online,
and a few MM-Food links may be dead. Both are normal, and process.py skips those rows.
"""
import csv
import os
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

N5K_URL = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
MMF_CSV = "https://huggingface.co/datasets/Codatta/MM-Food-100K/resolve/main/MM-Food-100K.csv"
SNAP_URL = "https://ndownloader.figshare.com/files/44532971"  # snapme_db_09Dec2022.tar.gz, ~2 GB
N5K, MMF, SNAP = "data/nutrition5k", "data/mmfood", "data/snapme"


# Save one URL to a path unless it already exists; returns True on success
def fetch(url, path):
    if os.path.exists(path):
        return True
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
        os.replace(path + ".part", path)
        return True
    except Exception:
        return False


# Download many (url, path) pairs at once and print how many failed
def fetch_all(name, jobs):
    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(lambda job: fetch(*job), jobs))
    print(f"{name}: {sum(ok)} ok, {ok.count(False)} failed")


# Nutrition5k labels, then one overhead photo per dish in the train and test splits
def nutrition5k():
    labels = ["metadata/dish_metadata_cafe1.csv",
              "dish_ids/splits/rgb_train_ids.txt", "dish_ids/splits/rgb_test_ids.txt"]
    fetch_all("nutrition5k labels", [(f"{N5K_URL}/{p}", f"{N5K}/{p}") for p in labels])
    dishes = []
    for split in ("train", "test"):
        with open(f"{N5K}/dish_ids/splits/rgb_{split}_ids.txt") as f:
            dishes += f.read().split()
    fetch_all("nutrition5k photos", [
        (f"{N5K_URL}/imagery/realsense_overhead/{d}/rgb.png", f"{N5K}/realsense_overhead/{d}/rgb.png")
        for d in dishes])


# MM-Food CSV, then photos for its first rows, saved as images/<row>.jpg
def mmfood(limit=6000):
    fetch_all("mmfood csv", [(MMF_CSV, f"{MMF}/MM-Food-100K.csv")])
    with open(f"{MMF}/MM-Food-100K.csv", encoding="utf-8") as f:
        rows = islice(csv.DictReader(f), limit)
        jobs = [(r["image_url"], f"{MMF}/images/{i}.jpg") for i, r in enumerate(rows)]
    fetch_all("mmfood photos", jobs)


# SNAPMe archive, unpacked whole in one pass, skipping symlinks (they break on Windows)
def snapme():
    archive = f"{SNAP}/snapme.tar.gz"
    # the archive is deleted only after a full unpack, so a leftover one means the last try died
    if os.path.exists(f"{SNAP}/snapme_cs_db") and not os.path.exists(archive):
        return print("snapme: already there")
    fetch_all("snapme archive", [(SNAP_URL, archive)])
    if not os.path.exists(archive):
        return print("snapme: download failed, run again")
    count = 0
    with tarfile.open(archive) as tar:
        for m in tar:
            if "/" in m.name and not (m.issym() or m.islnk()):
                m.name = m.name.split("/", 1)[1]  # drop the top folder so the layout matches process.py
                tar.extract(m, SNAP, filter="data")
                count += 1
    os.remove(archive)
    print(f"snapme: unpacked {count} files")


nutrition5k()
mmfood()
snapme()
