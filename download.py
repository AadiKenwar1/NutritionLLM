"""Downloads the four datasets into data/, in the exact layout process.py expects.
This is step zero of NutritionLLM: data/ is too big for git, so this script rebuilds it on any
machine (like a cloud GPU). It fetches only what process.py reads: Nutrition5k overhead rgb.png
photos plus its labels, the MM-Food CSV plus its first 6,000 photos, and the SNAPMe archive. SNAPMe
is unpacked whole because its "before" photos are links in snapme_cs_db that point at the real
files in snapme_nut_db. The fourth dataset is the USDA survey food table (FNDDS 2017-2018: macros
per 100 g and serving sizes for about 7,000 foods), which process.py turns into typed-meal practice
rows; it stays zipped, and process.py reads it that way. Files that already exist are skipped, so
it is safe to re-run after a crash.
Name datasets to fetch only some, like "python download.py nutrition5k snapme" on a test-only machine.
Look at the printed "failed" counts and the first error shown next to each: about 1,500 Nutrition5k
dishes have no overhead photo online, and a few MM-Food links may be dead. Both are normal, and
process.py skips those rows.
"""
import csv
import os
import shutil
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

N5K_URL = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
MMF_CSV = "https://huggingface.co/datasets/Codatta/MM-Food-100K/resolve/main/MM-Food-100K.csv"
SNAP_URL = "https://ndownloader.figshare.com/files/44532971"  # snapme_db_09Dec2022.tar.gz, ~2 GB
# FNDDS 2017-2018 (~4 MB), the release whose food codes and names match SNAPMe's labels best
USDA_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_json_2021-10-28.zip"
N5K, MMF, SNAP, USDA = "data/nutrition5k", "data/mmfood", "data/snapme", "data/usda"


# User agents to try in turn: the MM-Food photo host rejects Python's default one, while the SNAPMe
# host has refused the browser-style one from cloud machines
AGENTS = ("Mozilla/5.0", None)


# Save one URL to a path unless it already exists; returns None on success or the last error text
def fetch(url, path):
    if os.path.exists(path):
        return None
    for agent in AGENTS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": agent} if agent else {})
            with urllib.request.urlopen(req, timeout=30) as r:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".part", "wb") as f:
                    shutil.copyfileobj(r, f)
            os.replace(path + ".part", path)
            return None
        except Exception as e:
            error = f"{type(e).__name__}: {e}"[:100]
    return error


# Download many (url, path) pairs at once with a progress count; print how many failed and the first error
def fetch_all(name, jobs):
    errors = []
    with ThreadPoolExecutor(16) as pool:
        for n, error in enumerate(pool.map(lambda job: fetch(*job), jobs), 1):
            errors += [error] if error else []
            print(f"{name}: {n}/{len(jobs)}", end="\r", flush=True)
    hint = f", e.g. {errors[0]}" if errors else ""
    print(f"{name}: {len(jobs) - len(errors)} ok, {len(errors)} failed{hint}")


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


# USDA survey food table, kept zipped under its own file name (one JSON file inside)
def usda():
    fetch_all("usda table", [(USDA_URL, f"{USDA}/{os.path.basename(USDA_URL)}")])


# Datasets to fetch, chosen by name on the command line; no names means all four
STEPS = {"nutrition5k": nutrition5k, "mmfood": mmfood, "snapme": snapme, "usda": usda}
for name in sys.argv[1:] or STEPS:
    STEPS[name]()
