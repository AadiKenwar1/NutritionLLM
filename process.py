"""Builds data/processed/ with three JSONL files: one example per line, with image paths (no copies).
This is the first step of NutritionLLM. The model learns from train.jsonl, which mixes Nutrition5k
dishes (foods, grams, measured macros) with up to 6,000 MM-Food photos (dish name, ingredients,
and macros that another model estimated, so the MM-Food macros are distillation, not ground truth).
test_lab.jsonl (Nutrition5k lab photos) and test_real_world.jsonl (SNAPMe phone photos, never
trained on) check how well it works. Look at the printed counts: MM-Food only includes images that
have finished downloading, so re-run this after the download ends. Rows with a missing image or a
placeholder label ("deprecated", "plate only", "N/A") are skipped on purpose. A source that was never
downloaded is skipped with a note, so a test-only machine can build test_lab.jsonl without the rest.
"""
import csv
import json
import os
import random
from itertools import islice

N5K, MMF, SNAP, OUT = "data/nutrition5k", "data/mmfood", "data/snapme", "data/processed"


# Full-task example: foods with grams, plus macro totals rounded to 1 decimal
def full(image, items, totals):
    return {"image": image, "task": "full", "answer": {
        "items": [{"food": food, "grams": round(float(grams), 1)} for food, grams in items],
        "totals": {k: round(float(v), 1) for k, v in zip(("kcal", "protein_g", "carbs_g", "fat_g"), totals)}}}


# Nutrition5k dishes in one split, skipping missing photos and placeholder labels
def nutrition5k(split, meta):
    rows = []
    with open(f"{N5K}/dish_ids/splits/rgb_{split}_ids.txt") as f:
        dishes = f.read().split()
    for dish in dishes:
        image, r = f"{N5K}/realsense_overhead/{dish}/rgb.png", meta.get(dish)
        # id, kcal, mass, fat, carbs, protein, then 7 fields per ingredient
        if r and os.path.exists(image) and not {"deprecated", "plate only"} & set(r[7::7]):
            rows.append(full(image, zip(r[7::7], r[8::7]), (r[1], r[5], r[4], r[3])))
    return rows


# MM-Food rows with a downloaded image: dish name, ingredients, and model-estimated macros
def mmfood(limit=6000):
    rows = []
    with open(f"{MMF}/MM-Food-100K.csv", encoding="utf-8") as f:
        for i, r in enumerate(islice(csv.DictReader(f), limit)):
            image, p = f"{MMF}/images/{i}.jpg", json.loads(r["nutritional_profile"])
            if r["dish_name"] not in ("", "N/A") and os.path.exists(image):
                totals = (p["calories_kcal"], p["protein_g"], p["carbohydrate_g"], p["fat_g"])
                rows.append({"image": image, "task": "foods", "answer": {
                    "dish": r["dish_name"], "foods": json.loads(r["ingredients"]),
                    "totals": dict(zip(("kcal", "protein_g", "carbs_g", "fat_g"), map(float, totals)))}})
    return rows


# SNAPMe "before" photos, each with its foods and summed macros
def snapme():
    db = f"{SNAP}/snapme_cs_db"
    # before_photos/ symlinks were lost on Windows, so use their targets
    with open(f"{db}/check_stats/chk_before_links.txt") as f:
        targets = [line.split()[2].removeprefix("../") for line in f if line.strip()]
    paths = {os.path.basename(t): f"{SNAP}/{t}" for t in targets}
    meals = {}
    with open(f"{db}/master_SNAPME_linkfile.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["filename"] in paths:
                meals.setdefault(r["filename"], []).append(r)
    return [full(paths[name], [(r["Food_Description"], r["FoodAmt"]) for r in foods],
                 [sum(float(r[k]) for r in foods) for k in ("KCAL", "PROT", "CARB", "TFAT")])
            for name, foods in meals.items() if os.path.exists(paths[name])]


# Build one source's rows if its key file was downloaded; otherwise print a note and give none
def rows_if(path, build):
    if os.path.exists(path):
        return build()
    print(f"{path} missing, skipped")
    return []


# Save rows as JSONL and print how many went in
def write(name, rows):
    with open(f"{OUT}/{name}", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{name}: {len(rows)}")


# The splits only cover Cafe 1 dishes
with open(f"{N5K}/metadata/dish_metadata_cafe1.csv") as f:
    meta = {r[0]: r for r in csv.reader(f)}

os.makedirs(OUT, exist_ok=True)
n5k, mm = nutrition5k("train", meta), rows_if(f"{MMF}/MM-Food-100K.csv", mmfood)
print(f"train sources: {len(n5k)} Nutrition5k + {len(mm)} MM-Food")
train = n5k + mm
random.Random(42).shuffle(train)
write("train.jsonl", train)
write("test_lab.jsonl", nutrition5k("test", meta))
write("test_real_world.jsonl", rows_if(f"{SNAP}/snapme_cs_db/master_SNAPME_linkfile.csv", snapme))
