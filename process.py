"""Builds data/processed/ with three JSONL files: one example per line, with image paths (no copies).
This is the first step of NutritionLLM. The model learns from train.jsonl, which mixes Nutrition5k
dishes (foods, grams, measured macros) with up to 6,000 MM-Food photos (dish name, ingredients,
portions, and macros that another model estimated, so the MM-Food macros are distillation, not
ground truth). The portions are MM-Food's rough weight guesses like "rice:200g", saved as foods with
grams; a row gets an empty portions list if any of its weights can't be read (like "egg:1").
train.jsonl also holds 5,000 typed meals with no photo, built from the USDA food table: real food
names, usual serving sizes, and true macros, so the model practices drinks, long food lists, and big
totals. A fifth of the USDA foods are kept out of those meals. test_lab.jsonl (Nutrition5k lab
photos) and test_real_world.jsonl (SNAPMe phone photos, never trained on) check how well it works.
Each real-world row lists its "unseen" foods, the ones no typed meal used, so test.py can tell
learning from memorizing: SNAPMe's true macros come from this same USDA table. Look at the printed
counts: MM-Food only includes images that have finished downloading, so re-run this after the
download ends. Rows with a missing image or a placeholder label ("deprecated", "plate only", "N/A")
are skipped on purpose. A source that was never downloaded is skipped with a note, so a test-only
machine can build test_lab.jsonl without the rest.
"""
import csv
import json
import os
import random
import re
import zipfile
from itertools import cycle, islice

N5K, MMF, SNAP, OUT = "data/nutrition5k", "data/mmfood", "data/snapme", "data/processed"
USDA = "data/usda/FoodData_Central_survey_food_json_2021-10-28.zip"


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


# Grams in each MM-Food portion unit; a millilitre of drink or broth counts as one gram
UNITS = {"g": 1, "ml": 1, "kg": 1000}
PORTION = re.compile(r"\s*(.+?)\s*:\s*(\d+(?:\.\d+)?)\s*(kg|g|ml)\s*", re.I)


# MM-Food "name:NNNg" portions as foods with grams; one unreadable weight empties the whole list
def portions(texts):
    found = [PORTION.fullmatch(t) for t in texts]
    if not all(found):
        return []
    return [{"food": m[1], "grams": round(float(m[2]) * UNITS[m[3].lower()], 1)} for m in found]


# MM-Food rows with a downloaded image: dish name, ingredients, portions, and model-estimated macros
def mmfood(limit=6000):
    rows = []
    with open(f"{MMF}/MM-Food-100K.csv", encoding="utf-8") as f:
        for i, r in enumerate(islice(csv.DictReader(f), limit)):
            image, p = f"{MMF}/images/{i}.jpg", json.loads(r["nutritional_profile"])
            if r["dish_name"] not in ("", "N/A") and os.path.exists(image):
                totals = (p["calories_kcal"], p["protein_g"], p["carbohydrate_g"], p["fat_g"])
                rows.append({"image": image, "task": "foods", "answer": {
                    "dish": r["dish_name"], "foods": json.loads(r["ingredients"]),
                    "portions": portions(json.loads(r["portion_size"])),
                    "totals": dict(zip(("kcal", "protein_g", "carbs_g", "fat_g"), map(float, totals)))}})
    return rows


# USDA nutrient names for kcal, protein, carbs, and fat, each given per 100 g
NUTRIENTS = ("Energy", "Protein", "Carbohydrate, by difference", "Total lipid (fat)")


# USDA foods that list a usual serving: code, name, macros per 100 g, and serving sizes in grams
def usda_foods():
    with zipfile.ZipFile(USDA) as z, z.open(z.namelist()[0]) as f:
        table = json.load(f)["SurveyFoods"]
    foods = []
    for t in table:
        per100 = {n["nutrient"]["name"]: n.get("amount") for n in t["foodNutrients"]}
        servings = [(p["portionDescription"], p["gramWeight"]) for p in t["foodPortions"]]
        usual = dict(servings).get("Quantity not specified", 0)
        if usual > 0:
            # keep servings from half to double the usual one, so a coffee is 240 to 600 g, never 30 g
            foods.append({"code": t["foodCode"], "name": " ".join(t["description"].split()),
                          "per100": [per100[n] for n in NUTRIENTS],
                          "amounts": sorted({w for _, w in servings if usual / 2 <= w <= usual * 2})})
    return foods


# Foods per typed meal, picked at random: two thirds of the meals are one food, the rest two to six
SIZES = (1,) * 10 + (2, 3, 4, 5, 6)


# Typed meals built from USDA foods, plus the codes of the foods they use; a fifth of the foods are
# held out. The kept foods are dealt like a deck of cards: each shows its full name once, then they
# come round again under short names ("coffee" for "Coffee, brewed"), which is about half the meals
def usda(limit=5000):
    rng = random.Random(42)
    foods = usda_foods()
    rng.shuffle(foods)
    kept = foods[len(foods) // 5:]
    deck, rows, used = cycle(kept), [], set()
    for _ in range(limit):
        size = rng.choice(SIZES)
        meal = [(f, rng.choice(f["amounts"])) for f in islice(deck, size)]
        short = len(used) == len(kept)
        items = [(f["name"].split(",")[0].lower() if short else f["name"], grams) for f, grams in meal]
        totals = [sum(grams * f["per100"][k] / 100 for f, grams in meal) for k in range(4)]
        rows.append(dict(full(None, items, totals), task="typed"))
        used.update(f["code"] for f, _ in meal)
    return rows, used


# SNAPMe "before" photos, each with its foods and summed macros. Given the USDA codes that the typed
# meals used, each row also lists its "unseen" foods: the ones whose code is not among them
def snapme(practiced):
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
    rows = []
    for name, foods in meals.items():
        if os.path.exists(paths[name]):
            row = full(paths[name], [(r["Food_Description"], r["FoodAmt"]) for r in foods],
                       [sum(float(r[k]) for r in foods) for k in ("KCAL", "PROT", "CARB", "TFAT")])
            if practiced is not None:
                row["unseen"] = sorted({r["Food_Description"] for r in foods if r["FoodCode"] not in practiced})
            rows.append(row)
    return rows


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
# without the USDA table there are no typed meals, and real-world rows get no "unseen" list
typed, practiced = rows_if(USDA, usda) or ([], None)
weighed = sum(bool(r["answer"]["portions"]) for r in mm)
print(f"train sources: {len(n5k)} Nutrition5k + {len(mm)} MM-Food ({weighed} with portions) + {len(typed)} USDA typed meals")
train = n5k + mm + typed
random.Random(42).shuffle(train)
write("train.jsonl", train)
write("test_lab.jsonl", nutrition5k("test", meta))
real = rows_if(f"{SNAP}/snapme_cs_db/master_SNAPME_linkfile.csv", lambda: snapme(practiced))
write("test_real_world.jsonl", real)
if real and practiced is not None:
    print(f"real-world meals with an unseen food: {sum(bool(r['unseen']) for r in real)}")
