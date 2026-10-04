"""Scores one model on one test file, prints a short report, and saves every prediction.
This is how NutritionLLM measures progress: run it on the stock model for a baseline, then on each
fine-tuned run, and compare the saved reports in results/. Every test row is a photo with its true
foods, grams, and macros. Three answers are scored per row: "photo" (foods, grams, and macros straight
from the image), "chain" (macros from the foods the model listed, as text), and "text" (macros from
the true food list, as if the user typed it). Food names are matched by their words in singular
form, so "cherry tomatoes" counts for "Tomatoes, raw" but "egg" does not count for "eggplant". Look at
"parsed" first: a low value means the model is not returning usable JSON, and every other number
suffers from it.
"""
import argparse
import json
import os
import random
import re
import time

import predict

KEYS = ("kcal", "protein_g", "carbs_g", "fat_g")


# Plural endings and their singular forms, first match wins; "ie" keeps "cookie" equal to "cookies"
ENDINGS = (("ies", "y"), ("ie", "y"), ("oes", "o"), ("ches", "ch"), ("shes", "sh"), ("ss", "ss"), ("s", ""))


# A word without its plural ending: "tomatoes" -> "tomato", "berries" -> "berry", "eggs" -> "egg"
def singular(word):
    for end, new in ENDINGS:
        if word.endswith(end):
            return word[:-len(end)] + new
    return word


# A name's singular words, so "Tomatoes, raw" and "cherry tomato" share the word "tomato"
def words(name):
    return {singular(w) for w in re.findall(r"[a-z]+", name)}


# True if every word of the shorter name is in the longer one
def same_food(a, b):
    a, b = words(a), words(b)
    return bool(a and b) and (a <= b or b <= a)


# True if a name matches any other, as whole names or as the parts before the first comma
def matches(name, others):
    return any(same_food(name, o) or same_food(name.split(",")[0], o.split(",")[0]) for o in others)


# Precision and recall of predicted food names, using matches() to decide a hit
def food_scores(true, pred):
    true, pred = [t.lower() for t in true], [str(p).lower() for p in pred]
    precision = sum(matches(x, true) for x in pred) / len(pred) if pred else 0
    recall = sum(matches(x, pred) for x in true) / len(true) if true else 0
    return precision, recall


# Absolute error for each macro, or None if the answer is missing or unreadable
def macro_errors(true, pred):
    values = {k: predict.num(pred.get(k)) for k in KEYS} if isinstance(pred, dict) else {}
    if not values or None in values.values():
        return None
    return {k: abs(values[k] - true[k]) for k in KEYS}


# Get all three answers for one row and return everything worth saving
def score_row(model, processor, row):
    true = row["answer"]
    photo = predict.predict_photo(model, processor, row["image"]) or {}
    items = [i for i in photo.get("items") or [] if isinstance(i, dict) and "food" in i]
    chain = predict.predict_text(model, processor, predict.meal_text(items)) if items else None
    text = predict.predict_text(model, processor, predict.meal_text(true["items"]))
    precision, recall = food_scores([i["food"] for i in true["items"]], [i["food"] for i in items])
    grams = sum(predict.num(i.get("grams")) or 0 for i in items) if items else None
    true_grams = sum(i["grams"] for i in true["items"])
    return {"image": row["image"], "parsed": bool(photo), "precision": precision, "recall": recall,
            "grams_error": None if grams is None else abs(grams - true_grams),
            "photo": macro_errors(true["totals"], photo.get("totals")),
            "chain": macro_errors(true["totals"], chain), "text": macro_errors(true["totals"], text),
            "pred": {"photo": photo, "chain": chain, "text": text}}


# Average a list of numbers, skipping None; None if there is nothing to average
def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


# Build the report from scored rows, print it, and return it
def summarize(rows):
    report = {"rows": len(rows), "parsed": mean([r["parsed"] for r in rows]),
              "food_precision": mean([r["precision"] for r in rows]),
              "food_recall": mean([r["recall"] for r in rows]),
              "grams_mae": mean([r["grams_error"] for r in rows])}
    for name in ("photo", "chain", "text"):
        errors = [r[name] for r in rows if r[name]]
        report[f"{name}_answered"] = len(errors) / len(rows)
        report[f"{name}_mae"] = {k: mean([e[k] for e in errors]) for k in KEYS}
    for k, v in report.items():
        if isinstance(v, dict):
            v = json.dumps(v)
        elif v is not None:
            v = round(v, 3)
        print(f"{k:16}", v)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=predict.DEFAULT)
    p.add_argument("--data", default="data/processed/test_lab.jsonl")
    p.add_argument("--limit", type=int, default=20, help="rows to score after a fixed shuffle; 0 means all")
    a = p.parse_args()
    with open(a.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.Random(0).shuffle(rows)
    rows = rows[:a.limit or None]
    model, processor = predict.load(a.model)
    os.makedirs("results", exist_ok=True)
    model_name = os.path.basename(a.model.rstrip("/"))
    data_name = os.path.basename(a.data).split(".")[0]
    name = f"results/{model_name}_{data_name}"
    start, scored = time.time(), []
    with open(f"{name}.jsonl", "w", encoding="utf-8") as f:
        for n, row in enumerate(rows, 1):
            result = score_row(model, processor, row)
            scored.append(result)
            f.write(json.dumps(result) + "\n")
            print(f"{n}/{len(rows)} rows, {(time.time() - start) / n:.0f} s each", end="\r", flush=True)
    print()
    with open(f"{name}.json", "w") as f:
        json.dump(summarize(scored), f, indent=1)
