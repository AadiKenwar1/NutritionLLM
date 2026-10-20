# Timeline

Every feature and data source added or removed, in order, with the scores after each run, and why the
model behaves as it does on 2026-10-06. Scores come from `results/`: 1,478 SNAPMe phone photos
("real world") and 498 Nutrition5k lab plates ("lab"). Training facts come from `runs/*/train.json`.

## Terms

- Base model: `LFM2.5-VL-450M`, 450 million numbers that read pictures and text; never changed.
- Adapter (LoRA): the 7 million numbers we train and clip onto the base.
- Chat: one training example. Token: a word piece; answers are written one token at a time.
- Photo, chain, text answers: totals from the image; from the model's own food list as text; from the true food list as text (the typed task).
- Error: average size of the miss. Precision / recall: named foods that were right / true foods found.
- Control: a model trained with a data source left out, to show what that source bought.

## What happened, in order

### 2026-09-27: the pipeline

- Base model; LoRA rank 16 on the attention, short-conv, and feed-forward layers; learning rate 1e-4; one pass; one chat per step.
- Photos shrunk to 512 px; answers written without randomness, with a light repeat penalty; reply cap 300 tokens.
- In: Nutrition5k, 2,731 training plates and 498 lab test plates, weighed, macros measured.
- In: MM-Food, the first 6,000 phone photos: dish, ingredient names, macros estimated by another model, no grams.
- In: SNAPMe, 1,478 weighed phone meals, as the real-world test only.
- Three answers scored per photo from day one; names matched when one sat inside the other letter by letter.
- Base model, untrained: real-world photo 245 kcal, typed 256, grams 156; lab (20 rows) photo 136, typed 220.
- Out: result files left git. Reply cap raised 300 to 700 after long dish lists were cut off.
- First fine-tune, `runs/full`: Nutrition5k plus MM-Food, 17,460 chats, loss 0.67 to 0.23.

### 2026-09-28: first scores and a diagnosis

- In: iPhone HEIC photos (named `.jpeg`) made readable.
- `full`, real world: photo 130, chain 142, typed 126, grams 273, names 0.45 / 0.45; lab: photo 80, typed 33, grams 51.
- Calories halved against the base; names flat; weights worse than the base's 156 g.
- Cause: MM-Food answers had no grams, so the model skipped weights on phone photos (7 of 4,160 foods got one), and a missing weight counts as zero.
- Also seen: coffee at 170 to 210 kcal, a favorite 1,018 kcal answer 43 times, and a constant guess (240 g, 244 kcal) scoring 147 g and 197 kcal.
- Plan written in [proposedFixes.md](proposedFixes.md).

### 2026-10-03: Fix 1 and the USDA download

- In: MM-Food's `portion_size` parsed into a portions list, name and grams per item (5,974 of 5,999 rows parse); photo and typed answers built from it, so every food has grams.
- Out: the ingredient-name list from MM-Food answers (names got coarser, like "vegetables 150 g") and the 25 rows whose portions could not be read.
- In: `download.py` fetches the USDA FNDDS food table, the data for Fix 2.

### 2026-10-04: Fix 3, a fairer name score

- Out: letter-inside matching, which called "egg" right against "eggplant" and "cherry tomatoes" wrong against "Tomatoes, raw".
- In: singular-word matching, with a second try on the part before the first comma; rescoring the 09-28 run moved precision 0.454 to 0.518, no retrain.
- Name scores before this date are not comparable with those after.

### 2026-10-05, early: Fix 1 run, `runs/full_portions`

- 17,410 chats, loss 0.57 to 0.22; real world: photo 131.5, chain 133.8, typed 107.2, grams 113.3, names 0.60 / 0.49.
- Lab: photo 66.6, typed 32.0, grams 47.6, 30 of 498 answers unreadable (was 1).
- Weights fixed (273 to 113, beating the constant 147), chain 142 to 134, typed 126 to 107 from the MM-Food typed chats with grams; photo unchanged.

### 2026-10-05: Fix 2 code, Kaggle notebooks, README

- In: 5,000 USDA typed meals with no photo; two thirds one food, the rest two to six; each kept food shows its full name once, then its short name ("coffee" for "Coffee, brewed").
- In: a fifth of USDA foods held out; real-world rows list their "unseen" foods; `test.py` splits the typed error into seen and unseen meals.
- In: `--no-usda`, the `runs/none` control, raw text of unreadable answers saved; reply cap 700 to 1,000 for busy lab plates.
- In: `notebooks/quick_check`, `notebooks/full_run`, the Kaggle CLI workflow, the README.

### 2026-10-05 18:52 to 2026-10-06 01:37: Fix 2 run, `runs/full_usda` and `runs/none`

- Model A (`full_usda`): 22,410 chats, loss 0.63 to 0.30. Control (`none`): 5,462 chats, loss 0.80 to 0.41.
- Model A, real world: photo 131.4, chain 134.2, typed 72.1 (seen 58.3, unseen 87.5), grams 113.5, names 0.60 / 0.49; lab: photo 63.3, typed 26.4, grams 46.2, 32 unreadable.
- Control, real world: photo 164.9, chain 168.3, typed 143.3 (seen 139.4, unseen 147.8), grams 148.6, names 0.43 / 0.46, 3% unreadable; lab: photo 65.4, typed 33.6, grams 40.0.
- Typed error fell by a third; Model A beat Fix 1 on 965 of 1,478 meals; pure coffee, tea, and water meals went from 157 to 16; meals over 600 kcal from 255 to 200.
- Photo path identical to Fix 1 (443 of 1,478 photo totals are the same numbers).
- New fault: 86 answers of exactly 0 kcal, 30 of them wrong (beer, wine, tea with milk or sugar).
- Since: a progress line every 100 photos in `test.py`, [pipelineOptimizing.md](pipelineOptimizing.md), this document. Nothing removed.
- 2026-10-15: the four speed cuts went in (no control retrain, both GPUs for tests, 16-bit numbers, batches of 8 at learning rate 2e-4), and `notebooks/change_check` was added to prove a change before a full run.
- 2026-10-19: `change_check` ran twice: batched float16 training works (peak 12.9 GB at batch 8, so the default became 4), both GPUs test at once, but repeat penalty 1.2 broke the JSON on ordinary photos (2 of 5 readable), so it went back to 1.05 and a no-repeat brake (`NO_REPEAT`, `--ngram`) is the next thing to measure.

## Scores over time

Real world, 1,478 phone photos. Name scores marked * use the old letter-inside rule.

| Score | Base 09-27 | `full` 09-28 | `full_portions` 10-05 | `full_usda` 10-06 | `none` control 10-06 | Constant guess |
|---|---|---|---|---|---|---|
| Photo -> kcal error | 245 | 130 | 131 | 131 | 165 | 197 |
| Chain -> kcal error | 229 | 142 | 134 | 134 | 168 | 197 |
| Typed -> kcal error | 256 | 126 | 107 | 72 | 143 | 197 |
| Weight error | 156 g | 273 g | 113 g | 114 g | 149 g | 147 g |
| Names right / found | 0.47 / 0.43* | 0.45 / 0.45* (0.52 / 0.52 rescored) | 0.60 / 0.49 | 0.60 / 0.49 | 0.43 / 0.46 | n/a |
| Photo answers readable | 94% | 100% | 100% | 100% | 97% | n/a |

Lab, 498 Nutrition5k plates; the base model's lab test used 20 rows.

| Score | Base | `full` | `full_portions` | `full_usda` | `none` control |
|---|---|---|---|---|---|
| Photo -> kcal error | 136 | 80 | 67 | 63 | 65 |
| Typed -> kcal error | 220 | 33 | 32 | 26 | 34 |
| Weight error | 109 g | 51 g | 48 g | 46 g | 40 g |
| Names right / found | 0.47 / 0.20* | 0.75 / 0.69* | 0.70 / 0.69 | 0.71 / 0.67 | 0.71 / 0.67 |
| Photo answers readable | 90% | 99.8% | 94% | 94% | 93% |

## Why the model performs the way it does

### The photo path answers in buckets, and the food list is the bottleneck

- Over 1,478 photos, Model A gives only 37 different kcal totals; 150, 250, 350, 600, and 50 cover 73% of answers, and 92% of its grams are multiples of 50.
- That is MM-Food's style (94% of its weights are multiples of 50, and its macros are another model's guesses), and MM-Food is 6,000 of the 8,730 photo rows, so the student learned the teacher's rounding.
- Its scale is right: median answer 250 kcal and 250 g against a true 244 and 240, too high on 728 of 1,478 meals; the 131 is spread inside buckets plus wrong foods, not a bias.
- The control, trained on lab plates only, answers finely (246 totals, 3% round grams) but too small (median 121 g against 240), because lab plates are small.
- Fix 1 and Fix 2 landed on the same photo numbers (131.5 and 131.4) with 443 identical totals, so the photo path is saturated on this data.
- Given the true food list as text the error is 72; given the model's own list, 134; so the math is fine and the list is wrong: half the foods missed (recall 0.49, against 0.67 on lab plates it trained on), coarse names, round grams.
- Better photo calories now mean better recognition and portions, not better math.

### The typed path learned rules, not facts

- The 5,000 practice meals use 6,517 names, 5,272 of them once, so the model learned from the words in a name.
- That worked: 72 kcal, and the seen-to-unseen gap (58 against 87) is small next to the control's (139 against 148), so the gain is skill, not memory.
- Only 73 practice meals were under 10 kcal, nearly all plain drinks of 240 to 480 g, which taught "a liquid with big grams and no food words is zero".
- Right for black coffee, tea, and water (157 to 16); wrong for beer, wine, and sweet tea: beer appeared alone once, as a diet root beer at 0 kcal, wine three times, "with milk" and "sweetened" about once each.
- Fix 1 never saw a zero-calorie meal, since lab plates hold none, so it gave coffee 150 to 200 kcal.
- Favorites remain (319, 662, 501, and 33 answers near the old 1,018), and meals over 600 kcal are still 200 off, because large totals were rare in practice and the model seldom answers above about 660.
- Lab typed error is 26 against 72 real: lab uses 195 short names and small plates; real meals use 1,095 USDA names, drinks, and restaurant sizes.

### Unreadable answers are runaway lists on crowded plates

- 32 of 498 lab answers cannot be parsed; each lists about 60 foods, repeats itself, and stops mid-word at the 1,000-token cap.
- Their plates hold a median of 18 true foods; readable ones hold 3; Nutrition5k answers with 30 or more foods taught very long lists, and the 1.05 repeat penalty is too gentle to end one.
- Raising the cap from 700 to 1,000 did not help, because the list never ends; real-world photos have 0 unreadable answers.

### Lab and real world are different tests

- Lab: a fixed overhead camera, one cafeteria, 195 names, measured macros, and 2,731 training plates of the same kind; 63 kcal.
- Real world: phone angles, 1,095 names, drinks, home and restaurant portions; 131 kcal.
- The control gets 65 on lab and 165 on phones, which is how little lab skill transfers on its own.

### What the baselines say

- Fine-tuning halved every calorie error against the base model (245 to 131 on photos, 256 to 72 typed).
- The model beats the constant guess widely on calories (131 against 197) and narrowly on weights (113 g against 147 g).
- Each data source fixed the path it was aimed at and left the other alone: MM-Food bought 34 kcal, 35 g, and 0.17 precision on phone photos; USDA meals bought 35 kcal on typed meals (107 to 72) and nothing on photos.
- Train loss fell within every run (to 0.23, 0.22, 0.30 with more varied data, 0.41 with less), so the model learns what it is shown; what it is shown is the limit.

## Where that points

- Typed: deal drinks with calories and large meals more often in `process.py`'s `usda()`.
- Photo: the ceiling is MM-Food's round, estimated labels and the model's recall; options are better phone-photo labels, matching ingredients to portions, or a bigger base model.
- Lab: a strong repeat penalty mangles JSON, so the brake under test is `no_repeat_ngram_size`; an item cap is the fallback.
- Speed: see [pipelineOptimizing.md](pipelineOptimizing.md) before the next run.
