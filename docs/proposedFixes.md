# Proposed fixes

Plan after the first full test of `runs/full` (Nutrition5k + MM-Food). Numbers come from the 1,478
SNAPMe phone photos ("real world") and 498 Nutrition5k lab photos in `results/`, checked 2026-09-28.

## Where we are

Real-world phone photos, plus the fine-tuned model on lab photos. "Constant" means always guessing
the training median (240 g, 244 kcal).

| Score | Base | Fine-tuned | Constant | Fine-tuned, lab |
|---|---|---|---|---|
| Foods named right (precision) | 0.47 | 0.45 | n/a | 0.75 |
| Foods found (recall) | 0.43 | 0.45 | n/a | 0.69 |
| Weight error per meal | 156 g | 273 g | 147 g | 51 g |
| Photo -> calorie error | 245 kcal | 130 kcal | 197 kcal | 80 kcal |
| Typed meal -> calorie error | 256 kcal | 126 kcal | 197 kcal | 33 kcal |

- Fine-tuning cut calorie errors roughly in half and beats a constant guess. That worked.
- Food naming did not improve. Fixes 1 and 3.
- Weight error got worse for a mechanical reason: the model stopped writing grams on phone photos, and a missing weight counts as zero. Fix 1.
- The base model's 156 g is no better than the constant guess. So the bar for weights is "beats 147 g", not "beats 273 g".
- Typed meals are far worse on real meals (126) than on lab plates (33). Fix 2.

## Before any retrain

**Print constant-guess lines in every report.** `test.py` adds the weight and calorie error of
always guessing the training median. A fix counts only if it beats those lines.

**Run `--mmfood none` as a control.** Nutrition5k only, no new code. MM-Food stays in the plan
either way: it is our only source of phone-style photos and of the bigger model's answers
(distillation). The run decides how to use MM-Food, not whether.
- Today we cannot tell how much of the calorie gain came from MM-Food's copied macros versus Nutrition5k's measured ones. This run separates them.
- It also tests Fix 1's theory for free: if this model writes grams on phone photos, the missing grams came from MM-Food's answer shape.
- If `full` beats `none` on real-world calories, the copied macros help. Fix 1 makes them stronger.
- If `none` matches or wins, the copied macros are not teaching calories yet. Do Fix 1 anyway, since grams are the likely missing piece. If Fix 1 still does not beat `none`, fall back to `labels`: keep MM-Food's dish and food names, teach calories from measured data only. Caution: `labels` rows have no totals, so the model could learn to skip totals on phone photos the way it learned to skip grams.

## Fix 1: give phone photos weights

**Problem**
- On phone photos the model names foods but skips grams: 7 of 4,160 foods got a weight. On lab photos, every food did.
- Cause: the two training sets taught two answer shapes. Nutrition5k rows have grams and totals, no dish. MM-Food rows have a dish and food names, no grams. MM-Food is 5,999 of 8,730 rows, so "phone photo means no grams" won.
- Without grams, the chain answer (foods -> calories) is a guess from names alone. The 273 g is mostly this artifact.
- MM-Food has an unused `portion_size` column. 5,975 of 6,000 rows parse as "name:NNNg". 94% of the weights are multiples of 50 g. Its median meal is 350 g, against 240 g for real-world test meals.

**Fix**
- `process.py` keeps `portion_size` as a `portions` list next to `foods`. Rows that do not parse get an empty list.
- `train.py` builds MM-Food photo answers from the portion list, so every item has grams. Dish and totals stay.
- Portions, not ingredients: only half the ingredients can be matched to a weight, and a half-weighted answer keeps the "skip grams" habit alive. The cost is coarse names like "vegetables 150 g".
- MM-Food typed chats get grams too. `full` and `labels` both use the portion list.
- Retrain into its own folder, then rerun both tests.

**Success**
- Chain calorie error on real-world photos (141 today) is the target. Photo calorie error (130) must not get worse.
- Weight error must beat 147 g. Landing at 150 g is not a win.
- Lab typed error (33) is the canary. If it climbs, stop building typed chats from MM-Food rows.
- Food naming may dip below 0.45. If it drops a lot, try the ingredient list with matched grams instead.

**Risk:** MM-Food weights are round guesses on heavy restaurant plates. The model may learn "phone
photo means about 350 g" and overshoot home meals.

## Fix 2: teach the typed task drinks, USDA names, and big meals

**Problem**
- Zero-calorie drinks: "Coffee, brewed, 480 g" gets 170 to 210 kcal. The true answer is about 5. The answer grows with grams, as if coffee were 0.3 kcal per gram.
- 204 of 1,478 real meals are black coffee, tea, or water. They are 25% of the typed error. Fixing them alone takes 126 down to 95.
- Big meals: the model has a favorite answer near 1,018 kcal. It gave it 43 times, to meals that truly ranged from 83 to 2,221. Typed error on meals over 600 kcal is 263.
- Names: real meals use USDA names like "Egg omelet or scrambled egg, made with oil" (1,095 distinct). Lab training uses 195 short names. The lab typed score of 33 hides this gap.
- Cause: typed practice rows come only from the photo meals. Lab plates are small, use 195 names, and never hold a zero-calorie drink. MM-Food rows have no grams.

**Fix**
- Add `data/usda` from FoodData Central's "FNDDS 2021-2023 (JSON)" download (`FoodData_Central_survey_food_json_2024-10-31.zip`, 4 MB). SNAPMe's true numbers come from this same table. Each food has macros per 100 g and real portions like 1 cup = 240 g.
- `download.py` gets a fourth step. `process.py` writes photo-free rows with task "typed": food names, amounts, true totals.
- Amounts come from the food's own portion list, so a coffee is 240, 360, or 480 g, never 7 g.
- About a third of the rows are meals of two or three foods with summed totals, so the model practices adding and sees totals over 1,000 kcal.
- Half the rows use the short name (before the first comma, lower case), so "coffee" works as well as "Coffee, brewed".
- Hold out 20% of the foods. `process.py` saves the list. `test.py` splits the typed error into all-seen meals and meals with an unseen food.
- Cap at about 5,000 rows, near the lab chat count, so the typed task does not crowd out photos. `train.py` has no other balance control.
- `train.py` builds only the typed chat for "typed" rows. Retrain, then rerun both tests.

**Success**
- Typed error on real meals (126), split seen vs unseen. If unseen is much worse, the model memorized the table, and only the unseen number is honest.
- Coffee, tea, and water land near zero. Meals over 600 kcal improve on 263.
- Photo calorie error (130) and food naming (0.45) must not move.

**Risk:** training on the test's own lookup table inflates the seen score. Single foods and short
combos may not cure the canned 1,018 answer.

## Fix 3: make the food-name score fairer

**Problem**
- `test.py` counts a hit if one name sits inside the other, letter by letter.
- That calls "cherry tomatoes" wrong against "Tomatoes, raw", but "egg" right against "eggplant" and "cheese" right against "cheesecake". Unfair both ways.
- Without a fair score we cannot tell whether Fixes 1 and 2 helped naming.

**Fix**
- Split names into words and make each word singular ("tomatoes" -> "tomato").
- Rule A: a hit if every word of the shorter name appears in the longer one. Catches "latte" in "Coffee, Latte, flavored".
- Rule B: cut both names at the first comma, then apply rule A. Catches "cherry tomatoes" against "Tomatoes, raw".
- Either rule counts. The letter-inside rule goes away. No retrain.
- On the saved real-world results: precision 0.454 -> 0.518, recall 0.450 -> 0.517.
- Still wrong: "chicken" matches "chicken nuggets", and "toast" no longer matches "toasted". Accepted.

**Rescoring:** only the fine-tuned real-world run has all its per-row results saved. The other
three (base lab, base real world, fine-tuned lab) saved 20 rows. Rerun them with `--limit 0` on
Kaggle and copy back the `.jsonl` as well as the `.json`. From now on, always copy back both.

## Order

1. Fix 3 plus the three reruns. A few lines of code and a few hours of inference.
2. The `--mmfood none` control. It can run while Fix 1's code is written.
3. Fix 1.
4. Fix 2. It needs the new dataset and the hold-out split.

Every retrain reruns both tests, with the constant-guess lines in every report.
