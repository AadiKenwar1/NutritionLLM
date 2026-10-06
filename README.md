# NutritionLLM

One small vision-language model that reads a meal photo, or a typed meal, and returns the foods,
their grams, and the calories and macros. It is small enough to run on a phone.

- Base model: [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M), 450 million parameters.
- Fine-tuned with LoRA, so only small add-on layers learn (about 7 million weights, a 28 MB file).
- One model handles every task, and each task has its own score, so weak spots are easy to see.

| You give it | It answers with, as JSON |
|---|---|
| A meal photo | the dish name, each food with its grams, and totals (kcal, protein, carbs, fat) |
| A typed meal, like "2 eggs, 1 slice toast" | totals (kcal, protein, carbs, fat) |

The exact prompts are in `predict.py`.

## The data

Four sources. None of it is in git; `download.py` fetches about 16 GB into `data/`.

| Source | What it is | How we use it |
|---|---|---|
| Nutrition5k | Cafeteria dishes photographed from above, every ingredient weighed, macros measured | Training (2,731 dishes) and the lab test (498 dishes) |
| MM-Food-100K | Phone-style food photos with a dish name, ingredients, rough portions, and macros estimated by a bigger model | Training only, the first 6,000 photos; its macros are guesses, not measurements |
| SNAPMe | 1,478 real meals photographed with phones by study participants, with foods, grams, and nutrients from the USDA table | The real-world test only; never trained on |
| USDA FNDDS 2017-2018 | The US food table: about 7,000 foods with macros per 100 g and usual serving sizes | 5,000 typed practice meals; a fifth of the foods are held out of them |

Why these four: lab plates teach exact weights and measured macros, phone photos teach what real
photos look like, the USDA table teaches food names, drinks, and big totals, and SNAPMe shows how
well it all works on real meals.

## How it works

1. **Build** (`process.py`): writes `train.jsonl`, `test_lab.jsonl`, and `test_real_world.jsonl` into `data/processed/`. Each real-world row also lists its "unseen" foods, the ones no typed practice meal used.
2. **Train** (`train.py`): every row becomes one or two short chats, photo -> dish, foods, grams, totals, and typed food list -> totals. The LoRA layers save to their own folder under `runs/`.
3. **Score** (`test.py`): asks three questions about every test photo, then reports readable answers, food names right and found, weight error, and calorie and macro errors to `results/`.
4. **Compare**: two yardsticks, the stock model and "always guess the middle value". A fix counts only if it beats both. Today's numbers and next steps are in [docs/proposedFixes.md](docs/proposedFixes.md).

The three questions per test photo:

- **photo**: foods, grams, and totals straight from the image.
- **chain**: the model's own food list, fed back in as a typed meal.
- **text**: the true food list typed in. On the real-world test it is split into "seen" and "unseen" meals, to tell learning from memorizing the table.

## Running it

### Setup

```
python -m venv venv
venv\Scripts\activate            # Windows; elsewhere: source venv/bin/activate
pip install torch torchvision    # the build that matches your machine; Kaggle already has them
pip install -r requirements.txt
```

- Python 3.13 is what the project was tested with.
- Downloading, building the files, reading results, and editing code need no GPU.
- `train.py`, `test.py`, and `predict.py` need a GPU to be practical, so they run on Kaggle (below).

### The scripts

```
python download.py                              # all four sources, or some: python download.py nutrition5k snapme
python process.py                               # builds data/processed/
python train.py                                 # everything, saves to runs/full_usda
python train.py --mmfood none --no-usda         # Nutrition5k only, saves to runs/none
python test.py --model runs/full_usda --data data/processed/test_real_world.jsonl --limit 0
python predict.py --model runs/full_usda --image photo.jpg
python predict.py --model runs/full_usda --text "2 eggs, 1 slice toast"
```

- `download.py` skips files it already has, so re-run it after a crash. Some Nutrition5k and MM-Food links are dead; that "failed" count is normal, and `process.py` skips those rows.
- `train.py` switches: `--mmfood full|labels|none` keeps MM-Food's estimated totals, keeps only its names and grams, or drops it. `--no-usda` leaves the typed meals out. Also `--epochs`, `--lr`, `--rank`, `--out`.
- `--limit N` on `train.py` and `test.py` tries a few rows first. `test.py` scores 20 rows by default; `--limit 0` means all.
- `test.py` writes `results/<run>_<test>.json` (the report) and `.jsonl` (every row, with the raw text of unreadable answers).

### Kaggle

Model runs use Kaggle's free GPU hours, started from this folder with the Kaggle CLI. The notebooks
are in `notebooks/`: `quick_check` is a two-minute run that proves the setup works, and `full_run` is
the real thing.

One-time setup:

```
uv tool install kaggle        # or: pip install kaggle
kaggle auth login             # opens a browser
```

- Kaggle gives GPU and internet only to phone-verified accounts.
- Notebooks belong to one Kaggle account, so put your username in the `id` line of each `notebooks/*/kernel-metadata.json`.
- On Windows, run `setx PYTHONUTF8 1` once and reopen the terminal; without it the `logs` and `output` commands stop at the first progress bar with a `'charmap' codec` error.
- Kaggle clones the code from GitHub, not from your disk. A fork must change the clone URL in the first code cell of each notebook.

Each run:

```
git push                                                        # Kaggle clones the code from GitHub
kaggle kernels push -p notebooks/full_run                       # upload and start; about 8 hours
kaggle kernels status <you>/nutritionllm-full-run               # QUEUED, RUNNING, then COMPLETE or ERROR
kaggle kernels logs <you>/nutritionllm-full-run --follow        # live log, once it is RUNNING
kaggle kernels output <you>/nutritionllm-full-run -p results    # downloads output.zip and the log
unzip -o results/output.zip                                     # PowerShell: Expand-Archive results\output.zip -DestinationPath . -Force
```

- `<you>` is your Kaggle username; `push` also prints the notebook's web page, which shows the same status and log.
- `full_run` clones the repo, downloads the data, does a 40-row quick try with a safety stop, trains and tests the two "Model" cells (edit those to change the experiment), and zips `results/` plus the run folders.
- The zip holds everything the run made, `results/` and `runs/`, so unzipping at the project root puts them in place; each test's report is `results/<run>_<test>.json`.
- If the status ends in `ERROR`, `output` still brings down the log, and the error is at its end.
- The two-minute check is the same commands with `notebooks/quick_check` and `nutritionllm-quick-check`; it leaves `quick_check.txt` in the download.
- Kaggle gives about 30 GPU hours a week and stops a run at 12 hours; a stopped run may save nothing, so keep runs under that.
- Both notebooks `pip uninstall torchao` after installing; Kaggle's old copy makes peft refuse to load.

## Folders

- `data/` the four sources and the `processed/` files; not in git.
- `runs/` one folder per training run; the weight files are not in git.
- `results/` one report and one per-row file per test; not in git.
- `notebooks/` the two Kaggle notebooks and their settings files.
- `docs/proposedFixes.md` the plan: what is wrong today and what to try next.
- `AGENTS.md` the rules for anyone, human or AI, editing the code.
