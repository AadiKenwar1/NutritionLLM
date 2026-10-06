# NutritionLLM

A phone-sized vision-language model. Meal photo or typed meal in; foods, grams, and macros out.

- Base: [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M), 450M parameters.
- LoRA fine-tune: 7M trained weights, a 28 MB adapter.
- One model for every task, one score per task.

| Input | Output (JSON) |
|---|---|
| Meal photo | dish, foods with grams, totals (kcal, protein, carbs, fat) |
| Typed meal | totals |

Prompts: `predict.py`.

## Data

Not in git. `download.py` fetches about 16 GB into `data/`.

| Source | What | Use |
|---|---|---|
| Nutrition5k | Lab dishes from above, weighed, macros measured | Train (2,731), lab test (498) |
| MM-Food-100K | Phone photos; dish, ingredients, rough portions, macros estimated by another model | Train only, first 6,000 |
| SNAPMe | 1,478 real phone meals; foods, grams, USDA nutrients | Real-world test only |
| USDA FNDDS 2017-2018 | 7,000 foods; macros per 100 g, serving sizes | 5,000 typed practice meals; 1/5 of foods held out |

## Pipeline

1. `process.py` writes `train.jsonl`, `test_lab.jsonl`, `test_real_world.jsonl` to `data/processed/`. Real-world rows list "unseen" foods: ones no typed meal used.
2. `train.py` turns rows into chats (photo -> dish, foods, grams, totals; food list -> totals) and saves a LoRA adapter under `runs/`.
3. `test.py` scores each test photo three ways and writes `results/<run>_<test>.json` (report) and `.jsonl` (every row).
4. Compare against the stock model and the constant guess in [docs/proposedFixes.md](docs/proposedFixes.md). A fix must beat both.

Scores per photo:

- **photo**: foods, grams, totals from the image.
- **chain**: the model's own food list, fed back as text.
- **text**: the true food list as text. Real-world test splits this into seen and unseen meals.

Report fields: parsed, food precision and recall, grams error, kcal and macro error.

## Setup

```
python -m venv venv
venv\Scripts\activate            # or: source venv/bin/activate
pip install torch torchvision    # your machine's build; Kaggle has them
pip install -r requirements.txt
```

- Python 3.13.
- Download, process, and reading results need no GPU.
- `train.py`, `test.py`, `predict.py` need a GPU: run them on Kaggle.

## Scripts

```
python download.py                              # or: python download.py nutrition5k snapme
python process.py
python train.py                                 # -> runs/full_usda
python train.py --mmfood none --no-usda         # Nutrition5k only -> runs/none
python test.py --model runs/full_usda --data data/processed/test_real_world.jsonl --limit 0
python predict.py --model runs/full_usda --image photo.jpg
python predict.py --model runs/full_usda --text "2 eggs, 1 slice toast"
```

- `download.py` skips existing files; re-run after a crash. Some links are dead; `process.py` skips those rows.
- `--mmfood full|labels|none`: keep MM-Food's estimated totals, keep only names and grams, or drop it.
- `--no-usda`: drop the typed meals. Also `--epochs`, `--lr`, `--rank`, `--out`.
- `--limit N`: first N rows. `test.py` defaults to 20; `0` means all.

## Kaggle

Notebooks in `notebooks/`: `quick_check` (2 min, proves setup) and `full_run` (the real run).

Once:

```
uv tool install kaggle        # or: pip install kaggle
kaggle auth login
```

- GPU and internet need a phone-verified Kaggle account.
- Put your username in the `id` line of each `notebooks/*/kernel-metadata.json`.
- Forks: change the GitHub clone URL in each notebook's first code cell.

Each run:

```
git push                                                        # Kaggle clones from GitHub
kaggle kernels push -p notebooks/full_run                       # ~8 hours
kaggle kernels status <you>/nutritionllm-full-run
kaggle kernels logs <you>/nutritionllm-full-run --follow
kaggle kernels output <you>/nutritionllm-full-run -p results    # fix2.zip
unzip -o results/fix2.zip                                       # PowerShell: Expand-Archive results\fix2.zip -DestinationPath . -Force
```

- `full_run`: clone, download, 40-row quick try with a safety stop, two "Model" cells, zip. Edit the Model cells to change the experiment.
- Kaggle kills runs at 12 hours and may save nothing.
- Notebooks `pip uninstall torchao`; Kaggle's old copy breaks peft.

## Folders

- `data/`, `results/`: not in git.
- `runs/`: one folder per run; weights not in git.
- `docs/proposedFixes.md`: current numbers and next steps.
- `AGENTS.md`: code rules.
