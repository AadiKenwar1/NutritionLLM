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
- `--no-usda`: drop the typed meals. `--batch`: chats per step, default 8; `--lr` defaults to 2e-4. Also `--epochs`, `--rank`, `--out`.
- `--limit N`: first N rows. `test.py` defaults to 20; `0` means all. `--penalty`: repeat penalty for answers, default `predict.PENALTY` (1.2).

## Kaggle

Three notebooks in `notebooks/`. All clone the repo from GitHub, so `git push` first.

Once:

```
uv tool install kaggle        # or: pip install kaggle
kaggle auth login
```

- GPU and internet need a phone-verified Kaggle account.
- Put your username in the `id` line of each `notebooks/*/kernel-metadata.json`.
- Forks: change the GitHub clone URL in each notebook's first code cell.

Quick check, after changing how runs are sent or which libraries they use:

```
kaggle kernels push -p notebooks/quick_check                    # ~2 min
kaggle kernels status <you>/nutritionllm-quick-check            # QUEUED, RUNNING, COMPLETE, or ERROR
kaggle kernels output <you>/nutritionllm-quick-check -p results # passed if quick_check.txt is there
```

Change check, after changing `train.py`, `predict.py`, or `test.py`:

```
kaggle kernels push -p notebooks/change_check                   # ~30 min: quick try, both GPUs, repeat-penalty sweep
kaggle kernels output <you>/nutritionllm-change-check -p results
```

- It tests the repeat penalty with the last good adapter, a private Kaggle dataset named in its `kernel-metadata.json`; make yours once with `kaggle datasets create -p .` inside a folder holding `runs/<run>/adapter_*` and a `dataset-metadata.json`.

Full run:

```
kaggle kernels push -p notebooks/full_run                       # hours; see docs/pipelineOptimizing.md
kaggle kernels status <you>/nutritionllm-full-run
kaggle kernels logs <you>/nutritionllm-full-run --follow        # once RUNNING
kaggle kernels output <you>/nutritionllm-full-run -p results    # output.zip
unzip -o results/output.zip                                     # PowerShell: Expand-Archive results\output.zip -DestinationPath . -Force
```

- `full_run`: clone, download, 40-row quick try with a safety stop, one "Model" cell that trains and then runs both tests at once (one per GPU), zip. Edit the Model cell to change the experiment.
- The control (`runs/none`) is not retrained; its Oct 6 scores are in `docs/timeline.md`. Rerun it only when the test data or `test.py` scoring changes.
- Kaggle kills runs at 12 hours and may save nothing.
- Notebooks `pip uninstall torchao`; Kaggle's old copy breaks peft.

## Folders

- `data/`, `results/`: not in git.
- `runs/`: one folder per run; weights not in git.
- `notebooks/`: the two Kaggle notebooks and their `kernel-metadata.json` settings.
- `docs/proposedFixes.md`: current numbers and next steps.
- `docs/pipelineOptimizing.md`: where a Kaggle run's time goes and four cuts to shorten it.
- `docs/timeline.md`: what was added or removed when, scores after each run, and why the model behaves as it does.
- `AGENTS.md`: code rules.
