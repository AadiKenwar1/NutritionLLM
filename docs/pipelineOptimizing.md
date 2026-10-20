# Pipeline optimizing

Where a Kaggle full run spends its time, why, and four cuts that would take it from about 6.5 hours
to about 1.5. Numbers come from the Fix 2 run of `notebooks/full_run` on 2026-10-05: one Tesla T4,
one chat per training step, 32-bit numbers. Nothing here changes what the model learns or how it is
scored; it only changes how long we wait. Status 2026-10-15: all four cuts are in the code and
notebooks, checked by `notebooks/change_check`; the next full run measures the new time.

## Terms

- GPU, or graphics card: the chip that does the model's math, millions of multiplications at once; Kaggle's free machine has two Tesla T4 cards (2018, 15 GB each).
- Token: a word piece; the model writes answers one token at a time, and every token is a full pass through the model.
- Batch: how many chats the model learns from in one step; today it is one.
- float32, float16, bfloat16: how many bits store each number in the model; 32 bits is slow and exact, the two 16-bit kinds are fast and slightly rounded (see cut 3).
- Control: Model B, trained on Nutrition5k only, which exists only to show what the extra data bought.

## Where the time goes

| Step | Work | Measured | Time |
|---|---|---|---|
| Clone, install, download 16 GB, process, 40-row quick try | | | 12 min |
| Model A training | 22,410 chats | 0.3 s per chat | 1 h 45 min |
| Model A real-world test | 1,478 photos, 3 answers each | 3 s per photo | 1 h 15 min |
| Model A lab test | 498 photos, 3 answers each | 5 s per photo | 40 min |
| Model B training | 5,462 chats | 0.3 s per chat | 27 min |
| Model B tests | same two tests | | 2 h |
| Zip | | | 1 min |

- Total about 6.5 hours, and the control (Model B) is about 3 of them.
- Testing costs more than training: 4 of the 6.5 hours.
- Lab photos take longer per photo because they hold more foods, and 32 of 498 answers ramble to the 1,000-token cap at 30 s or more each.
- Kaggle's page shows `Output 0 B` until the last cell, because only the final zip lands in `/kaggle/working`.

## Why each part is slow

- `train.py` feeds one chat per step ([train.py:104](../train.py#L104)); a T4 could handle 8 to 16 at once, so most of the card waits each step.
- The T4 cannot do bfloat16 in hardware, so [predict.py:74](../predict.py#L74) falls back to float32, the slow lane, for training and testing alike.
- Each test writes three answers per photo, one token at a time, and a photo answer is 100 to 300 tokens.
- Unreadable answers run to the 1,000-token cap ([predict.py:85](../predict.py#L85)) and cost 30 s or more each.
- The model's "short conv" layers lack their fast GPU add-on (the `causal_conv1d` warnings in the log), so a slower built-in version runs.
- Everything runs twice: Model A, then the control.
- `.to("cuda")` ([predict.py:75](../predict.py#L75)) picks the first card, so the second idles the whole run.

## Cut 1: stop retraining the control

**Problem**
- 3 of the 6.5 hours redo a calculation whose answer we already have.
- Model B comes out the same every run, because nothing it depends on changes between experiments.

**Fix**
- Keep `results/none_test_real_world.json`, `results/none_test_lab.json`, and `runs/none` from the 2026-10-05 run.
- Skip the Model B cell in `notebooks/full_run`.
- Redo Model B once whenever `data/processed/test_*.jsonl` or the scoring in `test.py` changes, or A is compared against a stale number.
- Done: `notebooks/full_run` has no control cell; its first cell says when and how to rerun the control.

**Solves:** the repeat work disappears. **Does not touch:** Model A's own time. **Risk:** none. **Saves:** about 3 hours.

## Cut 2: run the two tests on the two cards

**Problem**
- Half the hardware sits idle while the other half works for 5 hours.
- Each model's two tests run one after the other, 1 h 15 min plus 40 min.

**Fix**
- Setting `CUDA_VISIBLE_DEVICES` on a command tells that program which card it may see, so two programs can each take one card.
- One notebook line runs both tests and waits for both; the background one logs to a file so its output is not lost:

```
!CUDA_VISIBLE_DEVICES=0 python test.py --model runs/full_usda --data data/processed/test_real_world.jsonl --limit 0 > results/full_usda_test_real_world.log 2>&1 & \
 CUDA_VISIBLE_DEVICES=1 python test.py --model runs/full_usda --data data/processed/test_lab.jsonl --limit 0; wait
```

- The same trick can train Model B on card 2 while Model A tests on card 1, if the control is being redone.
- Each process loads its own copy of the model, about 1 GB per card, well inside 15 GB.
- Done: the Model cell runs both tests in one shell line; the lab test logs to `results/<run>_test_lab.log`, printed when both finish.

**Solves:** two jobs overlap. **Does not touch:** the speed of any single test; the wait becomes the longest job, not the sum. **Risk:** none, each process has its own card and its own output files. **Saves:** about 40 minutes per model.

## Cut 3: shorter numbers on the T4

**Problem**
- Every number in the model is stored with 32 bits, so every multiplication runs in the card's slow lane and the model is 1.8 GB.
- Writing one token means reading the whole model from memory once, so size sets the speed of testing.

**Why we were not already**
- There are two 16-bit formats: bfloat16 holds huge numbers with little detail and almost never breaks, so it is the usual safe choice; float16 has more detail but tops out near 65,000, and a bigger number overflows into garbage.
- The T4 is from 2018, before bfloat16 existed in GPU hardware, so it can only do float16 fast.
- `predict.load()` asks for bfloat16 on newer cards and otherwise falls back to float32 because float32 always works; float16 was never tested, not rejected.

**Fix**
- Testing: in `predict.load()`, use float16 on a CUDA card below capability 8. One line.
- Training: keep the weights in float32 and do the multiplications in float16 with `torch.autocast`, plus a grad scaler so tiny updates do not round to zero. A few lines in `train.py`.
- Check with the quick try: `parsed` must stay at 1.0 on the 5 real-world photos and a few raw answers must read as sane JSON.
- Done: `predict.load()` picks float16 below capability 8, and `train.py` keeps a grad scaler that is on only for float16.

**Solves:** all math gets faster, and the model shrinks to 0.9 GB, so each token takes about half the time. **Does not touch:** the one-chat-at-a-time problem. **Risk:** overflow to garbage, caught by the quick try. **Saves:** about half of test time, roughly an hour per model, and some training time.

## Cut 4: train in groups

**Problem**
- One chat is too small a job for a T4, which can do trillions of operations a second, so the card spends most of each step waiting for instructions.

**Fix**
- Build each step from 8 chats (16 if memory allows), padded to the longest chat, with the padding masked out of the loss (`-100`).
- Keep photo chats and typed chats in separate groups, since the processor handles a batch of all-photo or all-text chats more simply than a mix.
- Fewer, bigger steps may need a different learning rate; compare the final loss with the one-at-a-time run (Model A ended near 0.28, Model B near 0.41) and the test scores must match or beat it.
- About 30 lines in `train.py`, proven with `--limit` before a full run.
- Done: `train.py --batch 4` (the default; 8 peaked at 12.9 GB of 15 on the quick try) at learning rate 2e-4; `encode()` pads a batch and masks prompts and padding.

**Solves:** the card stays busy during training, 1 h 45 min to about 30 min. **Does not touch:** the download, the tests, or the control. **Risk:** learning changes shape, so the loss and scores must be compared. **Saves:** about 1 h 15 min.

## Leave alone for now

- The fast add-on for the short conv layers (`causal_conv1d`) has to be compiled on Kaggle, which is slow and often fails, for maybe a 10 to 30 percent gain.
- Saving the data as a Kaggle dataset to skip the download saves only a few minutes.

## What none of them solve

- The tests still write about 6,000 answers (1,976 photos times 3), many 100 to 300 tokens long.
- Testing fewer photos (`--limit`) or capping answers shorter (`max_new_tokens`) would cut that, but both change what we measure, so neither is a cut.

## Order

- All four went in on 2026-10-15 and passed `notebooks/change_check` on 2026-10-19: batched float16 training (1.0 s per step of 4 chats, 6.9 GB), both GPUs testing at once (4 s per photo each).
- The next full run must end near the old loss (0.30) and match or beat the 2026-10-06 scores, or cut 4 comes out.

| Cut | Waste it removes | Phase it helps | Saves | Effort |
|---|---|---|---|---|
| 1. Skip control | repeat work | all of Model B | 3 h | skip a cell |
| 2. Both cards | idle hardware | tests | 40 min per model | notebook lines |
| 3. Shorter numbers | slow-lane math | tests most, training too | about 1 h | 1 line, more for training |
| 4. Group training | starved card | training only | about 1 h 15 min | about 30 lines |

With all four, a run is about 1 to 1.5 hours instead of 6.5.
