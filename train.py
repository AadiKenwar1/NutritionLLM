"""Fine-tunes the model with LoRA on train.jsonl and saves the adapter to its own run folder.
This is the learning step of NutritionLLM. Each training row becomes one or two short chats built
with the same prompts as predict.py: a photo chat (image -> dish, foods, grams, totals) and a typed
meal chat (food list -> totals), so the model learns exactly what test.py asks for. MM-Food rows
answer with their portion list, so every food has grams and the model never learns that phone photos
come without weights; rows with no readable portions are skipped. --mmfood picks how MM-Food rows
are used: "full" keeps their model-estimated macros, "labels" keeps only the dish, foods, and grams,
"none" drops them. USDA rows are typed meals with no photo, so each becomes one typed chat; --no-usda
drops them. Chats train in batches (--batch) of one kind at a time, photo or typed, padded to the
longest chat in the batch; one chat per step left the GPU mostly idle. On a float16 GPU a grad scaler
keeps small gradients from rounding to zero. Only the small LoRA layers train, so the run folder
holds a few tens of megabytes that predict.load() applies on top of the base model. Look at the
printed loss: it should fall steadily, and a run that ends near its starting loss learned nothing.
Prove a change with --limit on a few rows before a full run on a GPU.
"""
import argparse
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup

import predict

# Layers that get LoRA weights: attention (q/k/v/out), short-conv (in), and feed-forward (w1-w3)
TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]


# The JSON the model should answer for one photo, honoring the --mmfood switch
def photo_answer(row, mmfood):
    a = row["answer"]
    if row["task"] == "full":
        return {"items": a["items"], "totals": a["totals"]}
    answer = {"dish": a["dish"], "items": a["portions"]}
    if mmfood == "full":
        answer["totals"] = a["totals"]
    return answer


# A typed-meal chat: the food list as text -> its totals; the image slot is None
def typed_chat(items, totals):
    return None, predict.text_prompt(predict.meal_text(items)), json.dumps(totals)


# Turn one training row into (image path, prompt, answer text) chats; image None means a typed meal
def chats(row, mmfood, usda):
    a = row["answer"]
    # USDA rows have no photo, so they give only a typed chat; --no-usda drops them
    if row["task"] == "typed":
        return [typed_chat(a["items"], a["totals"])] if usda else []
    # MM-Food rows are dropped by --mmfood none, and skipped when they have no readable portions
    if row["task"] == "foods" and (mmfood == "none" or not a["portions"]):
        return []
    photo = photo_answer(row, mmfood)
    out = [(row["image"], predict.PHOTO, json.dumps(photo))]
    if "totals" in photo:
        out.append(typed_chat(photo["items"], photo["totals"]))
    return out


# Tokenize a batch of chats, padded on the right, masking prompts and padding so the loss covers only answers
def encode(processor, batch):
    conversations = [[predict.user_turn(prompt, image and predict.open_image(image)),
                      {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
                     for image, prompt, answer in batch]
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt", processor_kwargs={"padding": True})
    inputs = processor.apply_chat_template(conversations, **kw)
    labels = inputs["input_ids"].masked_fill(inputs["attention_mask"] == 0, -100)
    for row, conversation in enumerate(conversations):
        prompt_len = processor.apply_chat_template(conversation[:1], add_generation_prompt=True, **kw)["input_ids"].shape[1]
        labels[row, :prompt_len] = -100
    return inputs, labels


# Shuffle the chats and cut them into batches of one kind each, photo or typed, so padding stays small
def batches(examples, size, seed):
    rng = random.Random(seed)
    photos = [e for e in examples if e[0] is not None]
    typed = [e for e in examples if e[0] is None]
    out = []
    for group in (photos, typed):
        rng.shuffle(group)
        out += [group[i:i + size] for i in range(0, len(group), size)]
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=predict.DEFAULT, help="base model to fine-tune")
    p.add_argument("--data", default="data/processed/train.jsonl")
    p.add_argument("--mmfood", choices=["full", "labels", "none"], default="full")
    p.add_argument("--usda", action=argparse.BooleanOptionalAction, default=True, help="use the USDA typed meals")
    p.add_argument("--out", help="run folder; default runs/<mmfood>, with _usda added when --usda is on")
    p.add_argument("--limit", type=int, default=0, help="train on only the first N rows; 0 means all")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=4, help="chats per training step")
    p.add_argument("--lr", type=float, default=2e-4, help="was 1e-4 with one chat per step")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank; alpha is twice this")
    a = p.parse_args()
    out = a.out or f"runs/{a.mmfood}{'_usda' if a.usda else ''}"
    with open(a.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:a.limit or None]
    examples = [c for row in rows for c in chats(row, a.mmfood, a.usda)]
    steps = a.epochs * len(batches(examples, a.batch, 0))
    print(f"{len(rows)} rows -> {len(examples)} chats in {steps} steps, saving to {out}")
    torch.manual_seed(0)
    model, processor = predict.load(a.model)
    dtype = model.dtype
    # float16 gradients can round to zero; the scaler prevents that and does nothing for bfloat16 or float32
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=TARGETS))
    model.print_trainable_parameters()
    model.train()
    params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    sched = get_linear_schedule_with_warmup(opt, steps // 20, steps)
    os.makedirs(out, exist_ok=True)
    losses, start = [], time.time()
    for epoch in range(a.epochs):
        epoch_batches = batches(examples, a.batch, epoch)
        for step, batch in enumerate(epoch_batches, 1):
            inputs, labels = encode(processor, batch)
            loss = model(**inputs.to(model.device, dtype=dtype), labels=labels.to(model.device)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
            if step % 50 == 0 or step == len(epoch_batches):
                recent = losses[-50:]
                print(f"epoch {epoch + 1} step {step}/{len(epoch_batches)} loss {sum(recent) / len(recent):.3f} "
                      f"{(time.time() - start) / len(losses):.1f} s/step", flush=True)
        model.save_pretrained(out)
        with open(f"{out}/train.json", "w") as f:
            json.dump({"args": vars(a), "loss": losses}, f)
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
