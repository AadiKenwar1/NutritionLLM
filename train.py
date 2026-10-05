"""Fine-tunes the model with LoRA on train.jsonl and saves the adapter to its own run folder.
This is the learning step of NutritionLLM. Each training row becomes one or two short chats built
with the same prompts as predict.py: a photo chat (image -> dish, foods, grams, totals) and a typed
meal chat (food list -> totals), so the model learns exactly what test.py asks for. MM-Food rows
answer with their portion list, so every food has grams and the model never learns that phone photos
come without weights; rows with no readable portions are skipped. --mmfood picks how MM-Food rows
are used: "full" keeps their model-estimated macros, "labels" keeps only the dish, foods, and grams,
"none" drops them. USDA rows are typed meals with no photo, so each becomes one typed chat; --no-usda
drops them. Only the small LoRA layers train, so the run folder holds a few tens of megabytes
that predict.load() applies on top of the base model. Look at the printed loss: it should fall
steadily, and a run that ends near its starting loss learned nothing. Prove a change with --limit on
a few rows before a full run on a GPU.
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


# Tokenize one chat and mask the prompt so the loss only covers the answer
def encode(processor, image, prompt, answer):
    chat = [predict.user_turn(prompt, image and predict.open_image(image)),
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_len = processor.apply_chat_template(chat[:1], add_generation_prompt=True, **kw)["input_ids"].shape[1]
    inputs = processor.apply_chat_template(chat, **kw)
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    return inputs, labels


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=predict.DEFAULT, help="base model to fine-tune")
    p.add_argument("--data", default="data/processed/train.jsonl")
    p.add_argument("--mmfood", choices=["full", "labels", "none"], default="full")
    p.add_argument("--usda", action=argparse.BooleanOptionalAction, default=True, help="use the USDA typed meals")
    p.add_argument("--out", help="run folder; default runs/<mmfood>, with _usda added when --usda is on")
    p.add_argument("--limit", type=int, default=0, help="train on only the first N rows; 0 means all")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank; alpha is twice this")
    a = p.parse_args()
    out = a.out or f"runs/{a.mmfood}{'_usda' if a.usda else ''}"
    with open(a.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][:a.limit or None]
    examples = [c for row in rows for c in chats(row, a.mmfood, a.usda)]
    print(f"{len(rows)} rows -> {len(examples)} chats, saving to {out}")
    torch.manual_seed(0)
    model, processor = predict.load(a.model)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=TARGETS))
    model.print_trainable_parameters()
    model.train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    total = a.epochs * len(examples)
    sched = get_linear_schedule_with_warmup(opt, total // 20, total)
    os.makedirs(out, exist_ok=True)
    losses, start = [], time.time()
    for epoch in range(a.epochs):
        random.Random(epoch).shuffle(examples)
        for step, (image, prompt, answer) in enumerate(examples, 1):
            inputs, labels = encode(processor, image, prompt, answer)
            loss = model(**inputs.to(model.device), labels=labels.to(model.device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
            if step % 50 == 0 or step == len(examples):
                recent = losses[-50:]
                print(f"epoch {epoch + 1} step {step}/{len(examples)} loss {sum(recent) / len(recent):.3f} "
                      f"{(time.time() - start) / len(losses):.1f} s/step", flush=True)
        model.save_pretrained(out)
        with open(f"{out}/train.json", "w") as f:
            json.dump({"args": vars(a), "loss": losses}, f)
