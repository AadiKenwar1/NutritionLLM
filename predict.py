"""Runs the model on one photo or one typed meal and returns a parsed answer.
This is the bridge between the model and the rest of NutritionLLM: test.py calls it to score any
checkpoint, and train.py reuses the same prompts, image handling, and meal text so training and
testing match. The model path can be a Hugging Face id or a local folder. A LoRA adapter folder is
applied on top of its base model, so the stock model and every fine-tuned run are loaded and scored
the same way. On a GPU the model runs in 16-bit numbers (bfloat16 on newer cards, float16 on older
ones like Kaggle's T4), which halves memory and about doubles speed. Photos are shrunk to 512 px on
the longest side (the model's native tile size), so every image costs about the same number of
tokens. A few SNAPMe photos are iPhone HEIC files named .jpeg, so the HEIC reader is switched on.
Look at PHOTO and TEXT for the exact wording and JSON shape the model is asked for, and at PENALTY and
NO_REPEAT, the brakes on crowded plates turning into endless food lists.
"""
import argparse
import json
import os

import torch
from PIL import Image
from pillow_heif import register_heif_opener
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import LogitsProcessor, LogitsProcessorList, NoRepeatNGramLogitsProcessor

register_heif_opener()

DEFAULT = "LiquidAI/LFM2.5-VL-450M"
MAX_SIDE = 512
# Two brakes on endless food lists from crowded plates. PENALTY makes pieces already written score
# lower: 1.2 broke the JSON on ordinary photos (quotes, prose), so it stays gentle. NO_REPEAT forbids
# repeating any run of that many tokens within the answer, which is what a looping list does (loops repeat
# 24-token runs; the JSON between two foods is 14, so keep it at 16 or more); 0 turns it off. Measured
# 2026-10-19 on 55 lab plates: 16 rescued 8 of 11 runaway answers and left the other 44 as they were
PENALTY = 1.05
NO_REPEAT = 16
PHOTO = ('Name the dish, list every food in the photo with its weight in grams, and give the meal '
         'totals. Answer with JSON only, like {"dish": "...", "items": [{"food": "...", "grams": 0}], '
         '"totals": {"kcal": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}}')
TEXT = ('Meal: {meal}\nGive the meal totals. Answer with JSON only, like '
        '{"kcal": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}')


# Fill the typed-meal prompt; replace() is used because the JSON example has braces
def text_prompt(meal):
    return TEXT.replace("{meal}", meal)


# Open a photo as RGB, shrunk so its longest side is at most MAX_SIDE pixels
def open_image(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


# Turn a number-like value into a float, or None if it can't be read
def num(x):
    try:
        return float(str(x).rstrip("g "))
    except (ValueError, TypeError):
        return None


# Turn a list of {food, grams} into text like "78.6 g wheat berry, 124.2 g carrot"
def meal_text(items):
    parts = [f"{i['grams']} g {i['food']}" if num(i.get("grams")) else str(i["food"]) for i in items]
    return ", ".join(parts)


# Ban repeating a run of n tokens inside the answer only; the prompt is exempt, so an answer may still
# copy the prompt's JSON example, like an all-zero totals line for a glass of water
class NoRepeatInAnswer(LogitsProcessor):
    def __init__(self, n, prompt_len):
        self.inner = NoRepeatNGramLogitsProcessor(n)
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        return self.inner(input_ids[:, self.prompt_len:], scores)


# Build the user turn of a chat: an optional image followed by the prompt text
def user_turn(prompt, image=None):
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    return {"role": "user", "content": content}


# Load a base model, or a LoRA adapter folder merged into its base model
def load(path=DEFAULT):
    base = path
    adapter_config = f"{path}/adapter_config.json"
    if os.path.exists(adapter_config):
        with open(adapter_config) as f:
            base = json.load(f)["base_model_name_or_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 16-bit on a GPU: bfloat16 needs an Ampere or newer card (capability 8+); older cards like the T4
    # take float16, which overflows past 65,504 and would show up as garbage answers in the quick try
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype).to(device)
    if base != path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    return model.eval(), AutoProcessor.from_pretrained(base)


# Send one prompt (with an optional image) through the model and return its reply text. The cap must
# cover the longest true answer (Nutrition5k dishes with 30+ foods run past 600 tokens) with room to
# spare: on busy plates the model lists more foods than the truth, and a cut-off reply is unreadable
def ask(model, processor, prompt, image=None, max_new_tokens=1000):
    inputs = processor.apply_chat_template([user_turn(prompt, image)], add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True, tokenize=True)
    inputs = inputs.to(model.device, dtype=model.dtype)
    prompt_len = inputs["input_ids"].shape[1]
    brakes = LogitsProcessorList([NoRepeatInAnswer(NO_REPEAT, prompt_len)] if NO_REPEAT else [])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=PENALTY,
                             logits_processor=brakes)
    return processor.decode(out[0, prompt_len:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


# Pull the first JSON object out of a reply, or None if there isn't a valid one
def parse(reply):
    try:
        return json.JSONDecoder().raw_decode(reply, reply.index("{"))[0]
    except ValueError:
        return None


# Photo -> (dict with dish, items (food, grams), and totals; the raw reply, to inspect when the dict is None)
def predict_photo(model, processor, image_path):
    reply = ask(model, processor, PHOTO, open_image(image_path))
    return parse(reply), reply


# Typed meal text -> dict with kcal, protein_g, carbs_g, fat_g
def predict_text(model, processor, meal):
    return parse(ask(model, processor, text_prompt(meal)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--image", help="photo to describe")
    p.add_argument("--text", help="typed meal to estimate")
    a = p.parse_args()
    model, processor = load(a.model)
    if a.image:
        reply = ask(model, processor, PHOTO, open_image(a.image))
        print(reply, "\n->", json.dumps(parse(reply)))
    if a.text:
        reply = ask(model, processor, text_prompt(a.text))
        print(reply, "\n->", json.dumps(parse(reply)))
