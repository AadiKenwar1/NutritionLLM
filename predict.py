"""Runs the model on one photo or one typed meal and returns a parsed answer.
This is the bridge between the model and the rest of NutritionLLM: test.py calls it to score any
checkpoint, and train.py reuses the same prompts, image handling, and meal text so training and
testing match. The model path can be a Hugging Face id or a local folder. A LoRA adapter folder is
applied on top of its base model, so the stock model and every fine-tuned run are loaded and scored
the same way. Photos are shrunk to 512 px on the longest side (the model's native tile size), so
every image costs about the same number of tokens. Look at PHOTO and TEXT for the exact wording
and JSON shape the model is asked for.
"""
import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT = "LiquidAI/LFM2.5-VL-450M"
MAX_SIDE = 512
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


# Build the user turn of a chat: an optional image followed by the prompt text
def user_turn(prompt, image=None):
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    return {"role": "user", "content": content}


# Load a base model, or a LoRA adapter folder merged into its base model
def load(path=DEFAULT):
    base = path
    if os.path.exists(f"{path}/adapter_config.json"):
        with open(f"{path}/adapter_config.json") as f:
            base = json.load(f)["base_model_name_or_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 needs a GPU with compute capability 8+ (Ampere); older cards like the T4 only emulate it
    fast = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if fast else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype).to(device)
    if base != path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    return model.eval(), AutoProcessor.from_pretrained(base)


# Send one prompt (with an optional image) through the model and return its reply text
def ask(model, processor, prompt, image=None, max_new_tokens=300):
    inputs = processor.apply_chat_template([user_turn(prompt, image)], add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True, tokenize=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


# Pull the first JSON object out of a reply, or None if there isn't a valid one
def parse(reply):
    try:
        return json.JSONDecoder().raw_decode(reply, reply.index("{"))[0]
    except ValueError:
        return None


# Photo -> dict with dish, items (food, grams), and totals
def predict_photo(model, processor, image_path):
    return parse(ask(model, processor, PHOTO, open_image(image_path)))


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
