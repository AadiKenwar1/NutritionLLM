# Project: NutritionLLM

## Overview
We fine-tune one small, phone-friendly VLM (such as the Liquid LFM2-VL models), using both its vision and text abilities, so that:
- Photo → identify the dish, its foods, and their amounts.
- Photo or typed meal → calories and macros.

One model handles every task. Each task gets its own baseline and test so weak spots are easy to see.


## Code style
- One line comments above each function
- A brief paragraph at the top of every file explaining what the file builds, the role iot plays in the whole project, and what to look for.
- Keep code readable, maintanable, and iteratable

## Workflow
- Always run the code simplifier plugin after writing more then 100 lines of code

## Decisions
- MM-Food macros are estimates from another model (distillation), not measured. Nutrition5k macros are measured.
- process.py always saves the richest data. Switches live in train.py.
- train.py takes `--mmfood full|labels|none`. Each run saves to its own folder.
