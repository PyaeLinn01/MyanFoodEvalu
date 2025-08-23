#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#python3 prepare_json.py -i "Coconut noodle VQA.json" -o "Coconut noodle VQA_text.json"

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any

# Generic phrases commonly used in questions that refer to the dish without naming it
GENERIC_TARGETS = [
    "ပုံတွင်ပြထားသော အစားအစာ",
    "ပုံထဲက အစားအစာ",
    "ဤအစားအစာ",
]

def transform_items(items: List[Dict[str, Any]], fallback_food_name: str = None) -> List[Dict[str, Any]]:
    """
    - Replace generic dish phrases in 'question' with the specific Myanmar food name from food_meta.food_name_my
    - Drop 'food_meta'
    """
    out: List[Dict[str, Any]] = []

    for it in items:
        it = dict(it)  # shallow copy

        # Determine the food name to replace (from food_meta if present)
        food_name_my = None
        if "food_meta" in it and isinstance(it["food_meta"], dict):
            food_name_my = it["food_meta"].get("food_name_my")

        # Replace only in 'question'
        q = it.get("question")
        if isinstance(q, str):
            # Use food_name_my if available, otherwise fallback (if provided)
            name_to_use = food_name_my or fallback_food_name
            if name_to_use:
                for tgt in GENERIC_TARGETS:
                    q = q.replace(tgt, name_to_use)
                it["question"] = q

        # Drop food_meta entirely after using it
        it.pop("food_meta", None)

        out.append(it)

    return out

def main():
    parser = argparse.ArgumentParser(description="Convert VQA JSON to text-only questions and drop food meta.")
    parser.add_argument(
        "-i", "--input",
        default="Coconut noodle VQA.json",
        help="Path to the input JSON file (default: Coconut noodle VQA.json)"
    )
    parser.add_argument(
        "-o", "--output",
        default="Coconut noodle VQA_text.json",
        help="Path to the output JSON file (default: Coconut noodle VQA_text.json)"
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of question items.")

    transformed = transform_items(data)

    # Optionally, ensure answers stay as lists of ints (keep original)
    # and no 'food_meta' remains.

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=4)

    print(f"Written text-only VQA JSON to: {out_path}")

if __name__ == "__main__":
    main()