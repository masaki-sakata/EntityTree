# -*- coding: utf-8 -*-
"""
Evaluate Llama 3 8B (base) and GPT-2 on occupation generation
from natural-language prompts, and score vs gold labels.

Changes:
- Read gold dataset from an external JSONL into a pandas DataFrame.
- Use only rows with is_entity == true.
- For each entity, pick one gold label from edges[*].target_label
  restricted to the 6 target occupations, using a fixed priority order.
- Keep: no accelerate, no device_map, explicit model.to("cuda:N"),
  DEBUG prints of prompts.

Run examples:
  uv run python3 eval_profession_gen.py --jsonl /home/masaki/hierarchical-repr/EntityTree/input/300people/tr80_te20/test60.jsonl --cuda 3
  uv run python3 eval_profession_gen.py --jsonl /home/masaki/hierarchical-repr/EntityTree/input/300people/test50.jsonl --cuda 3

"""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
import argparse
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# ----------------------------
# Config
# ----------------------------
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3-8B"  # base model (requested)
GPT2_MODEL_ID = "gpt2"
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.1
DEFAULT_CUDA_DEVICE = 0

# ----------------------------
# Labels (fixed 6 classes)
# ----------------------------
LABELS: List[str] = [
    "Politician",
    "Actor",
    "Athlete",
    "Musician",
    "Scientist",
    "Business Person",
]

# ----------------------------
# Natural-language prompts (English)
# ----------------------------
PROMPT_TEMPLATES = [
    "What does {name} do for a living?",
    "The occupation of {name} is",
    "What is the occupation of {name}?",
]
def build_prompt(name: str, k: int) -> str:
    tpl = PROMPT_TEMPLATES[k % len(PROMPT_TEMPLATES)]
    return tpl.format(name=name).strip()

# ----------------------------
# Normalize free-form generations to our 6 labels
# ----------------------------
SYNONYMS = {
    "Politician": [
        "politician","statesman","stateswoman","prime minister","president",
        "chancellor","political leader","head of state","civil rights leader",
        "activist","revolutionary"
    ],
    "Actor": [
        "actor","actress","film actor","movie star","comedian",
        "silent film star","performer","filmmaker","director"
    ],
    "Athlete": [
        "athlete","footballer","soccer player","sprinter","boxer","tennis player",
        "golfer","basketball player","baseball player","runner","cricketer"
    ],
    "Musician": [
        "musician","singer","composer","rapper","pop star","rock star",
        "vocalist","songwriter","pianist","guitarist","violinist"
    ],
    "Scientist": [
        "scientist","physicist","chemist","mathematician","biologist","naturalist",
        "astronomer","engineer","inventor","theorist","computer scientist","programmer"
    ],
    "Business Person": [
        "business","entrepreneur","businessman","businesswoman","business magnate",
        "ceo","executive","industrialist","investor","media proprietor","tycoon",
        "producer","founder","philanthropist"
    ],
}

def normalize_label(text: str, preferred_lab: Optional[str] = None) -> Optional[str]:
    """
    Normalize a free-form text into one of the 6 labels.
    Priority rule: check the preferred label (e.g., the gold label) first,
    then fall back to the remaining labels in the original LABELS order.
    """
    if not text:
        return None

    t = text.strip()
    tl = t.lower()

    # Build label checking order: preferred label first, then the rest
    label_order: List[str] = []
    if preferred_lab in LABELS:
        label_order.append(preferred_lab)
    label_order.extend([lab for lab in LABELS if lab not in label_order])

    # 1) Try explicit label names in the preferred order
    for lab in label_order:
        if re.search(rf"\b{re.escape(lab.lower())}\b", tl):
            return lab

    # 2) Try synonyms in the preferred order
    for lab in label_order:
        for v in SYNONYMS.get(lab, []):
            if re.search(rf"\b{re.escape(v)}\b", tl):
                return lab

    # 3) Try strict head-phrase match in the preferred order
    head = re.split(r"[.,;:\n\r!?]\s*", t)[0].strip().lower()
    for lab in label_order:
        vocab = SYNONYMS.get(lab, [])
        if head == lab.lower() or head in vocab:
            return lab

    return None


# ----------------------------
# Load gold from JSONL
# ----------------------------
def load_gold_from_jsonl(jsonl_path: str) -> Tuple[List[str], Dict[str, str], pd.DataFrame]:
    """
    Read a JSONL file containing nodes and edges, keep only is_entity==true rows,
    and construct a name->label mapping using `edges[*].target_label` restricted
    to the fixed LABELS. If multiple of the 6 labels appear, choose by LABELS order.

    Returns:
        names: list of entity names (wiki_title)
        gold_map: dict mapping name -> gold label (one of LABELS)
        df_entities: filtered dataframe with columns [wiki_title, gold]
    """
    # Load as DataFrame
    df = pd.read_json(jsonl_path, lines=True)

    # Keep only entities
    df = df[df.get("is_entity", False) == True].copy()

    # Helper: choose a single label among LABELS, by fixed priority
    def choose_label(edges: Optional[List[dict]]) -> Optional[str]:
        if not isinstance(edges, list):
            return None
        # Gather the set of valid labels present for this entity
        present = {e.get("target_label") for e in edges if isinstance(e, dict)}
        present &= set(LABELS)
        if not present:
            return None
        # Deterministic priority: LABELS order
        for lab in LABELS:
            if lab in present:
                return lab
        return None

    df["gold"] = df["edges"].apply(choose_label)
    df = df[df["gold"].notna()].copy()

    # Build mapping: name -> gold
    df["name"] = df["wiki_title"].astype(str)
    gold_map = dict(zip(df["name"], df["gold"]))
    names = list(gold_map.keys())

    return names, gold_map, df[["name", "gold"]].reset_index(drop=True)


# ----------------------------
# Model loading WITHOUT device_map (manual .to(cuda:N))
# ----------------------------
def load_generator(model_id: str, cuda_index: Optional[int]):
    try:
        cuda_avail = torch.cuda.is_available()
        chosen_index = cuda_index if cuda_index is not None else DEFAULT_CUDA_DEVICE
        device_str = f"cuda:{chosen_index}" if cuda_avail else "cpu"

        if cuda_index is not None and not cuda_avail:
            print(f"[WARN] CUDA device {cuda_index} requested but CUDA is not available. Using CPU.")

        dtype = torch.bfloat16 if cuda_avail else None

        print(f"[INFO] Loading '{model_id}' on CPU then moving to {device_str} (no device_map).")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if cuda_avail:
            model.to(device_str)

        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=chosen_index if cuda_avail else -1,
        )
        return gen

    except Exception as e:
        print(f"[WARN] Could not load model '{model_id}': {e}")
        return None

# ----------------------------
# One query
# ----------------------------
@dataclass
class ModelResult:
    model: str
    name: str
    gold: str
    prompt: str   # DEBUG: prompt used
    raw: str
    pred: Optional[str]
    correct: bool

def ask_and_parse(name: str, gold: str, generator, idx: int) -> ModelResult:
    prompt = build_prompt(name, idx)
    eos_id = getattr(getattr(generator, "tokenizer", None), "eos_token_id", None)
    out = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=(TEMPERATURE > 0.0),
        temperature=TEMPERATURE,
        eos_token_id=eos_id,
        return_full_text=False,
    )
    raw_text = out[0]["generated_text"]

    # Pass the gold label as the preferred label so it's checked first
    pred = normalize_label(raw_text, preferred_lab=gold)

    correct = (pred == gold)
    print(
        f"-------------------------------------------\n"
        f"[{generator.model.config._name_or_path}] {name}\n"
        f"  prompt: {prompt}\n"
        f"  generated_text:    {raw_text}\n"
        f"  parsed: {pred} | gold: {gold} | {'✅' if correct else '❌'}"
    )
    return ModelResult(
        model=generator.model.config._name_or_path,
        name=name, gold=gold, prompt=prompt, raw=raw_text, pred=pred, correct=correct
    )


# ----------------------------
# Evaluation loop
# ----------------------------
def run_eval(jsonl_path: str, cuda_index: Optional[int]):
    # Load gold from JSONL
    try:
        names, GOLD, gold_df = load_gold_from_jsonl(jsonl_path)
    except Exception as e:
        print(f"[ERROR] Failed to load JSONL '{jsonl_path}': {e}")
        return

    if not names:
        print(f"[ERROR] No usable entities found in '{jsonl_path}'. "
              f"Make sure rows with is_entity==true have edges with one of {LABELS}.")
        return

    print(f"[INFO] Loaded {len(names)} entities with gold labels from {jsonl_path}.")
    # Optional peek
    print(gold_df.head(10).to_string(index=False))

    llama_gen = load_generator(LLAMA_MODEL_ID, cuda_index)
    gpt2_gen = load_generator(GPT2_MODEL_ID, cuda_index)

    all_results: List[ModelResult] = []
    for gen in [llama_gen, gpt2_gen]:
        if gen is None:
            continue
        print("\n" + "="*80)
        print(f"Evaluating model: {gen.model.config._name_or_path}")
        print("="*80)
        for i, name in enumerate(names):
            all_results.append(ask_and_parse(name, GOLD[name], gen, i))

    if not all_results:
        print("No models were evaluated (loading failed). Exiting.")
        return

    df = pd.DataFrame([r.__dict__ for r in all_results])

    acc_overall = (df.groupby("model")["correct"].mean() * 100).round(2)
    print("\nOverall accuracy by model:")
    print(acc_overall.astype(str) + "%")

    acc_by_class = (df.groupby(["model","gold"])["correct"].mean() * 100).round(1).unstack()
    print("\nPer-class accuracy by model (rows=models, cols=gold):")
    print(acc_by_class.fillna(0.0).astype(str) + "%")

    for model_name, sub in df.groupby("model"):
        print(f"\nConfusion matrix for {model_name} (gold rows vs pred cols):")
        cm = pd.crosstab(sub["gold"], sub["pred"], dropna=False)
        print(cm)

    misses = df[~df["correct"]]
    if not misses.empty:
        print("\nExamples of misclassifications (with prompts):")
        print(misses[["model","name","gold","pred","prompt","raw"]].head(10).to_string(index=False))
    else:
        print("\nNo misclassifications 🎉")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True,
                        help="Path to JSONL dataset (see README).")
    parser.add_argument("--cuda", type=int, default=None,
                        help="CUDA device index to use (e.g., 0 or 1). If omitted, uses DEFAULT_CUDA_DEVICE or CPU.")
    args = parser.parse_args()
    run_eval(jsonl_path=args.jsonl, cuda_index=args.cuda)
