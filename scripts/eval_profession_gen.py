# -*- coding: utf-8 -*-
"""
Evaluate Llama 3 8B (base) and GPT-2 on occupation generation
from natural-language prompts, and score vs gold labels.
- No accelerate required
- No device_map usage
- Explicit model.to("cuda:N") placement
- DEBUG: print prompt used for each prediction

Run examples:
  python eval_profession_gen.py --cuda 0
  python eval_profession_gen.py --cuda 1

Requirements:
  pip install transformers torch pandas
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional
import pandas as pd
import torch
import argparse

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
TEMPERATURE = 0.5
DEFAULT_CUDA_DEVICE = 0

# ----------------------------
# Dataset (gold)
# ----------------------------
DATA = {
    "Politician": [
        "Barack Obama","Margaret Thatcher","Winston Churchill",
        "Nelson Mandela","Franklin D. Roosevelt","Abraham Lincoln",
        "John F. Kennedy","Angela Merkel","Mahatma Gandhi"
    ],
    "Actor": [
        "Meryl Streep","Denzel Washington","Tom Hanks",
        "Leonardo DiCaprio","Marilyn Monroe","Robert De Niro",
        "Audrey Hepburn","Morgan Freeman","Charlie Chaplin"
    ],
    "Athlete": [
        "Michael Jordan","Lionel Messi","Usain Bolt",
        "Serena Williams","Muhammad Ali","Tiger Woods",
        "Cristiano Ronaldo","Babe Ruth"
    ],
    "Musician": [
        "John Lennon","Michael Jackson","Madonna","Bob Dylan",
        "Elvis Presley","Aretha Franklin","Mozart","Beyoncé"
    ],
    "Scientist": [
        "Albert Einstein","Marie Curie","Isaac Newton","Charles Darwin",
        "Nikola Tesla","Stephen Hawking","Galileo Galilei","Ada Lovelace"
    ],
    "Business Person": [
        "Steve Jobs","Bill Gates","Warren Buffett","Elon Musk",
        "Jeff Bezos","Henry Ford","Walt Disney","Oprah Winfrey"
    ],
}
LABELS = list(DATA.keys())
GOLD: Dict[str, str] = {n: lab for lab, names in DATA.items() for n in names}

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

def normalize_label(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    tl = t.lower()

    for lab in LABELS:
        if lab.lower() in tl:
            return lab
    for lab, vocab in SYNONYMS.items():
        for v in vocab:
            if re.search(rf"\b{re.escape(v)}\b", tl):
                return lab
    head = re.split(r"[.,;:\n\r]\s*", t)[0].strip().lower()
    for lab, vocab in SYNONYMS.items():
        if any(head == v for v in vocab):
            return lab
    return None

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
    prompt: str   # <-- added
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
    pred = normalize_label(raw_text)
    correct = (pred == gold)
    # ---- DEBUG print with prompt included ----
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
def run_eval(cuda_index: Optional[int]):
    names = list(GOLD.keys())

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
    parser.add_argument("--cuda", type=int, default=None,
                        help="CUDA device index to use (e.g., 0 or 1). If omitted, uses DEFAULT_CUDA_DEVICE or CPU.")
    args = parser.parse_args()
    run_eval(cuda_index=args.cuda)
