
import re
import random
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from chatml_tokenizer import get_chatml_tokenizer, IM_START_ID, IM_END_ID, PAD_ID

random.seed(42)
np.random.seed(42)


CTX_LEN = 4096
OUT_DIR = "/kaggle/working"
SHARD_PREFIX = "sft_train_shard"
EXAMPLES_PER_SHARD = 2000
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


'''
stratified sampling targets — we want ~80k examples spread across 7 categories 
so the model sees a balanced mix of coding, STEM, writing, reasoning etc 
not just whatever category dominates the dataset
'''
TARGET_COUNTS = {
    "coding": 21333,
    "stem": 16000,
    "writing": 13333,
    "reasoning": 8000,
    "mathematics": 8000,
    "roleplay": 5333,
    "multilingual": 8000,
}

'''
source hints help classify examples by matching keywords in the source field 
when the content-based regex patterns below don't match anything
'''
SOURCE_HINTS = {
    "coding": ["glaive-code-assist", "code_bagel", "wizardcoder", "leetcode"],
    "mathematics": ["camelai_math", "mathinstruct", "orca_math", "metamath"],
    "stem": ["camelai_physics", "camelai_chemistry", "camelai_biology", "airoboros"],
    "roleplay": ["airoboros_roleplay", "pippa", "roleplay"],
    "reasoning": ["cot", "orca", "platypus"],
    "writing": ["gpt4-llm", "gpteacher"],
}

'''
regex patterns used to classify each example into a category based on the content 
of the human turns — we check these in order: code → math → roleplay → reasoning → writing
if none match and the text isn't english it goes to multilingual, otherwise fallback to stem
'''
CODE_PATTERNS = re.compile(
    r"```|def \w+\(|class \w+[:\(]|import \w+|function\s*\(|SELECT .* FROM|"
    r"<html|#include|public static void|console\.log|npm install|pip install",
    re.IGNORECASE,
)
MATH_PATTERNS = re.compile(
    r"\\frac|\\int|\\sum|\\sqrt|solve for [a-z]|derivative of|integral of|"
    r"prove that|theorem|\bfactorize\b|quadratic equation",
    re.IGNORECASE,
)
ROLEPLAY_PATTERNS = re.compile(
    r"pretend (you|to be)|role-?play|act as (a|an)|you are a character|"
    r"stay in character|write (his|her|their) dialogue",
    re.IGNORECASE,
)
REASONING_PATTERNS = re.compile(
    r"step by step|think through|logic puzzle|riddle|deduce|"
    r"if .* then .* (else|otherwise)|explain your reasoning",
    re.IGNORECASE,
)
WRITING_PATTERNS = re.compile(
    r"write a (story|poem|essay|article|blog|letter|speech)|compose a|"
    r"draft a|write in the style of",
    re.IGNORECASE,
)


def get_human_text(example):
    turns = example.get("conversations", [])
    return " ".join(t.get("value", "") for t in turns if t.get("from") == "human")


def is_non_english(text):
    try:
        from langdetect import detect
        return detect(text) != "en"
    except Exception:
        return False


def classify(example):
    text = get_human_text(example)
    source = (example.get("source") or "").lower()

    if CODE_PATTERNS.search(text):
        return "coding"
    if MATH_PATTERNS.search(text):
        return "mathematics"
    if ROLEPLAY_PATTERNS.search(text):
        return "roleplay"
    if REASONING_PATTERNS.search(text):
        return "reasoning"
    if WRITING_PATTERNS.search(text):
        return "writing"
    if text.strip() and is_non_english(text[:300]):
        return "multilingual"
    for cat, hints in SOURCE_HINTS.items():
        if any(h in source for h in hints):
            return cat
    return "stem"


'''
goes through every example, classifies it into a category bucket, 
then takes up to TARGET_COUNTS[category] from each bucket 
if any category is short, fills the gap from a general pool of leftover examples
'''
def stratified_sample(ds):
    print("Classifying examples for stratified sampling...")
    buckets = defaultdict(list)
    for idx, ex in enumerate(ds):
        buckets[classify(ex)].append(idx)
        if idx % 100000 == 0 and idx > 0:
            print(f"  scanned {idx} -> {({k: len(v) for k, v in buckets.items()})}")

    sampled_indices = []
    shortfall = 0
    for cat, target in TARGET_COUNTS.items():
        available = buckets[cat][:]
        random.shuffle(available)
        take = min(target, len(available))
        sampled_indices.extend(available[:take])
        if take < target:
            shortfall += target - take
            print(f"  WARNING: '{cat}' short by {target - take}")

    if shortfall > 0:
        used = set(sampled_indices)
        remaining_pool = [i for idxs in buckets.values() for i in idxs if i not in used]
        random.shuffle(remaining_pool)
        sampled_indices.extend(remaining_pool[:shortfall])
        print(f"  Redistributed {min(shortfall, len(remaining_pool))} from general pool")

    random.shuffle(sampled_indices)
    print(f"Final sample size: {len(sampled_indices)}")
    return sampled_indices


'''
takes one conversation and formats it in ChatML template then tokenizes it 
returns (token_ids, loss_mask) where loss_mask[i] = 1 means that token is part of an 
assistant turn and should count toward the loss — system and user turns are masked out (0)
the assistant's trailing <|im_end|> is also masked in so the model learns when to stop
'''
def build_chatml_tokens(example, tokenizer):
    turns = example.get("conversations", [])
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    has_system = any(t.get("from") == "system" for t in turns)

    token_ids, loss_mask = [], []
    allowed = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}

    def add_turn(role, content, mask_value):
        header = f"<|im_start|>{role}\n"
        footer = "<|im_end|>\n"
        header_ids = tokenizer.encode(header, allowed_special=allowed)
        content_ids = tokenizer.encode(content, allowed_special=allowed)
        footer_ids = tokenizer.encode(footer, allowed_special=allowed)

        token_ids.extend(header_ids)
        loss_mask.extend([0] * len(header_ids))
        token_ids.extend(content_ids)
        loss_mask.extend([mask_value] * len(content_ids))
        token_ids.extend(footer_ids)
        loss_mask.extend([mask_value] * len(footer_ids))

    if not has_system:
        add_turn("system", DEFAULT_SYSTEM_PROMPT, mask_value=0)

    for t in turns:
        role = role_map.get(t.get("from"))
        if role is None:
            continue
        mask_value = 1 if role == "assistant" else 0
        add_turn(role, t.get("value", ""), mask_value)

    return token_ids, loss_mask


'''
greedy packing — stuffs multiple conversations into rows of CTX_LEN + 1 tokens each 
the +1 is so the training script can slice x = tokens[:-1], y = tokens[1:] and get full 
CTX_LEN-length input/target pairs without losing the last token 
this matches ShardDataset's convention in chadgpt.py exactly

position_ids reset to 0 at each packed document boundary so RoPE sees each conversation 
as starting fresh instead of continuing from the previous one's position count
padding gets position 0 too but it's always loss_mask=0 so it's never trained on
'''
def pack_sequences(all_token_ids, all_loss_masks, ctx_len, pad_id):
    pack_len = ctx_len + 1
    packed_tokens, packed_masks, packed_positions = [], [], []
    buf_tokens, buf_masks, buf_positions = [], [], []

    def flush():
        nonlocal buf_tokens, buf_masks, buf_positions
        if not buf_tokens:
            return
        pad_needed = pack_len - len(buf_tokens)
        packed_tokens.append(buf_tokens + [pad_id] * pad_needed)
        packed_masks.append(buf_masks + [0] * pad_needed)
        packed_positions.append(buf_positions + [0] * pad_needed)
        buf_tokens, buf_masks, buf_positions = [], [], []

    for tok_ids, loss_mask in zip(all_token_ids, all_loss_masks):
        if len(tok_ids) > pack_len:
            tok_ids = tok_ids[:pack_len]
            loss_mask = loss_mask[:pack_len]
        if not tok_ids:
            continue

        if len(buf_tokens) + len(tok_ids) > pack_len:
            flush()

        buf_tokens.extend(tok_ids)
        buf_masks.extend(loss_mask)
        buf_positions.extend(range(len(tok_ids)))

    flush()

    return (
        np.array(packed_tokens, dtype=np.uint16),
        np.array(packed_masks, dtype=np.int8),
        np.array(packed_positions, dtype=np.int32),
    )


'''
main pipeline:
  1. load OpenHermes-2.5 and stratified-sample ~80k examples
  2. tokenize each conversation with ChatML formatting and build per-token loss masks
  3. greedily pack into CTX_LEN+1 token rows with position resets at document boundaries
  4. save as .npy shards (tokens / loss_mask / positions)
'''
def main():
    print("Loading OpenHermes-2.5...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    print(f"Loaded {len(ds)} examples")

    sampled_indices = stratified_sample(ds)
    subset = ds.select(sampled_indices)

    tokenizer = get_chatml_tokenizer()
    print(f"Tokenizer ready. <|im_start|>={IM_START_ID} <|im_end|>={IM_END_ID} pad={PAD_ID}")

    print("Tokenizing conversations with ChatML formatting...")
    all_token_ids, all_loss_masks = [], []
    skipped = 0
    for i, ex in enumerate(subset):
        tok_ids, loss_mask = build_chatml_tokens(ex, tokenizer)
        if not any(loss_mask):
            skipped += 1
            continue
        all_token_ids.append(tok_ids)
        all_loss_masks.append(loss_mask)
        if i % 20000 == 0 and i > 0:
            print(f"  tokenized {i}/{len(subset)}")

    print(f"Tokenized {len(all_token_ids)} conversations, skipped {skipped} with no assistant content")
    total_tokens = sum(len(t) for t in all_token_ids)
    print(f"Total raw tokens before packing: {total_tokens:,}")

    print(f"Packing into {CTX_LEN}+1 = {CTX_LEN + 1}-token rows...")
    packed_tokens, packed_masks, packed_positions = pack_sequences(
        all_token_ids, all_loss_masks, CTX_LEN, PAD_ID
    )
    print(f"Packed into {len(packed_tokens)} rows")

    utilization = 1.0 - (packed_tokens == PAD_ID).sum() / packed_tokens.size
    print(f"Packing utilization (non-pad fraction): {utilization:.2%}")

    print(f"Saving shards ({EXAMPLES_PER_SHARD} rows per shard)...")
    num_shards = (len(packed_tokens) + EXAMPLES_PER_SHARD - 1) // EXAMPLES_PER_SHARD
    for shard_idx in range(num_shards):
        start = shard_idx * EXAMPLES_PER_SHARD
        end = min(start + EXAMPLES_PER_SHARD, len(packed_tokens))
        np.save(f"{OUT_DIR}/{SHARD_PREFIX}_tokens_{shard_idx}.npy", packed_tokens[start:end])
        np.save(f"{OUT_DIR}/{SHARD_PREFIX}_lossmask_{shard_idx}.npy", packed_masks[start:end])
        np.save(f"{OUT_DIR}/{SHARD_PREFIX}_positions_{shard_idx}.npy", packed_positions[start:end])
        print(f"  saved shard {shard_idx} ({end - start} rows)")

    print(f"\nDone. {num_shards} shards saved to {OUT_DIR}, prefix '{SHARD_PREFIX}'")
    print(f"New vocab size needed in chadgpt_sft.py: 50259 (base 50257 + 2 ChatML tokens)")


if __name__ == "__main__":
    main()