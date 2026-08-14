
import os
import numpy as np
import tiktoken
from datasets import load_dataset, interleave_datasets


'''
dataset download and shard creation for ChadGPT pretraining

downloads cosmopedia-v2 from HuggingFace in streaming mode, tokenizes with GPT-2 BPE,
and saves as 50 shards of 100M tokens each (40 train + 10 val)
naming matches what ShardDataset in chadgpt.py expects: slm_train_shard_0.npy through slm_train_shard_49.npy

run this on a CPU session before starting training:
    python dataset_creation.py
'''

SHARD_DIR        = "/kaggle/working"
SHARD_SIZE       = 100_000_000      # 100M tokens per shard
N_TRAIN_SHARDS   = 40               # shards 0–39  → 4B tokens train
N_VAL_SHARDS     = 10               # shards 40–49 → 1B tokens val
EOS_TOKEN        = 50256            # GPT-2 <|endoftext|>

'''
dataset mix config — each entry gets streamed and interleaved by weight
text_field is the column name containing the document text 
right now it's just cosmopedia-v2 at 100% weight but you can add more sources here
'''
DATASETS = [
    {
        "name":       "HuggingFaceTB/smollm-corpus",
        "config":     "cosmopedia-v2",
        "split":      "train",
        "text_field": "text",
        "weight":     1.0,
    },
]

os.makedirs(SHARD_DIR, exist_ok=True)
tokenizer = tiktoken.get_encoding("gpt2")


def tokenize(text: str) -> list[int]:
    tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    tokens.append(EOS_TOKEN)
    return tokens


def save_shard(tokens: list[int], shard_index: int):
    path = os.path.join(SHARD_DIR, f"slm_train_shard_{shard_index}.npy")
    arr  = np.array(tokens[:SHARD_SIZE], dtype=np.uint16)
    np.save(path, arr)
    kind = "TRAIN" if shard_index < N_TRAIN_SHARDS else "VAL"
    print(f"  [{kind}] saved slm_train_shard_{shard_index}.npy  ({len(arr):,} tokens)", flush=True)


print("Loading datasets (streaming)...")
raw_datasets = []
weights      = []

for ds_cfg in DATASETS:
    ds = load_dataset(
        ds_cfg["name"],
        ds_cfg["config"],
        split=ds_cfg["split"],
        streaming=True,
        trust_remote_code=True,
    )
    if ds_cfg["text_field"] != "text":
        ds = ds.rename_column(ds_cfg["text_field"], "text")
    ds = ds.select_columns(["text"])

    raw_datasets.append(ds)
    weights.append(ds_cfg["weight"])
    print(f"  loaded: {ds_cfg['name']} / {ds_cfg['config']}  (text_field='{ds_cfg['text_field']}')")

mixed = interleave_datasets(
    raw_datasets,
    probabilities=weights,
    seed=42,
    stopping_strategy="first_exhausted",
).shuffle(buffer_size=10_000, seed=42)


'''
tokenize the entire stream into a buffer and flush to disk every time 
the buffer hits SHARD_SIZE (100M tokens) — each shard becomes one .npy file
'''
total_shards = N_TRAIN_SHARDS + N_VAL_SHARDS
print(f"\nTokenizing → {total_shards} shards × {SHARD_SIZE:,} tokens")
print(f"  Train: shards 0–{N_TRAIN_SHARDS-1}  ({SHARD_SIZE*N_TRAIN_SHARDS/1e9:.1f}B tokens)")
print(f"  Val:   shards {N_TRAIN_SHARDS}–{total_shards-1}  ({SHARD_SIZE*N_VAL_SHARDS/1e9:.1f}B tokens)\n")

buffer      = []
shard_index = 0

for doc in mixed:
    text = doc["text"].strip()
    if not text:
        continue

    buffer.extend(tokenize(text))

    while len(buffer) >= SHARD_SIZE:
        if shard_index >= total_shards:
            break
        save_shard(buffer, shard_index)
        buffer      = buffer[SHARD_SIZE:]
        shard_index += 1

    if shard_index >= total_shards:
        break

# save leftover as partial final shard
if buffer and shard_index < total_shards:
    save_shard(buffer, shard_index)
    shard_index += 1

print(f"\nDone.")
print(f"  Total shards saved : {shard_index}")
print(f"  Files in {SHARD_DIR}:")
for f in sorted(os.listdir(SHARD_DIR)):
    if f.endswith(".npy"):
        size_mb = os.path.getsize(os.path.join(SHARD_DIR, f)) / 1e6
        print(f"    {f}  ({size_mb:.0f} MB)")