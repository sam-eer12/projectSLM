"""Run the custom ChadGPT model from a downloaded Hub snapshot."""

import argparse
import base64
import json
from pathlib import Path

import tiktoken
import torch
from safetensors.torch import load_file

from modeling_chadgpt import GPTModel


def load_tokenizer(model_dir):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
    ranks = {}
    for line in (model_dir / "tokenizer.tiktoken").read_bytes().splitlines():
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return tiktoken.Encoding(
        name=cfg["name"], pat_str=cfg["pat_str"],
        mergeable_ranks=ranks, special_tokens=cfg["special_tokens"],
    )


def load_model(model_dir, device="auto"):
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    # Meta initialization avoids allocating a second copy of the model weights.
    with torch.device("meta"):
        model = GPTModel(cfg)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True, assign=True)
    return model.to(device).eval(), load_tokenizer(model_dir)


def format_messages(tokenizer, messages, add_generation_prompt=True):
    ids = []
    special = tokenizer._special_tokens
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {role}")
        ids.append(special["<|im_start|>"])
        ids.extend(tokenizer.encode_ordinary(role + "\n"))
        ids.extend(tokenizer.encode_ordinary(message["content"]))
        ids.append(special["<|im_end|>"])
        ids.extend(tokenizer.encode_ordinary("\n"))
    if add_generation_prompt:
        ids.append(special["<|im_start|>"])
        ids.extend(tokenizer.encode_ordinary("assistant\n"))
    return ids


@torch.inference_mode()
def chat(model, tokenizer, prompt=None, *, messages=None, max_new_tokens=256,
         temperature=0.7, top_k=40):
    if (prompt is None) == (messages is None):
        raise ValueError("Provide exactly one of prompt or messages.")
    if max_new_tokens < 1 or temperature < 0 or top_k < 0:
        raise ValueError("Use max_new_tokens >= 1, temperature >= 0, and top_k >= 0.")
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    ids = format_messages(tokenizer, messages)
    if len(ids) + max_new_tokens > model.config["context_length"]:
        raise ValueError("Prompt plus max_new_tokens must fit within 4096 tokens.")
    device = next(model.parameters()).device
    current = torch.tensor([ids], device=device, dtype=torch.long)
    past = None
    output = []
    for _ in range(max_new_tokens):
        logits, past = model(current, past_key_values=past)
        logits = logits[:, -1, :]
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            next_id = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
        token = next_id.item()
        if token in {tokenizer._special_tokens["<|im_end|>"], tokenizer.eot_token}:
            break
        output.append(token)
        current = next_id
    return tokenizer.decode(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--prompt", default="Write a Python function that checks if a string is a palindrome.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7, help="Use 0 for greedy decoding.")
    parser.add_argument("--top-k", type=int, default=40, help="Use 0 to disable top-k filtering.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model_dir, args.device)
    print(chat(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
               temperature=args.temperature, top_k=args.top_k))


if __name__ == "__main__":
    main()
