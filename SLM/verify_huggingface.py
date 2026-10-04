"""Check the exported package against the original ChadGPT SFT checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import ModelCard
from safetensors import safe_open

from chadgpt import GPTModel
from chadgpt_sft import enable_position_aware_rope
from chatml_tokenizer import get_chatml_tokenizer


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/huggingface-chadgpt")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/sft_latest.pt")
    args = parser.parse_args()
    sys.path.insert(0, str(args.model_dir.resolve()))
    from inference import chat, format_messages, load_model

    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    with safe_open(args.model_dir / "model.safetensors", framework="pt", device="cpu") as exported:
        expected_keys = set(checkpoint["model"]) - {"out_head"}
        assert set(exported.keys()) == expected_keys
        for name in exported.keys():
            assert torch.equal(exported.get_tensor(name), checkpoint["model"][name]), name
    print("All exported tensors exactly match the checkpoint.")

    model, tokenizer = load_model(args.model_dir, device="cpu")
    original_tokenizer = get_chatml_tokenizer()
    assert tokenizer._mergeable_ranks == original_tokenizer._mergeable_ranks
    assert tokenizer._special_tokens == original_tokenizer._special_tokens
    assert tokenizer._pat_str == original_tokenizer._pat_str
    for text in ["Hello, world!", "नमस्ते 世界 🧠", " spaces\n\n  tabs\t", "<|im_start|>user\nHi<|im_end|>"]:
        assert tokenizer.encode(text, allowed_special="all") == original_tokenizer.encode(text, allowed_special="all")
        assert tokenizer.decode(tokenizer.encode(text, allowed_special="all")) == text
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    ids = format_messages(tokenizer, messages)
    assert ids == original_tokenizer.encode(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n",
        allowed_special="all",
    )
    print("Tokenizer ranks, regex, ChatML IDs, and Unicode round trips match.")

    enable_position_aware_rope(pi_scale=checkpoint["train_cfg"]["pi_scale"])
    with torch.device("meta"):
        reference = GPTModel(checkpoint["model_cfg"])
    reference.load_state_dict(checkpoint["model"], strict=True, assign=True)
    reference.eval()
    idx = torch.tensor([ids])
    with torch.inference_mode():
        expected, _ = reference(idx)
        actual, _ = model(idx)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        split = len(ids) // 2
        _, cache = model(idx[:, :split])
        chunk_logits, _ = model(idx[:, split:], past_key_values=cache)
        # Different GEMM/attention shapes introduce small FP32 rounding differences.
        torch.testing.assert_close(chunk_logits, actual[:, split:], atol=1e-4, rtol=1e-5)
        assert torch.equal(chunk_logits.argmax(-1), actual[:, split:].argmax(-1))
        _, cache = model(idx[:, :-1])
        token_logits, _ = model(idx[:, -1:], past_key_values=cache)
        torch.testing.assert_close(token_logits, actual[:, -1:], atol=1e-4, rtol=1e-5)
        assert torch.equal(token_logits.argmax(-1), actual[:, -1:].argmax(-1))
        _, reference_cache = reference(idx[:, :-1])
        reference_cached, _ = reference(idx[:, -1:], past_key_values=reference_cache, start_pos=len(ids) - 1)
        torch.testing.assert_close(token_logits, reference_cached, atol=1e-5, rtol=1e-5)
        edge_expected, _ = reference(idx[:, -1:], start_pos=4095)
        edge_actual, _ = model(idx[:, -1:], start_pos=4095)
        torch.testing.assert_close(edge_actual, edge_expected, atol=1e-5, rtol=1e-5)
    print("Original SFT logits, cached token/chunk decoding, and RoPE at position 4095 match.")

    info = json.loads((args.model_dir / "training_info.json").read_text())
    assert sum(p.numel() for p in model.parameters()) == info["export"]["parameter_count"]
    card = ModelCard.load(args.model_dir / "README.md")
    assert card.data.license == "mit" and card.data.pipeline_tag == "text-generation"
    for action in [
        lambda: chat(model, tokenizer, "Hi", max_new_tokens=4096),
        lambda: model(idx[:, -1:], start_pos=4096),
    ]:
        try:
            action()
        except ValueError:
            pass
        else:
            raise AssertionError("Expected rejection of context overflow.")
    print("Parameter count, model-card metadata, and context limit checks pass.")
    print("Greedy inference smoke test:", chat(model, tokenizer, "Hello!", max_new_tokens=16, temperature=0))


if __name__ == "__main__":
    main()
