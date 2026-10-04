"""Export the final ChadGPT SFT checkpoint as a standalone Hugging Face package.

Run from the project root:
    .venv/bin/python SLM/export_huggingface.py
    hf upload sam-eer12/chadGPT models/huggingface-chadgpt .
"""

import argparse
import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file

from chatml_tokenizer import get_chatml_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models/sft_latest.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/huggingface-chadgpt")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("phase") != "sft":
        raise ValueError("This exporter expects a ChatML SFT checkpoint.")
    source = checkpoint["model"]
    if not torch.equal(source["out_head"], source["tok_emb.weight"]):
        raise ValueError("Output and input embeddings must be tied.")
    weights = {name: tensor.contiguous() for name, tensor in source.items() if name != "out_head"}
    if any(tensor.dtype != torch.float32 for tensor in weights.values()):
        raise ValueError("Expected the original float32 SFT weights.")
    cfg = dict(checkpoint["model_cfg"])
    train_cfg = checkpoint["train_cfg"]
    cfg.update({
        "architectures": ["GPTModel"],
        "model_type": "chadgpt",
        "torch_dtype": "float32",
        "rope_base": 10000,
        "pi_scale": train_cfg["pi_scale"] if train_cfg["use_position_interpolation"] else 1.0,
        "tie_word_embeddings": True,
        "bos_token_id": 50257,
        "eos_token_id": 50258,
        "pad_token_id": 50256,
    })
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    package = PROJECT_ROOT / "SLM/huggingface"
    for name in ["README.md", "LICENSE", "modeling_chadgpt.py", "inference.py", "requirements.txt"]:
        shutil.copy2(package / name, out / name)
    shutil.copy2(PROJECT_ROOT / "SLM/evaluate_arc_easy.py", out / "evaluate_arc_easy.py")
    # The exported evaluator sits beside the standalone inference dependencies.
    eval_requirements = (PROJECT_ROOT / "SLM/requirements-eval.txt").read_text()
    (out / "requirements-eval.txt").write_text(
        eval_requirements.replace("-r huggingface/requirements.txt", "-r requirements.txt")
    )
    shutil.copy2(PROJECT_ROOT / "SLM/chadgpt.ipynb", out / "chadgpt.ipynb")
    shutil.copy2(PROJECT_ROOT / "SLM/chadgpt_playground.ipynb", out / "chadgpt_playground.ipynb")
    # The Hub uses this exact filename for its Colab and Kaggle launch routes.
    shutil.copy2(PROJECT_ROOT / "SLM/chadgpt_playground.ipynb", out / "notebook.ipynb")
    playground = out / "playground"
    playground.mkdir(exist_ok=True)
    for name in ["app.py", "README.md", "requirements.txt", "requirements-notebook.txt"]:
        shutil.copy2(PROJECT_ROOT / "SLM/huggingface_space" / name, playground / name)
    shutil.copy2(package / "LICENSE", playground / "LICENSE")
    write_json(out / "config.json", cfg)
    save_file(weights, str(out / "model.safetensors"), metadata={"format": "pt"})

    tokenizer = get_chatml_tokenizer()
    if tokenizer.n_vocab != cfg["vocab_size"]:
        raise ValueError("Tokenizer vocabulary must match model embeddings.")
    with (out / "tokenizer.tiktoken").open("wb") as stream:
        for token, rank in sorted(tokenizer._mergeable_ranks.items(), key=lambda item: item[1]):
            stream.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")
    write_json(out / "tokenizer_config.json", {
        "name": tokenizer.name,
        "implementation": "tiktoken.Encoding",
        "pat_str": tokenizer._pat_str,
        "special_tokens": tokenizer._special_tokens,
        "model_max_length": cfg["context_length"],
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "chat_format": "ChatML",
    })

    training = out / "training"
    training.mkdir(exist_ok=True)
    scripts = ["chadgpt.py", "chadgpt_finetune.py", "chadgpt_sft.py", "chatml_tokenizer.py",
               "dataset_creation.py", "prepare_openhermes_sft.py"]
    for name in scripts:
        shutil.copy2(PROJECT_ROOT / "SLM" / name, training / name)
    parameter_count = sum(tensor.numel() for name, tensor in weights.items() if not name.endswith(".theta"))
    info = {key: checkpoint[key] for key in [
        "phase", "step", "base_tokens_seen", "sft_tokens_seen", "tokens_seen", "model_cfg", "train_cfg"
    ]}
    info["export"] = {
        "source_checkpoint": args.checkpoint.name,
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "project_url": "https://github.com/sam-eer12/projectSLM",
        "source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "training_source_sha256": {name: sha256(training / name) for name in scripts},
        "parameter_count": parameter_count,
        "weight_dtype": "float32",
        "weight_sha256": sha256(out / "model.safetensors"),
        "weight_bytes": (out / "model.safetensors").stat().st_size,
        "omitted_state_dict_aliases": {"out_head": "tok_emb.weight"},
        "optimizer_included": False,
        "precision_conversion": False,
    }
    write_json(out / "training_info.json", info)
    evaluation = PROJECT_ROOT / "evaluations/arc_easy"
    if (evaluation / "results.json").exists():
        result = json.loads((evaluation / "results.json").read_text())
        provenance = result["provenance"]
        if provenance["weight_sha256"] != info["export"]["weight_sha256"]:
            raise ValueError("ARC-Easy results belong to different weights; do not publish them with this export.")
        if not provenance["full_test_split"]:
            raise ValueError("The model card requires a full ARC-Easy test-split evaluation.")
        shutil.copytree(evaluation, out / "evaluation/arc_easy", dirs_exist_ok=True)
    print(f"Exported {parameter_count:,} parameters to {out}")
    print(f"Weights: {info['export']['weight_bytes']:,} bytes (float32, no precision conversion)")


if __name__ == "__main__":
    main()
