"""Evaluate the released ChadGPT weights with EleutherAI's ARC-Easy task.

Install SLM/requirements-eval.txt in a project checkout, or requirements-eval.txt
in a downloaded Hugging Face model folder, then run this script.
The default is the entire test split, zero-shot, FP32, without a chat template.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_LAYOUT = (SCRIPT_DIR / "huggingface/inference.py").is_file()
ROOT = SCRIPT_DIR.parent if PROJECT_LAYOUT else SCRIPT_DIR
INFERENCE_DIR = SCRIPT_DIR / "huggingface" if PROJECT_LAYOUT else SCRIPT_DIR
DEFAULT_MODEL_DIR = ROOT / "models/huggingface-chadgpt" if PROJECT_LAYOUT else ROOT
DEFAULT_OUTPUT_DIR = ROOT / ("evaluations/arc_easy" if PROJECT_LAYOUT else "evaluation_runs/arc_easy")
# Keep downloaded data and locks inside the workspace.
os.environ.setdefault("HF_HOME", str(ROOT / "models/eval-cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "models/eval-cache/datasets"))

import lm_eval
from lm_eval.api.model import TemplateLM
from lm_eval.api.task import ConfigurableTask
from huggingface_hub import HfApi
import numpy as np
import torch
import yaml

sys.path.insert(0, str(INFERENCE_DIR))
from inference import load_model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class ChadGPTLM(TemplateLM):
    """Likelihood-only adapter using the harness's own pair tokenization.

    Each question is prefetched once; answer continuations share its KV cache.
    Right padding is causal and excluded from scoring. A full-forward reference
    checks token alignment, cache positions, and batching on real task requests.
    """

    def __init__(self, model_dir, device, validate_questions=8):
        super().__init__()
        self.model, self.tokenizer = load_model(model_dir, device)
        self._device = next(self.model.parameters()).device
        self.validate_questions = validate_questions
        self.validation_errors = []
        self.max_input_tokens = 0

    @property
    def eot_token_id(self):
        return self.tokenizer.eot_token

    def tok_encode(self, string, add_special_tokens=None, **kwargs):
        return self.tokenizer.encode_ordinary(string)

    @torch.inference_mode()
    def score_answers(self, context, answers):
        max_answer = max(map(len, answers))
        self.max_input_tokens = max(self.max_input_tokens, len(context) + max_answer)
        if not context or not all(answers):
            raise ValueError("Context and answer token sequences must be nonempty.")
        if len(context) + max_answer > self.model.config["context_length"]:
            raise ValueError("Evaluation input exceeds the model context; no truncation is allowed.")
        logits, past = self.model(torch.tensor([context], device=self.device))
        first = logits[0, -1].log_softmax(-1)
        first_tokens = torch.tensor([answer[0] for answer in answers], device=self.device)
        scores = first[first_tokens]
        greedy = first.argmax() == first_tokens
        del logits, first
        if max_answer > 1:
            batch = len(answers)
            inputs = torch.full((batch, max_answer - 1), self.eot_token_id,
                                dtype=torch.long, device=self.device)
            targets = inputs.clone()
            valid = torch.zeros_like(inputs, dtype=torch.bool)
            for i, answer in enumerate(answers):
                count = len(answer) - 1
                inputs[i, :count] = torch.tensor(answer[:-1], device=self.device)
                targets[i, :count] = torch.tensor(answer[1:], device=self.device)
                valid[i, :count] = True
            expanded = [(k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1))
                        for k, v in past]
            logits, _ = self.model(inputs, past_key_values=expanded)
            token_scores = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            scores = scores + token_scores.masked_fill(~valid, 0).sum(-1)
            greedy = greedy & ((logits.argmax(-1) == targets) | ~valid).all(-1)
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite answer likelihoods.")
        return list(zip(scores.cpu().tolist(), greedy.cpu().tolist()))

    @torch.inference_mode()
    def reference_score(self, context, answer):
        inputs = torch.tensor([(context + answer)[:-1]], device=self.device)
        logits, _ = self.model(inputs)
        selected = logits[0, len(context) - 1:len(context) - 1 + len(answer)]
        targets = torch.tensor(answer, device=self.device)
        score = selected.log_softmax(-1).gather(-1, targets[:, None]).sum().item()
        greedy = bool((selected.argmax(-1) == targets).all().item())
        return score, greedy

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        groups = defaultdict(list)
        for index, (key, context, continuation) in enumerate(requests):
            groups[tuple(context)].append((index, key, continuation))
        results = [None] * len(requests)
        started = last_update = time.monotonic()
        for number, (context, group) in enumerate(groups.items(), 1):
            answers = [item[2] for item in group]
            scored = self.score_answers(list(context), answers)
            if number <= self.validate_questions:
                reference = [self.reference_score(list(context), answer) for answer in answers]
                actual_scores = np.array([x[0] for x in scored])
                reference_scores = np.array([x[0] for x in reference])
                np.testing.assert_allclose(actual_scores, reference_scores, atol=5e-4, rtol=1e-5)
                assert [x[1] for x in scored] == [x[1] for x in reference]
                lengths = np.array([len(item[1][1].removeprefix(" ")) for item in group])
                assert actual_scores.argmax() == reference_scores.argmax()
                assert (actual_scores / lengths).argmax() == (reference_scores / lengths).argmax()
                self.validation_errors.append(float(np.abs(actual_scores - reference_scores).max()))
            for (index, key, _), result in zip(group, scored, strict=True):
                results[index] = result
                self.cache_hook.add_partial("loglikelihood", key, result)
            now = time.monotonic()
            if number % 100 == 0 or now - last_update >= 30 or number == len(groups):
                elapsed = now - started
                remaining = elapsed / number * (len(groups) - number)
                print(f"ARC-Easy: {number}/{len(groups)} questions; "
                      f"{elapsed:.0f}s elapsed; ~{remaining:.0f}s remaining", flush=True)
                last_update = now
        return results

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError("This adapter implements multiple-choice likelihood evaluation only.")

    def generate_until(self, requests, **kwargs):
        raise NotImplementedError("ARC-Easy scores answer likelihoods; it does not generate answers.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, help="Smoke test only; omit for the model-card result.")
    parser.add_argument("--dataset-revision", help="Hub commit; defaults to resolving the current commit.")
    parser.add_argument("--validate-questions", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.validate_questions < 1 or args.cpu_threads < 1:
        parser.error("Validation questions and CPU threads must be positive")
    if (args.output_dir / "results.json").exists():
        parser.error("Output already contains results.json; choose a new directory to preserve the run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    revision = args.dataset_revision or HfApi().dataset_info("allenai/ai2_arc").sha
    task_yaml = Path(lm_eval.__file__).parent / "tasks/arc/arc_easy.yaml"
    task_config = yaml.safe_load(task_yaml.read_text())
    task_config["dataset_kwargs"] = {"revision": revision}
    task = ConfigurableTask(config=task_config)
    lm = ChadGPTLM(args.model_dir, args.device, args.validate_questions)
    weight_hash = sha256(args.model_dir / "model.safetensors")
    training_info = json.loads((args.model_dir / "training_info.json").read_text())
    if weight_hash != training_info["export"]["weight_sha256"]:
        raise ValueError("Local weights do not match the recorded release export")
    print(f"Evaluating final ChadGPT SFT checkpoint on {lm.device}, float32; "
          f"ARC-Easy test split, 0-shot, revision {revision}", flush=True)
    result = lm_eval.simple_evaluate(
        model=lm, tasks=[task], num_fewshot=0, device=str(lm.device),
        limit=args.limit, bootstrap_iters=100000, log_samples=True,
        apply_chat_template=False, random_seed=0, numpy_random_seed=1234,
        torch_random_seed=1234, fewshot_random_seed=1234,
    )
    samples = result.pop("samples")["arc_easy"]
    if args.limit is None and len(samples) != len(task.test_docs()):
        raise ValueError("Full evaluation did not score every test question")
    # Independently recompute both metrics from the saved answer scores.
    correct = correct_norm = 0
    for sample in samples:
        doc = sample["doc"]
        scores = np.array([response[0][0] for response in sample["resps"]])
        gold = doc["choices"]["label"].index(doc["answerKey"])
        correct += int(scores.argmax() == gold)
        lengths = np.array([len(choice) for choice in doc["choices"]["text"]])
        correct_norm += int((scores / lengths).argmax() == gold)
    metrics = result["results"]["arc_easy"]
    assert np.isclose(metrics["acc,none"], correct / len(samples))
    assert np.isclose(metrics["acc_norm,none"], correct_norm / len(samples))
    dataset_hash = hashlib.sha256()
    for doc in task.test_docs():
        dataset_hash.update((json.dumps(doc, sort_keys=True, ensure_ascii=False) + "\n").encode())
    result["provenance"] = {
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "model": "sam-eer12/chadGPT", "phase": training_info["phase"],
        "sft_step": training_info["step"], "weight_dtype": "float32",
        "weight_sha256": weight_hash,
        "source_checkpoint_sha256": training_info["export"]["source_checkpoint_sha256"],
        "model_files_sha256": {name: sha256(args.model_dir / name) for name in
                               ["config.json", "tokenizer_config.json", "tokenizer.tiktoken"]},
        "inference_files_sha256": {name: sha256(INFERENCE_DIR / name) for name in
                                   ["inference.py", "modeling_chadgpt.py"]},
        "dataset": "allenai/ai2_arc", "subset": "ARC-Easy", "split": "test",
        "dataset_revision": revision, "test_docs_sha256": dataset_hash.hexdigest(),
        "task_yaml_sha256": sha256(task_yaml), "num_fewshot": 0,
        "apply_chat_template": False, "add_bos_token": False,
        "prompt": "Question: {question}\nAnswer:",
        "continuation": " {answer_text}",
        "acc_norm_definition": "Answer log-likelihood divided by answer text character count",
        "samples_evaluated": len(samples), "full_test_split": args.limit is None,
        "correct_acc": correct, "correct_acc_norm": correct_norm,
        "uniform_random_expected_accuracy": float(np.mean([
            1 / len(sample["doc"]["choices"]["text"]) for sample in samples])),
        "device": str(lm.device), "cpu_threads": args.cpu_threads,
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ["lm_eval", "datasets", "torch", "tiktoken", "safetensors", "numpy"]},
        "script_sha256": sha256(__file__),
        "source_git_commit": source_git_commit(),
        "cached_scoring_validation": {
            "questions_checked": len(lm.validation_errors),
            "max_absolute_loglikelihood_error": max(lm.validation_errors),
            "reference": "Uncached full-sequence FP32 forward pass for every answer",
            "answer_rankings_match": True,
        },
        "max_input_tokens": lm.max_input_tokens, "truncated_inputs": 0,
        "decontamination": "Not performed; training data overlap with ARC was not audited",
    }
    with (args.output_dir / "samples.jsonl").open("w") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved {len(samples)} evaluated questions to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
