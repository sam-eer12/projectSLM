# ChadGPT ARC-Easy evaluation

The final 350-step ChadGPT SFT checkpoint was evaluated on October 4, 2026.

| Metric | Score | Standard error | Correct / total |
|---|---|---|---|
| Accuracy (`acc`) | **42.55%** | 1.01 percentage points | 1,011 / 2,376 |
| Length-normalized accuracy (`acc_norm`) | **41.37%** | 1.01 percentage points | 983 / 2,376 |

All **2,376 test questions** were evaluated, with **zero examples** in the prompt.
Uniform random choice would give an expected accuracy of 25.02% on this split.

## Protocol

- Framework: [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), `lm_eval==0.4.13`, standard `arc_easy` task version 1.0.
- Dataset: [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc), `ARC-Easy`, `test`; pinned Hub revision `210d026faf9955653af8916fad021475a3f00453`.
- Model: `sam-eer12/chadGPT`, final SFT step 350; local `models/huggingface-chadgpt/model.safetensors`, matching the exported `models/sft_latest.pt` checkpoint.
- Precision and hardware: original FP32 weights, PyTorch 2.10.0, Apple M4 MPS. The full evaluation took approximately 208 seconds, including setup and checks.
- Prompt: `Question: {question}\nAnswer:`. Each answer continuation is a space followed by its answer text. No chat template, system instruction, or added BOS token is used.
- Scoring: sum the conditional log-probabilities of the answer tokens. `acc` selects the highest-scoring answer. The harness's `acc_norm` divides by the number of characters in the original answer text before selecting.
- No input truncation: the longest question plus answer was 159 tokens, within the 4,096-token context.
- Seeds: Python 0; NumPy, PyTorch, and few-shot selection 1234. Standard errors use the harness's analytic standard error of the mean.
- Decontamination: not performed. Potential overlap between training data and ARC has not been audited.

This benchmark measures science-question answer likelihoods. It does not measure conversational generation or instruction-following quality.

## Validation

The custom adapter inherits the harness's context/continuation tokenization. It
prefills each distinct prompt once and scores the answer continuations in a batch
using the prompt's KV cache. Duplicate test prompts share this computation; all
2,376 documents retain their own answer choices, labels, and metrics.

For the first eight test questions, every answer score was checked against an
uncached full-sequence FP32 forward pass. The largest absolute log-likelihood
difference was **0.00003433**, and both answer rankings matched. Both aggregate
metrics were independently recomputed from the saved answer scores and agreed
with the harness. Model and source-checkpoint SHA-256 checksums were verified.

## Artifacts

- [`results.json`](results.json): harness metrics, standard errors, task configuration, sample counts, versions, seeds, dataset revision, model and source hashes, runtime, and validation results.
- [`samples.jsonl`](samples.jsonl): one record per test question, including the prompt, choices, correct label, every answer log-likelihood, and both accuracy indicators.
- Evaluator: `SLM/evaluate_arc_easy.py` in the Project SLM repository.
- [`evaluate_arc_easy_at_run.py`](evaluate_arc_easy_at_run.py): the exact evaluator source used for this recorded run, matching `script_sha256` in `results.json`. The current evaluator also supports a standalone Hub download.
- Dependencies: `SLM/requirements-eval.txt`. Additional installed package versions are recorded in `results.json`.

Question text in `samples.jsonl` comes from the linked AI2 ARC dataset, distributed
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Reproduce

From the Project SLM root, with the exported model available at
`models/huggingface-chadgpt`:

```bash
.venv/bin/python -m pip install -r SLM/requirements-eval.txt
.venv/bin/python SLM/evaluate_arc_easy.py \
  --model-dir models/huggingface-chadgpt \
  --dataset-revision 210d026faf9955653af8916fad021475a3f00453 \
  --device mps \
  --output-dir models/eval-cache/arc_easy_reproduction
```

Use `--device cpu` or `--device cuda` for other hardware, or omit the flag to
select an available accelerator automatically. Use a fresh output directory;
the script preserves existing completed results. Omitting `--limit` evaluates
the full test split. For an eight-question scoring check, add `--limit 8` and
choose a separate output directory.

If the local export is missing, create it with
`.venv/bin/python SLM/export_huggingface.py` first. The exporter includes this
report with matching weights under `evaluation/arc_easy/` in the Hub package.

From a downloaded Hugging Face model folder, use the included standalone evaluator:

```bash
python -m pip install -r requirements-eval.txt
python evaluate_arc_easy.py \
  --dataset-revision 210d026faf9955653af8916fad021475a3f00453
```

This loads the weights in that folder and writes a new run to
`evaluation_runs/arc_easy/`. The published report remains in `evaluation/arc_easy/`.
