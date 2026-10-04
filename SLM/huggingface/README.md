---
license: mit
language:
- en
pipeline_tag: text-generation
library_name: pytorch
datasets:
- HuggingFaceTB/smollm-corpus
- teknium/OpenHermes-2.5
tags:
- chadgpt
- small-language-model
- trained-from-scratch
- causal-lm
- conversational
- chatml
- grouped-query-attention
- rope
- instruction-tuned
- educational
---

# ChadGPT — 250M instruction-tuned language model

ChadGPT is a **249,832,448-parameter decoder-only language model** built and trained from scratch by [sam-eer12](https://huggingface.co/sam-eer12) as part of [Project SLM](https://github.com/sam-eer12/projectSLM). This release contains the final **350-step supervised fine-tuning checkpoint**, with a **4,096-token context window** and ChatML conversation formatting.

The model uses a custom PyTorch implementation. Run it with the included `inference.py` and `modeling_chadgpt.py`, or use a Hugging Face Gradio Space as described below. This release does not implement the Transformers `AutoModelForCausalLM` interface.

## Test in your browser on Hugging Face

### Temporary public playground with a notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://huggingface.co/sam-eer12/chadGPT/colab)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://huggingface.co/sam-eer12/chadGPT/kaggle)

Click either button, sign in to that platform, and run **Install dependencies → Download and load → Test inference**. The notebook automatically fetches the Gradio app, inference code, tokenizer, configuration, and approximately 1 GB of model weights from this public repository. No cloning, manual file uploads, Hugging Face PRO subscription, or model-download token is needed.

The optional **Launch and share** cell uses the same Gradio app and calls `demo.launch(share=True)` to generate a public `https://…gradio.live` link. You can also [download the notebook](https://huggingface.co/sam-eer12/chadGPT/resolve/main/chadgpt_playground.ipynb?download=true). The root [`notebook.ipynb`](notebook.ipynb) is an identical copy used by [Hugging Face's Colab and Kaggle launch integration](https://huggingface.co/docs/hub/notebooks).

On Kaggle, enable **Internet** in the notebook settings and optionally select a GPU accelerator. CUDA is used when available; CPU also works. Keep the last cell running and share the public URL it prints. The link lasts only while the notebook runtime is active, with a maximum share-link lifetime of one week. See the [notebook setup guide](https://huggingface.co/sam-eer12/chadGPT/blob/main/playground/README.md) and [Gradio's sharing documentation](https://gradio.app/guides/sharing-your-app).

For Colab, the **Test inference** cell runs directly in the notebook. Colab's free managed runtimes restrict using a web UI as the primary interface and may terminate a shared Gradio demo; use the optional sharing cell only in a runtime whose rules permit it. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html). This route hosts generation in your notebook runtime; the Hugging Face model page provides the weights and launcher files.

### Persistent playground using a Hugging Face Space

The included [playground files](playground/) run the model in a **Gradio Space**, so visitors can chat entirely on Hugging Face without downloading the model or installing Python.

To set up the playground from the Hugging Face website:

1. Open [Create a new Space](https://huggingface.co/new-space), select your owner, and name it `chadGPT-playground`.
2. Select **Gradio**, **Public** visibility, and **CPU Basic** hardware.
3. Open the Space's **Files** tab and use **Add file → Upload files** to upload the four files from this repository's [`playground/`](playground/) folder into the **Space root**: `app.py`, `requirements.txt`, `README.md`, and `LICENSE`.
4. Keep the YAML header in the playground README. Hugging Face will install the dependencies, download the published model, and start the app automatically.
5. When the status is **Running**, open **App** and send a message. Share the Space URL with anyone who wants to try the model.

The interface supports multi-turn chat, example prompts, a system prompt, temperature, top-k, and an output-length slider. CPU generation takes longer than GPU generation, so the playground defaults to short responses. No API key or Inference Provider is required. The normal model-page inference widget is not used for this custom architecture; the Space executes the included PyTorch code.

CPU Basic has no hourly hardware charge. Current Hugging Face account rules may require a paid plan to create a Gradio compute Space; if the website prompts for one, that account step must be completed there. See the official [Spaces setup and account requirements](https://huggingface.co/docs/hub/spaces-overview). No paid GPU is required for this playground.

## Quickstart

Download the model and install the four inference dependencies:

```bash
hf download sam-eer12/chadGPT --local-dir chadGPT
python -m pip install -r chadGPT/requirements.txt
python chadGPT/inference.py \
  --prompt "Write a Python function that checks if a string is a palindrome." \
  --max-new-tokens 256 --temperature 0.7 --top-k 40
```

The script automatically selects CUDA, Apple MPS, or CPU. Use `--device cpu` to choose CPU explicitly or `--temperature 0` for greedy decoding. The FP32 weights occupy about **1.00 GB** on disk; runtime memory also includes activations and the KV cache.

To use the model from Python after downloading the repository:

```python
import sys
sys.path.insert(0, "chadGPT")
from inference import load_model, chat

model, tokenizer = load_model("chadGPT", device="auto")
print(chat(model, tokenizer, "Explain grouped-query attention in simple terms."))

# Pass the full conversation for multi-turn chat.
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a neural network?"},
    {"role": "assistant", "content": "A neural network learns patterns from examples."},
    {"role": "user", "content": "Give a simple example."},
]
print(chat(model, tokenizer, messages=messages, max_new_tokens=128))
```

The prompt, conversation history, and requested output must fit together within 4,096 tokens. The loader uses only files in the downloaded folder, so inference can run offline after dependencies and model files are installed.

## Architecture

| Property | Value |
|---|---|
| Trainable parameters | 249,832,448 (approximately 250M) |
| Transformer blocks | 18 |
| Hidden dimension | 1,024 |
| Feed-forward dimension | 4,096; GELU activation |
| Query heads / key-value heads | 16 / 4; head dimension 64 |
| Attention | Causal grouped-query attention; KV caching during generation |
| Normalization | Pre-LayerNorm; epsilon 1e-5 |
| Position encoding | RoPE, base 10,000; position interpolation factor 0.25 |
| Context | 1,024 in base pretraining; extended to 4,096 |
| Vocabulary | 50,259: GPT-2 BPE plus two ChatML tokens |
| Embeddings | Input embeddings tied to the output projection |
| Dropout | 0.0 |
| Released weight dtype | float32; preserved from the source checkpoint |

## Tokenizer and prompt format

The GPT-2 tokenizer was extended without changing its existing token IDs:

| Token | ID | Role |
|---|---|---|
| `<\|endoftext\|>` | 50256 | End of text; padding during training |
| `<\|im_start\|>` | 50257 | Start of a ChatML message |
| `<\|im_end\|>` | 50258 | End of a ChatML message; generation stop token |

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Your question here.<|im_end|>
<|im_start|>assistant
```

`chat()` applies this format automatically. `tokenizer.tiktoken` and `tokenizer_config.json` store the exact merge ranks, regex pattern, and special token IDs used by the included loader.

## Training

**The model was trained on Kaggle using two NVIDIA T4 GPUs. The full training process took approximately two months end to end**, covering data preparation, base pretraining, context extension, and supervised instruction tuning. The timeline is the author's elapsed training timeline, not a measurement of uninterrupted GPU compute time.

The training code uses Hugging Face Accelerate for two-process distributed training with FP16 mixed precision, AdamW, cosine learning-rate decay, warmup, gradient accumulation, and periodic evaluation and checkpointing. Training has three stages:

1. **Pretraining:** next-token prediction on the `cosmopedia-v2` subset of [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), using a 1,024-token context. The data preparation script targets 50 shards of 100M tokens each: 40 for training and 10 reserved for validation/context-extension use. This is a 5B-token prepared corpus, rather than a claim of 5B unique training tokens.
2. **Context extension:** continued training at 4,096 tokens with RoPE position interpolation (`position * 0.25`) and activation checkpointing. The source configuration uses shards 41–42 for training and shard 43 for validation.
3. **Instruction tuning:** ChatML conversations sampled from [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5). The preparation script targets approximately 80,000 conversations across coding, STEM, writing, reasoning, mathematics, roleplay, and multilingual categories. It packs conversations into 4,097-token rows, resets positions at conversation boundaries, and masks loss to assistant responses and their closing tokens. The final checkpoint does not record the exact sampled conversation count.

### Complete Kaggle workflow

The original [training and inference notebook](chadgpt.ipynb) is included, alongside the executable [`training/`](training/) scripts. The notebook retains its original local/Kaggle file paths and `.pt` checkpoint-loading cells. For the published Safetensors release, use the playground or the packaged inference code in the Quickstart.

1. **Prepare the Kaggle environment.** Create a Kaggle notebook, select the two-T4 GPU accelerator, enable internet access for dataset downloads, and place the six `training/` Python scripts in `/kaggle/working`. Install `torch`, `numpy`, `tiktoken`, `accelerate`, `datasets`, and `langdetect`. Run the stages from `/kaggle/working` so the relative shard paths match the scripts.
2. **Build the pretraining corpus.** `dataset_creation.py` streams Cosmopedia-v2, tokenizes with GPT-2 BPE, appends end-of-text markers, and saves the planned 50 token shards. This separates corpus preparation from training.
3. **Train the base model from scratch.** `chadgpt.py` trains the 18-layer decoder at 1,024 tokens with peak learning rate `3e-4`, 500 warmup steps, and 64 gradient-accumulation microsteps. The configuration targets 15,259 optimizer steps. With batch size 2 on each of two GPUs, the effective batch is 262,144 input tokens per step. It saves model, optimizer, and progress state so training can resume across Kaggle sessions.
4. **Extend the context.** `chadgpt_finetune.py` initializes from the base `latest.pt`, resets the optimizer, applies RoPE position interpolation at `0.25`, and continues at 4,096 tokens. Its configuration targets 500 steps at peak learning rate `3e-5`, with activation checkpointing and batch size 1 per GPU. Phase-2 progress is saved separately as `checkpoint.pt`.
5. **Prepare instruction data.** `prepare_openhermes_sft.py` samples the OpenHermes categories, formats conversations with ChatML, adds the two new special tokens, packs sequences, and writes token, assistant-loss-mask, and position-ID shards.
6. **Instruction-tune the model.** `chadgpt_sft.py` loads the context-extended checkpoint, grows the tied vocabulary from 50,257 to 50,259, keeps the same RoPE scaling, and trains only assistant targets for 350 steps at peak learning rate `1.5e-5`. It writes `sft_checkpoint.pt`; the final downloaded checkpoint used for this release is stored locally as `sft_latest.pt`.
7. **Export and publish.** The final model state is exported to `model.safetensors` without optimizer or scaler state and without changing precision. The tokenizer and inference implementation are checked against the original checkpoint before publication.

After installing dependencies and copying the scripts, the stage commands are:

```bash
cd /kaggle/working
python dataset_creation.py
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 chadgpt.py
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 chadgpt_finetune.py
python prepare_openhermes_sft.py
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 chadgpt_sft.py
```

These commands describe the full workflow; keep the shards and checkpoints between Kaggle sessions rather than rebuilding them for every resumed run. The configured stage lengths above come from the source scripts; the final saved checkpoint counters below are the verified record for this release.

### Verified checkpoint metadata

| Field | Saved value |
|---|---|
| Source checkpoint | `models/sft_latest.pt` |
| Phase | `sft` |
| SFT optimizer steps | 350 |
| Tokens processed before SFT | 4,246,732,800 |
| SFT tokens processed | 183,500,800 |
| Total tokens processed | 4,430,233,600 |
| SFT learning rate | 1.5e-5; cosine schedule; 15 warmup steps |
| Optimizer | AdamW; betas (0.9, 0.95); weight decay 0.1 |
| Microbatch / gradient accumulation | 1 sequence per device / 64 microsteps |
| Gradient clipping | 1.0 |

Token counts are training-loop counters, including repeated examples and SFT tokens whose loss is masked. They do not measure unique corpus size or the number of assistant tokens contributing to the loss. See `training_info.json` for the saved configuration and export provenance, and `training/` for the project training scripts.

## Evaluation and limitations

### ARC-Easy

The final 350-step SFT checkpoint was evaluated on **all 2,376 ARC-Easy test questions**, **zero-shot**, using [EleutherAI's LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) **0.4.13** and a custom adapter for ChadGPT's PyTorch architecture. Evaluation date: **October 4, 2026**.

| Benchmark | Shots | Metric | Score | Correct / total |
|---|---|---|---|---|
| ARC-Easy (test) | 0 | Accuracy (`acc`) | **42.55% ± 1.01 pp** | 1,011 / 2,376 |
| ARC-Easy (test) | 0 | Length-normalized accuracy (`acc_norm`) | **41.37% ± 1.01 pp** | 983 / 2,376 |

Uncertainty values are one standard error in percentage points. The run used the original **FP32 weights** on Apple MPS, the harness's standard `Question: {question}\nAnswer:` prompt, and answer-text conditional log-likelihoods, with **no chat template and no added BOS token**. `acc` selects the answer with the highest summed token log-likelihood; `acc_norm` divides that score by the answer text's character count before selecting. Every test question was scored, with no input truncation.

The dataset is [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc), configuration `ARC-Easy`, pinned to revision `210d026faf9955653af8916fad021475a3f00453`. See [`evaluation/arc_easy/`](evaluation/arc_easy/) for the full results, per-question answer scores, weight and dataset hashes, exact settings, and reproduction instructions. The included [`evaluate_arc_easy.py`](evaluate_arc_easy.py) runs directly from a downloaded model folder after installing `requirements-eval.txt`; its source is maintained in the project's `SLM/evaluate_arc_easy.py`.

These results measure multiple-choice science-question answer likelihoods. Conversational generation and instruction-following quality were not measured by this benchmark. Training-data overlap with ARC has not been audited, and no decontamination was performed.

### Other limitations

The checkpoint contains no saved validation loss or perplexity, so this card does not report those metrics. Export verification checks weight equality, tokenizer compatibility, finite inference logits, and cached decoding against the original SFT implementation; those checks are not an evaluation of answer quality.

ChadGPT is an educational and research model for exploring small language models, local text generation, and instruction tuning. It can produce incorrect facts, weak reasoning, repetitive text, invalid code, or biased and offensive content. English is the primary language; multilingual capability is not evaluated. The 4,096-token configuration is not evidence of reliable long-context retrieval. Review generated text and test generated code before use. This release is not validated for high-stakes decisions.

## Files and license

- `model.safetensors`: inference weights with the shared output embedding stored once; optimizer and gradient-scaler state are excluded.
- `config.json`: architecture and position-interpolation settings.
- `tokenizer.tiktoken`, `tokenizer_config.json`: complete tokenizer data.
- `modeling_chadgpt.py`, `inference.py`, `requirements.txt`: standalone PyTorch loading and chat code.
- `training_info.json`, `training/`: checkpoint metadata, provenance, and source training scripts.
- `evaluation/arc_easy/`: full zero-shot ARC-Easy results, per-question scores, and reproduction instructions.
- `evaluate_arc_easy.py`, `requirements-eval.txt`: standalone benchmark runner and evaluation dependencies.
- `chadgpt.ipynb`: the original Kaggle training and local inference notebook.
- `chadgpt_playground.ipynb`: a launcher notebook for temporary public Gradio demos.
- `notebook.ipynb`: the same launcher under the filename required for direct Colab and Kaggle opening.
- `playground/`: ready-to-upload Gradio Space files and website setup instructions.
- `LICENSE`: MIT license, matching the license selected for this model repository.

Training datasets retain their respective licenses and source attributions; consult the linked dataset cards. The GPT-2 BPE tokenizer originates from [OpenAI's GPT-2](https://github.com/openai/gpt-2) and is used through [tiktoken](https://github.com/openai/tiktoken).
