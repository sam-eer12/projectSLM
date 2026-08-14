# Project SLM (Small Language Models & Deep Learning)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to **Project SLM**! This repository is a comprehensive, hands-on learning environment for modern deep learning and language modeling. It tracks a progressive journey starting from PyTorch tensor fundamentals, advancing through Convolutional and Graph Neural Networks, and culminating in the ground-up architecture design, multi-billion token pre-training, context extension, and instruction tuning of **ChadGPT** — a custom **250M parameter Small Language Model**.

---

## Table of Contents

1. [ChadGPT: 250M Parameter SLM (`SLM/`)](#1-chadgpt-250m-parameter-slm)
   - [Model Architecture & Hyperparameters](#model-architecture--hyperparameters)
   - [3-Stage Training Pipeline](#3-stage-training-pipeline)
   - [Interactive Inference & Chat](#interactive-inference--chat)
2. [Deep Learning Foundations (`DL/`)](#2-deep-learning-foundations)
   - [Core PyTorch Notebooks](#core-pytorch-notebooks)
   - [Graph Neural Networks (PyG)](#graph-neural-networks-pyg)
3. [From-Scratch SLM Tutorial Series (`SLM - learning/`)](#3-from-scratch-slm-tutorial-series)
4. [Saved Checkpoints & Models (`models/`)](#4-saved-checkpoints--models)
5. [Repository Structure](#5-repository-structure)
6. [Installation & Quickstart](#6-installation--quickstart)

---

## 1. ChadGPT: 250M Parameter SLM

All production implementation files for ChadGPT are located in the **[`SLM/`](SLM)** directory.

### Model Architecture & Hyperparameters

ChadGPT is built on a modern decoder-only transformer architecture tailored for efficient training and inference on commodity accelerators:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | ~250 Million | 18 Transformer Layers |
| **Embedding Dimension (`emb_dim`)** | 1024 | Hidden dimension across all attention and feed-forward layers |
| **Query Heads (`n_heads`)** | 16 | Multi-head query projections (head dim = 64) |
| **KV Heads (`n_kv_heads`)** | 4 | **Grouped Query Attention (GQA)**: 4 query heads per KV group (4× memory savings in KV cache) |
| **Positional Encoding** | RoPE | **Rotary Position Embedding** applied dynamically to Q/K |
| **Base Context Length** | 1024 | Pretrained sequence length (extended to 4096 in Phase 2) |
| **Extended Context Length** | 4096 | Scaled via **Position Interpolation (PI)** |
| **Vocabulary Size** | 50,257 → 50,259 | GPT-2 BPE, expanded with `<\|im_start\|>` (50257) & `<\|im_end\|>` (50258) for ChatML |
| **Weight Tying** | Enabled | Input embeddings tied with output linear projection (`logits = x @ tok_emb.weight.T`) |
| **Normalization** | Pre-LayerNorm | `LayerNorm` applied before attention and FFN blocks with residual connections |

---

### 3-Stage Training Pipeline

ChadGPT was trained end-to-end on Kaggle (2× NVIDIA T4 GPUs) using Hugging Face **Accelerate** in FP16 mixed precision:

```
[ Cosmopedia-v2 (5B tokens) ] ──> Phase 1: Pre-training (ctx=1024) ──> latest.pt
                                                                           │
                                                                           ▼
[ Pinned Shards 41-43 ]       ──> Phase 2: Context Extension (ctx=4096, PI) ──> checkpoint.pt
                                                                                      │
                                                                                      ▼
[ OpenHermes-2.5 (80k ChatML) ] ──> Phase 3: SFT with Loss Masking (ChatML) ──> sft_latest.pt
```

#### Phase 1: Base Pre-training
* **Data Preparation ([`SLM/dataset_creation.py`](SLM/dataset_creation.py)):** Streams and tokenizes `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`) into 50 memory-mapped `.npy` shards (40 train, 10 validation) of 100M tokens each (**5 Billion tokens total**).
* **Pre-training Loop ([`SLM/chadgpt.py`](SLM/chadgpt.py)):** Uses `ShardDataset` with zero-copy memory mapping, cosine learning rate decay with linear warmup (3e-4 peak, 500 warmup steps), and gradient accumulation (effective batch ~262k tokens/step).

#### Phase 2: Long-Context Extension (1024 → 4096)
* **Position Interpolation ([`SLM/chadgpt_finetune.py`](SLM/chadgpt_finetune.py)):** Rescales RoPE angles by `0.25` (`1024 / 4096`) so extended positions map directly into the pretrained frequency distribution without destabilizing attention.
* **Memory Optimization:** Implements `GPTModelFT` with **activation checkpointing** (`torch.utils.checkpoint`), cutting activation memory by >2.5 GB per GPU to run batch size 1 with context length 4096 on 16 GB T4 GPUs.
* **Checkpoint Safety:** Phase 1 weights are read-only (`init_from: latest.pt`); Phase 2 checkpoints write exclusively to `checkpoint.pt`.

#### Phase 3: Supervised Fine-Tuning (SFT / ChatML)
* **Tokenizer Extension ([`SLM/chatml_tokenizer.py`](SLM/chatml_tokenizer.py)):** Registers `<|im_start|>` (ID 50257) and `<|im_end|>` (ID 50258) without modifying base GPT-2 token IDs.
* **Dataset Sharding & Packing ([`SLM/prepare_openhermes_sft.py`](SLM/prepare_openhermes_sft.py)):** Stratified-samples ~80,000 multi-turn conversations from `teknium/OpenHermes-2.5` across 7 domains (coding, math, reasoning, roleplay, STEM, writing, multilingual). Formats conversations into ChatML, builds **loss masks** (computing loss only on assistant turns), and packs conversations into 4097-token rows with document-boundary position resets.
* **Masked Instruction Tuning ([`SLM/chadgpt_sft.py`](SLM/chadgpt_sft.py)):** Loads Phase 2 weights at base vocab size (50,257), expands embedding table to 50,259, and trains with masked cross-entropy loss over 350 steps (~2.5 epochs).

---

### Interactive Inference & Chat

The complete interactive workflow is available in **[`SLM/chadgpt.ipynb`](SLM/chadgpt.ipynb)**.

The notebook loads the final checkpoint (`models/sft_latest.pt`) and provides a `chat()` helper function that automatically formats prompts with ChatML tags:

```python
# User provides a plain-text prompt — ChatML framing is added automatically
response = chat(
    prompt="Write a Python function that checks if a string is a palindrome.",
    max_new_tokens=512,
    temperature=0.7,
    top_k=40
)
print(response)
```

The generation loop uses **KV caching** for $O(T)$ autoregressive decoding and terminates upon encountering `<|im_end|>` (token ID 50258).

---

## 2. Deep Learning Foundations

Located in the **[`DL/`](DL)** directory, this series establishes core deep learning mechanics:

### Core PyTorch Notebooks

| Notebook | Topic | Key Techniques |
| :--- | :--- | :--- |
| **[`book1.ipynb`](DL/book1.ipynb)** | Tensor Fundamentals | Tensor creation, shapes, GPU device allocation, broadcasting, indexing |
| **[`book2.ipynb`](DL/book2.ipynb)** | Linear Regression Workflow | `nn.Module`, parameter initialization, loss functions, optimizers, state dicts |
| **[`book3.ipynb`](DL/book3.ipynb)** | Classification & Decision Boundaries | Binary & multi-class classification, activation functions (ReLU, Sigmoid, Softmax), metric evaluation |
| **[`book4.ipynb`](DL/book4.ipynb)** | CNNs on FashionMNIST | `nn.Conv2d`, `nn.MaxPool2d`, flattening, feature maps, confusion matrices |
| **[`book5.ipynb`](DL/book5.ipynb)** | Custom Food Dataset Classification | Custom `Dataset` & `DataLoader`, torchvision transforms, data augmentation, TinyVGG |

### Graph Neural Networks (PyG)

* **[`gnn.ipynb`](DL/gnn.ipynb) (Node Classification with GCN):** Graph data structures (`torch_geometric.data.Data`), multi-layer Graph Convolutional Networks on Cora dataset, 2D embedding visualization.
* **[`gnn_mp.ipynb`](DL/gnn_mp.ipynb) (Message-Passing on QM9):** Custom Message-Passing Neural Network layer (`CustomMPNNLayer`), molecular property regression with QM9 dataset.

---

## 3. From-Scratch SLM Tutorial Series

Located in **[`SLM - learning/`](SLM%20-%20learning)**, this tutorial breaks down each component of a generative LLM:

1. **[`tokenizer.ipynb`](SLM%20-%20learning/tokenizer.ipynb):** Byte Pair Encoding (BPE), vocabulary mapping, sliding-window dataset generation.
2. **[`vembedding.ipynb`](SLM%20-%20learning/vembedding.ipynb):** Token embeddings, absolute learned positional embeddings.
3. **[`attention.ipynb`](SLM%20-%20learning/attention.ipynb):** Self-attention scores, causal triangular masking, multi-head attention.
4. **[`llm_architecture.ipynb`](SLM%20-%20learning/llm_architecture.ipynb):** Assembling residual connections, LayerNorm, FeedForward networks, and output heads.
5. **[`Instruction_Tuning.ipynb`](SLM%20-%20learning/Instruction_Tuning.ipynb):** Fine-tuning on 1,000 instruction-response pairs with prompt masking (`ignore_index=-100`).

---

## 4. Saved Checkpoints & Models

Checkpoints are stored in the **[`models/`](models)** directory:

| Checkpoint | File Size | Description |
| :--- | :--- | :--- |
| **`sft_latest.pt`** | ~2.8 GB | **Final ChadGPT SFT Checkpoint** (350 steps, ~4.4B total tokens seen). Primary model for inference. |
| **`latest.pt`** | ~2.8 GB | Phase 2 context-extended model checkpoint (4096 context length). |
| **`chadGPT.pt`** | ~312 MB | Phase 1 pretrained model weights. |
| **`model_3.pt`** | ~3.4 KB | Multi-class classification model (`book3.ipynb`). |
| **`best.pt`** | ~2 KB | TinyVGG Food101 classification weights (`book5.ipynb`). |
| **`fashionmnist.pt`** | ~52 KB | FashionMNIST CNN model (`book4.ipynb`). |
| **`gnn_model.pt`** | ~7 KB | GCN node classification checkpoint (`gnn.ipynb`). |
| **`gnn_mpnn.pt`** | ~237 KB | Custom MPNN regression checkpoint (`gnn_mp.ipynb`). |

---

## 5. Repository Structure

```text
projectSLM/
│
├── DL/                                # Deep Learning foundational notebooks
│   ├── book1.ipynb                    # PyTorch tensor basics
│   ├── book2.ipynb                    # Linear regression workflow
│   ├── book3.ipynb                    # Classification & decision boundaries
│   ├── book4.ipynb                    # CNNs on FashionMNIST
│   ├── book5.ipynb                    # Custom Food-101 image classification
│   ├── gnn.ipynb                      # Node classification with GCN
│   ├── gnn_mp.ipynb                   # Message-Passing GNN on QM9
│   ├── helper_function.py            # Plotting and metric utilities
│   └── graph_animation.gif           # Graph node clustering animation
│
├── SLM - learning/                    # Step-by-step LLM tutorial notebooks
│   ├── tokenizer.ipynb               # BPE tokenization walkthrough
│   ├── vembedding.ipynb              # Vector & positional embeddings
│   ├── attention.ipynb               # Causal self-attention & multi-head attention
│   ├── llm_architecture.ipynb        # GPT model assembly from scratch
│   ├── Instruction_Tuning.ipynb      # Simple instruction tuning with prompt masking
│   ├── instruction-data.json         # 1,000 instruction-response tuning pairs
│   └── the-verdict.txt               # Sample corpus for tokenization exercises
│
├── SLM/                               # ChadGPT — 250M Parameter Model Implementation
│   ├── chadgpt.py                    # Phase 1: Model architecture & distributed pre-training
│   ├── dataset_creation.py           # Phase 1: Cosmopedia-v2 download & shard creation (5B tokens)
│   ├── chadgpt_finetune.py           # Phase 2: Context extension (1024 -> 4096) with RoPE PI
│   ├── chatml_tokenizer.py           # Phase 3: ChatML tokenizer extension (<|im_start|>, <|im_end|>)
│   ├── prepare_openhermes_sft.py     # Phase 3: OpenHermes-2.5 stratified sampling & packing
│   ├── chadgpt_sft.py               # Phase 3: SFT training loop with masked cross-entropy
│   └── chadgpt.ipynb                 # Interactive ChatML inference notebook with chat() helper
│
├── models/                            # Trained weights & checkpoints (local storage)
├── data/                              # Dataset downloads (FashionMNIST, MNIST, QM9)
├── dataset/                           # Image dataset files
├── requirements.txt                   # Environment dependencies
└── README.md
```

---

## 6. Installation & Quickstart

### Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/sam-eer12/projectSLM.git
cd projectSLM

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt
```

### Running ChadGPT Inference Locally

Open **[`SLM/chadgpt.ipynb`](SLM/chadgpt.ipynb)** in Jupyter Lab or VS Code and run the notebook cells. The notebook will automatically load `models/sft_latest.pt` onto your active accelerator (`cuda`, `mps`, or `cpu`) and run the `chat()` helper function.

### Running Distributed Training on Kaggle (2× T4 GPUs)

```bash
cd SLM

# --- Step 1: Pre-training (5B tokens) ---
python dataset_creation.py
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 chadgpt.py

# --- Step 2: Context Extension (4096 tokens) ---
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 chadgpt_finetune.py

# --- Step 3: SFT Instruction Tuning ---
python prepare_openhermes_sft.py
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 chadgpt_sft.py
```
