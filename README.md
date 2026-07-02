# Project SLM (Small Language Models & Deep Learning)

Welcome to **Project SLM**! This repository is a comprehensive, hands-on learning environment for deep learning, starting from PyTorch tensor fundamentals, advancing through Convolutional and Graph Neural Networks, and culminating in the custom construction, pre-training, and instruction-tuning of a **250M parameter Small Language Model (ChadGPT)**.

---

## Table of Contents
1. [Core Deep Learning Notebooks (`book1` to `book5`)](#1-core-deep-learning-notebooks)
2. [Graph Neural Networks (GNNs)](#2-graph-neural-networks-gnns)
3. [From Scratch SLM/LLM Learning Series (`SLM - learning/`)](#3-from-scratch-slmllm-learning-series)
4. [ChadGPT: 250M Parameter GPT Model (`chadgpt.py` / `.ipynb`)](#4-chadgpt-250m-parameter-gpt-model)
5. [Repository Structure](#5-repository-structure)
6. [Saved Checkpoints & Models](#6-saved-checkpoints--models)
7. [Helper Utilities (`helper_function.py`)](#7-helper-utilities)
8. [Installation & Setup](#8-installation--setup)

---

## 1. Core Deep Learning Notebooks

A progressive step-by-step introduction to deep learning using PyTorch:

| Notebook | Focus Area | Key Concepts Covered |
| :--- | :--- | :--- |
| **[book1.ipynb](book1.ipynb)** | PyTorch Tensor Fundamentals | Tensor creation, shapes, datatypes, math operations, indexing, and GPU/device allocation basics. |
| **[book2.ipynb](book2.ipynb)** | Linear Regression Workflow | Data preparation, modeling, loss functions, optimizers, training/testing loops, and model saving/loading. |
| **[book3.ipynb](book3.ipynb)** | Neural Network Classification | Binary and multiclass classification, activation functions (ReLU, Sigmoid, Softmax), evaluation metrics (Accuracy), and decision boundary plotting. |
| **[book4.ipynb](book4.ipynb)** | CNNs on FashionMNIST | Convolutional layers (`nn.Conv2d`), pooling (`nn.MaxPool2d`), classification architectures, training pipelines, and confusion matrices. |
| **[book5.ipynb](book5.ipynb)** | Custom Dataset Training | Building custom `Dataset` and `DataLoader` pipelines to classify custom food images (pizza, steak, sushi), transfer learning preparation, and data augmentation. |

---

## 2. Graph Neural Networks (GNNs)

Deep learning applied to graph-structured data using **PyTorch Geometric (PyG)**:

*   **[gnn.ipynb](gnn.ipynb) (Introduction to GCNs):**
    *   Covers handling graph data structures (`torch_geometric.data.Data`).
    *   Trains a Graph Convolutional Network (GCN) on the Cora dataset for node classification.
    *   Visualizes node embedding projections as they cluster during training.
*   **[gnn_mp.ipynb](gnn_mp.ipynb) (Message-Passing GNN on QM9):**
    *   Implements a custom **Message-Passing Neural Network (MPNN)** layer (`CustomMPNNLayer`) from scratch.
    *   Defines a regression model (`CustomQM9Model`) to predict chemical properties of molecules using the QM9 dataset.
    *   Includes training evaluation with regression tolerance metrics and visual molecular graph plots.

---

## 3. From Scratch SLM/LLM Learning Series

Located in the **[SLM - learning/](SLM%20-%20learning)** directory, this series walks through building and tuning a Small Language Model from first principles:

1.  **[tokenizer.ipynb](SLM%20-%20learning/tokenizer.ipynb):** Explores text processing, byte pair encoding (BPE) using GPT-2 tokenizers, special context tokens, and formatting sliding-window input-target pairs for autoregressive training.
2.  **[vembedding.ipynb](SLM%20-%20learning/vembedding.ipynb):** Implements token and positional embeddings, showing how integer token IDs map to continuous vector spaces.
3.  **[attention.ipynb](SLM%20-%20learning/attention.ipynb):** Walks step-by-step through self-attention, causal attention masking (preventing future token leakage), and multi-head attention.
4.  **[llm_architecture.ipynb](SLM%20-%20learning/llm_architecture.ipynb):** Assembles the full GPT-style decoder architecture by stacking embedding layers, multi-head attention, layer normalization, FeedForward blocks, and final linear output heads.
5.  **[Instruction_Tuning.ipynb](SLM%20-%20learning/Instruction_Tuning.ipynb):** Fine-tunes the constructed model on a custom dataset of 1,000 instruction-response pairs (**[instruction-data.json](SLM%20-%20learning/instruction-data.json)**).
    *   *Features:* Instruction prompt formatting (`### Instruction`, `### Input`, `### Response`), custom collation with padding, and **target masking** (using `-100` ignore index to compute loss only on response tokens, not prompt tokens).

---

## 4. ChadGPT: 250M Parameter GPT Model

**[chadgpt.py](chadgpt.py)** (and its notebook counterpart **[chadgpt.ipynb](chadgpt.ipynb)**) represents the flagship architecture in this repo—a high-performance, custom GPT model designed to scale.

### Key Architectural Features
*   **Grouped Query Attention (GQA):** Implements query grouping (shared Key/Value heads) to drastically reduce KV cache memory footprints while preserving generation quality.
*   **Rotary Positional Embeddings (RoPE):** Applies relative positional information by rotating Query and Key representations, allowing better sequence length extrapolation.
*   **KV Caching:** Caches historical token keys and values during generation to accelerate inference (speeds up autoregressive generation from $O(T^2)$ to $O(T)$).
*   **Weight Tying:** Shares weights between token embeddings and the pre-softmax linear output head.

### Training & Scaling Config
*   **Multi-GPU Training:** Powered by Hugging Face's **`Accelerate`** to perform 16-bit mixed-precision (FP16) training across multiple GPUs.
*   **Efficient Dataset Sharding:** Uses **[ShardDataset](chadgpt.py#L330)**, a custom `numpy.memmap` dataset that streams pre-tokenized tokens across 50 memory-mapped shards (40 train, 10 val) representing over **5 billion tokens** of the `smollm-corpus` (specifically `cosmopedia-v2`).
*   **Cosine Decaying Learning Rate:** Incorporates a linear warmup (500 steps) followed by a cosine learning rate decay down to $10\%$.
*   **Gradient Accumulation & Clipping:** Supports high effective batch sizes (e.g. batch size 2 $\times$ 64 gradient accumulation steps) and clips gradients to avoid exploding gradients.

---

## 5. Repository Structure

```text
projectSLM/
├── book1.ipynb                    # PyTorch tensor basics
├── book2.ipynb                    # Linear regression training loop
├── book3.ipynb                    # Binary & multi-class classification
├── book4.ipynb                    # CNNs on FashionMNIST dataset
├── book5.ipynb                    # Food image classification (Custom Dataset)
├── gnn.ipynb                      # Node classification with PyG
├── gnn_mp.ipynb                   # Custom Message-Passing GNN on QM9
├── chadgpt.py                     # 250M GPT model and distributed training script
├── chadgpt.ipynb                  # Notebook walk-through for ChadGPT
├── helper_function.py             # Reusable plotting & accuracy utilities
├── requirements.txt               # Package dependencies
├── graph_animation.gif            # GNN training visualization
│
├── SLM - learning/                # Build-your-own LLM tutorial series
│   ├── tokenizer.ipynb            # Text tokenization & BPE
│   ├── vembedding.ipynb           # Vector & positional embeddings
│   ├── attention.ipynb            # Self-attention & multi-head attention
│   ├── llm_architecture.ipynb     # Full GPT model structure from scratch
│   ├── Instruction_Tuning.ipynb   # SFT / Instruction-tuning implementation
│   ├── instruction-data.json      # 1,000 instruction-response tuning pairs
│   └── the-verdict.txt            # Corpus text used for exercises
│
├── data/                          # Dataset downloads (FashionMNIST, MNIST, QM9)
├── dataset/                       # Pizza/Steak/Sushi image directories
├── models/                        # Pre-trained models and check-points
└── requirements.txt               # Main dependencies
```

---

## 6. Saved Checkpoints & Models

Checkpoints are stored in the **`models/`** directory (not uploaded to Git due to size, but saved locally during training):

*   **`models/best.pt`** & **`models/fashionmnist.pt`**: Best CNN evaluation checkpoints.
*   **`models/model_3.pt`**: Classification model parameters.
*   **`models/gnn_model.pt`** & **`models/gnn_mpnn.pt`**: Trained Graph Neural Network checkpoints.
*   **`models/chadGPT.pt`**: Trained 250M Parameter GPT weights.
*   **`models/latest.pt`**: Main model, optimizer state, and training configuration backup for resuming runs.

---

## 7. Helper Utilities

**[helper_function.py](helper_function.py)** provides visual and analytical helper utilities used across the training notebooks:

*   `plot_decision_boundary()`: Plots classification decision regions on 2D space.
*   `plot_predictions()`: Draws ground-truth vs. regression prediction comparisons.
*   `plot_loss_curves()`: Graphically plots training/test accuracy and loss.
*   `pred_and_plot_image()`: Performs inference on individual images and plots predictions.
*   `download_data()`: Utility to download and unpack compressed data archives.

---

## 8. Installation & Setup

### Requirements
*   Python 3.10+
*   PyTorch 2.0+ (preferably with CUDA/MPS support)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sam-eer12/projectSLM.git
    cd projectSLM
    ```

2.  **Set up virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Running ChadGPT Pre-training:**
    Ensure you configure your accelerate device details:
    ```bash
    !accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 /kaggle/working/chadgpt.py
    ```
