# projectSLM — PyTorch Deep Learning from Scratch

A hands-on, progressive journey through deep learning fundamentals using **PyTorch**. This project is organized as a series of Jupyter notebooks (books 1–4), each building on the concepts of the previous one — starting from tensor basics and culminating in Convolutional Neural Networks (CNNs) for image classification.

## Table of Contents

- [Overview](#overview)
- [Notebooks](#notebooks)
  - [Book 1 — PyTorch & Tensor Fundamentals](#book-1--pytorch--tensor-fundamentals)
  - [Book 2 — Linear Regression (PyTorch Workflow)](#book-2--linear-regression-pytorch-workflow)
  - [Book 3 — Neural Network Classification](#book-3--neural-network-classification)
  - [Book 4 — Convolutional Neural Networks (CNNs)](#book-4--convolutional-neural-networks-cnns)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Saved Models](#saved-models)
- [Key Dependencies](#key-dependencies)

## Overview

| Notebook | Topic | Key Concepts |
|----------|-------|-------------|
| `book1.ipynb` | Tensor Fundamentals | Tensor creation, operations, reshaping, indexing |
| `book2.ipynb` | Linear Regression | Data prep, training loop, loss/optimizer, model saving |
| `book3.ipynb` | Classification | Binary & multi-class classification, non-linear data, metrics |
| `book4.ipynb` | CNNs | Conv2d, pooling, FashionMNIST, confusion matrix |

## Notebooks

### Book 1 — PyTorch & Tensor Fundamentals

Covers the foundational building blocks of PyTorch:

- Creating tensors (scalars, vectors, matrices, n-dimensional)
- Random tensors and reproducibility with random seeds
- Special tensors (zeros, ones)
- Tensor attributes (`dtype`, `shape`, `ndim`, `device`)
- Tensor operations — addition, subtraction, element-wise multiplication, division, matrix multiplication
- Transposing tensors
- Reshaping, stacking, squeezing, and unsqueezing
- Tensor views and permutations
- GPU/MPS device availability checks

### Book 2 — Linear Regression (PyTorch Workflow)

Implements a complete PyTorch workflow using a simple linear regression problem:

- **Data preparation** — generating synthetic linear data (`y = 0.7x + 0.3`), train/test split
- **Visualization** — plotting training and test data
- **Model building** — defining a `LinearRegression` model with `nn.Module`, `nn.Parameter`
- **Key PyTorch modules** — `torch.nn`, `torch.optim`, `nn.Parameter`, `forward()`
- **Making predictions** before and after training
- **Loss functions** — L1Loss (MAE)
- **Optimizers** — SGD
- **Training loop** — forward pass, loss computation, backpropagation, gradient descent
- **Saving & loading models** — `torch.save()`, `torch.load()`, `load_state_dict()`

### Book 3 — Neural Network Classification

Tackles classification problems of increasing complexity:

- **Binary classification** on the `make_circles` dataset (sklearn)
- Building classification models with `nn.Module` and `nn.Sequential`
- **Loss functions** — `BCELoss`, `BCEWithLogitsLoss` (for binary), `CrossEntropyLoss` (for multi-class)
- **Activation functions** — Sigmoid, ReLU
- Decision boundary visualization using helper functions
- **Improving models** — more layers, more hidden units, more epochs, different learning rates
- **Non-linear data** — solving non-linearly separable datasets with hidden layers and ReLU
- **Multi-class classification** on `make_blobs` (4 classes)
- **Evaluation metrics** — accuracy, precision, recall, F1 score, confusion matrix (`torchmetrics`)
- Saving trained models to the `models/` directory

### Book 4 — Convolutional Neural Networks (CNNs)

Applies deep learning to image classification on the **FashionMNIST** dataset:

- **Dataset & DataLoaders** — loading FashionMNIST, mini-batch preparation (`batch_size=32`)
- **Baseline model** (`FashionMnistModelV0`) — a simple flatten + linear + ReLU architecture
- **CNN model** (`FashionMnistModelV1`) — `Conv2d`, `ReLU`, `MaxPool2d` convolutional blocks
- **Understanding Conv2d** — kernel size, stride, padding, and their effect on spatial dimensions
- **Training loop** with tqdm progress bars, timing, and accuracy tracking
- **Evaluation** — `eval_mode()` function, per-class performance
- **Confusion matrix** visualization with sklearn
- **Custom image prediction** — loading and predicting on external images
- CPU vs GPU training considerations
- Model saving (`fashionmnist.pt`, `best.pt`)

## Project Structure

```
projectSLM/
├── book1.ipynb            # Tensor fundamentals
├── book2.ipynb            # Linear regression workflow
├── book3.ipynb            # Classification (binary & multi-class)
├── book4.ipynb            # CNNs with FashionMNIST
├── book5.ipynb            # (Placeholder for future work)
├── helper_function.py     # Reusable utilities (plotting, metrics, timing)
├── requirements.txt       # Python dependencies
├── README.md
├── data/
│   └── FashionMNIST/      # Downloaded FashionMNIST dataset
├── models/
│   ├── best.pt            # Best CNN model checkpoint
│   ├── fashionmnist.pt    # Baseline FashionMNIST model
│   └── model_3.pt         # Classification model from Book 3
└── __pycache__/
```

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/projectSLM.git
cd projectSLM

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Then open any notebook in VS Code or Jupyter and run cells sequentially.

## Saved Models

| File | Source | Description |
|------|--------|-------------|
| `models/model_3.pt` | Book 3 | Non-linear binary classification model |
| `models/fashionmnist.pt` | Book 4 | Baseline flatten + linear model on FashionMNIST |
| `models/best.pt` | Book 4 | Best-performing CNN model on FashionMNIST |

## Key Dependencies

- **PyTorch** (`torch`, `torchvision`, `torchaudio`) — core deep learning framework
- **matplotlib** — visualization and plotting
- **scikit-learn** — datasets (`make_circles`, `make_blobs`), train/test split, confusion matrix
- **torchmetrics** — classification metrics (F1 score, accuracy)
- **mlxtend** — additional ML utilities
- **pandas** / **numpy** — data manipulation
- **tqdm** — progress bars for training loops
- **Pillow** — image loading for custom predictions

See [requirements.txt](requirements.txt) for the full list.

## Helper Functions

The `helper_function.py` module provides reusable utilities used across all notebooks:

| Function | Description |
|----------|-------------|
| `plot_predictions()` | Plots train/test data and optional predictions |
| `plot_decision_boundary()` | Visualizes model decision boundaries for 2D classification |
| `accuracy_fn()` | Computes accuracy between predictions and ground truth |
| `print_train_time()` | Measures and prints training duration |
| `plot_loss_curves()` | Plots training and test loss/accuracy curves |
| `pred_and_plot_image()` | Predicts on a single image and displays the result |

