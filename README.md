# projectSLM

Hands-on deep learning notebooks using PyTorch and PyTorch Geometric, starting from tensor basics and progressing to CNNs and message-passing GNNs.

## What This Repo Covers

| Notebook | Focus Area | Highlights |
|---|---|---|
| book1.ipynb | PyTorch tensor fundamentals | Tensor ops, shapes, dtypes, indexing, device basics |
| book2.ipynb | Linear regression workflow | Full training loop, optimization, model persistence |
| book3.ipynb | Neural network classification | Binary/multiclass setups, metrics, decision boundaries |
| book4.ipynb | CNNs on FashionMNIST | Conv blocks, pooling, evaluation, confusion matrix |
| book5.ipynb | Additional experiments | Extra/iterative practice notebook |
| gnn.ipynb | Intro GNN concepts | Graph data handling with PyG |
| gnn_mp.ipynb | Message-passing GNN on QM9 | Custom MessagePassing layer, graph regression, training curves/animation |

## Current Project Structure

```text
projectSLM/
├── book1.ipynb
├── book2.ipynb
├── book3.ipynb
├── book4.ipynb
├── book5.ipynb
├── gnn.ipynb
├── gnn_mp.ipynb
├── helper_function.py
├── requirements.txt
├── README.md
├── data/
│   ├── FashionMNIST/
│   ├── MNIST/
│   └── QM9/
├── dataset/
│   └── images/
│       ├── train/
│       └── test/
├── models/
│   ├── best.pt
│   ├── fashionmnist.pt
│   ├── gnn_model.pt
│   └── model_3.pt
└── train/, test/ (pizza/steak/sushi image folders)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open notebooks in VS Code or Jupyter and run cells from top to bottom.

## Running the GNN QM9 Notebook

The main graph-regression workflow is in gnn_mp.ipynb.

1. Load QM9 and create train/test DataLoaders.
2. Define CustomMPNNLayer and CustomQM9Model.
3. Train with MSE loss on target index 0.
4. Visualize molecule graphs and training progress.

Notes for stability in gnn_mp.ipynb:
- Use a separate criterion variable (for example, criterion = nn.MSELoss()) to avoid shadowing.
- For graph drawing, color nodes using per-node features such as atomic numbers (data.z), not graph-level target vectors.
- The notebook uses a regression-style tolerance metric in addition to MSE.

## Saved Models

| File | Purpose |
|---|---|
| models/model_3.pt | Classification experiment checkpoint |
| models/fashionmnist.pt | FashionMNIST model checkpoint |
| models/best.pt | Best CNN checkpoint |
| models/gnn_model.pt | GNN checkpoint |

## Main Dependencies

- torch, torchvision, torchaudio
- torch-geometric
- matplotlib
- networkx
- scikit-learn
- torchmetrics
- numpy, pandas
- tqdm
- pillow

See requirements.txt for exact versions.

## Helper Utilities

helper_function.py contains reusable plotting and evaluation helpers used by multiple notebooks.

