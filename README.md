# Scene Graph Decomposition

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)

A GNN-based model that, given a scene graph and a query node, predicts all triplets (edges + attributes) incident to that node.

## Example

```
Graph:  node3 -edge3- node1 -edge1- node2 -edge2- node4
Query:  node1
Output: node1-edge1-node2, node1-edge3-node3
```

## Architecture

| Component | Description |
|---|---|
| **Encoder** | Object node embeddings from `(class_idx, bbox)` refined by RGCN-style relational message passing |
| **Query head** | Concat query embedding with every candidate, MLP scores `link_prob` + `rel_logits` |
| **Decoder** | Threshold `link_prob`, emit predicted triplet strings |

The model is graph-size-agnostic — the same MLP scores every node, so it generalises to any graph without retraining.

## Results

Two decoding strategies were compared on the indoor scene-graph test set. See [`docs/Evaluation.md`](docs/Evaluation.md) for the full evaluation protocol, sample predictions, and discussion.

| Metric | Model 1 — threshold (`gnn_decompose.py`) | Model 2 — DistMult + Top-K (`gnn_decompose_new.py`) |
|---|---:|---:|
| Precision | 0.501 | **0.819** |
| Recall | **0.746** | 0.659 |
| F1 Score | 0.600 | **0.730** |
| False Positives | 3996 | **785** |
| False Negatives | **1367** | 1839 |

**Takeaway:** Model 1 (fixed-probability threshold) favors recall and recovers more ground-truth triplets, but at the cost of many false positives. Model 2 (DistMult scorer + degree-predicted Top-K decoding) trades a little recall for substantially higher precision, giving the better overall F1.

## Project structure

```
Scene-Graph-Decomposition/
├── prepare_data.py                 # Raw scene graphs -> compact graph format for GNN training
├── gnn_decompose.py                 # Model 1: RGCN encoder, threshold-based decoding
├── gnn_decompose_new.py              # Model 2: DistMult scorer, degree-based Top-K decoding
├── generate_predictions.py          # Run a trained model over a dataset, dump predictions to JSON
├── requirements.txt
├── docs/
│   ├── Data.md                      # Dataset format, preprocessing, and graph schema
│   └── Evaluation.md                # Evaluation protocol, sample predictions, full results
├── Input_data/
│   ├── consolidated_indoor_dicts.json          # idx_to_label / idx_to_predicate / idx_to_attr
│   ├── consolidated_indoor_top_sg_train.json   # raw training scene graphs
│   └── consolidated_indoor_top_sg_test.json    # raw test scene graphs
└── Output_predictions/
    ├── predictions_train.json       # predicted vs. ground-truth triplets, train split
    └── predictions_test.json        # predicted vs. ground-truth triplets, test split
```

## Files

| File | Description |
|---|---|
| `prepare_data.py` | Preprocesses raw scene-graph JSONs into compact graph format for GNN training |
| `gnn_decompose.py` | Model 1: RGCN-style encoder, BCE link prediction, fixed-threshold decoding |
| `gnn_decompose_new.py` | Model 2: DistMult bilinear scorer, degree prediction head, Top-K decoding |
| `generate_predictions.py` | Loads a trained model and writes predicted + ground-truth triplets for every node to JSON |
| `Input_data/consolidated_indoor_dicts.json` | Vocabulary mappings: `idx_to_label`, `idx_to_predicate`, `idx_to_attr` |
| `Input_data/consolidated_indoor_top_sg_train.json` | Raw training scene graphs |
| `Input_data/consolidated_indoor_top_sg_test.json` | Raw test scene graphs |
| `Output_predictions/` | Generated predictions (train/test), produced by `generate_predictions.py` |

> **Note:** `train_graphs.json` and `test_graphs.json` — the compact graphs produced by `prepare_data.py` — are intermediate/generated files and are excluded via `.gitignore`. Run `prepare_data.py` to regenerate them locally before training.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1 — Preprocess raw data (first time only)

```bash
python prepare_data.py
```

Reads the raw train/test JSONs and writes `train_graphs.json` / `test_graphs.json`.

### 2 — Train & evaluate

```bash
python gnn_decompose.py
```

- Automatically runs `prepare_data.py` if `train_graphs.json` is missing.
- Trains on `train_graphs.json`, evaluates on `test_graphs.json`.
- Saves model weights to `model.pt`.

To train the higher-precision variant instead, run `python gnn_decompose_new.py`.

### 3 — Generate predictions

```bash
python generate_predictions.py --model model.pt --graphs test_graphs.json --dicts Input_data/consolidated_indoor_dicts.json --out predictions.json
```

## Documentation

- [`docs/Data.md`](docs/Data.md) — dataset format, preprocessing steps, and the compact graph schema
- [`docs/Evaluation.md`](docs/Evaluation.md) — evaluation protocol, sample predictions, and full results for both models

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
