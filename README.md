# Scene Graph Decomposition

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

## Files

| File | Description |
|---|---|
| `prepare_data.py` | Preprocesses raw scene-graph JSONs into compact graph format for GNN training |
| `gnn_decompose.py` | Full GNN model definition, training loop, inference, and evaluation |
| `consolidated_indoor_dicts.json` | Vocabulary mappings: `idx_to_label`, `idx_to_predicate`, `idx_to_attr` |
| `test_graphs.json` | Preprocessed test graphs (1 712 graphs, included for quick eval) |

> **Note:** The large raw files (`consolidated_indoor_top_sg_train.json`, `consolidated_indoor_top_sg_test.json`) and generated `train_graphs.json` are excluded from the repo via `.gitignore`. Run `prepare_data.py` to regenerate them.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1 — Preprocess raw data (first time only)

```bash
python prepare_data.py
```

Reads the raw train/test JSONs and writes `train_graphs.json` / `test_graphs.json` to the same directory.

### 2 — Train & evaluate

```bash
python gnn_decompose.py
```

- Automatically runs `prepare_data.py` if `train_graphs.json` is missing.
- Trains on `train_graphs.json`, evaluates on `test_graphs.json`.
- Saves model weights to `model.pt`.

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
