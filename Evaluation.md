# Model Evaluation

This document summarizes the evaluation procedure and performance of the Scene Graph Decomposition models.

The objective of the model is to predict **all triplets incident to a queried node** in a scene graph.

For every query node, the model predicts

- object-object relationships
- object-attribute relationships

These predictions are compared against the ground-truth decomposition.

---

# Evaluation Pipeline

```
Scene Graph
      │
      ▼
Select Query Node
      │
      ▼
Graph Encoder
      │
      ▼
Candidate Scoring
      │
      ▼
Triplet Prediction
      │
      ▼
Compare with Ground Truth
      │
      ▼
Precision • Recall • F1
```

---

# Evaluation Metrics

Each predicted triplet is treated as a binary prediction.

The following metrics are reported over the complete test dataset.

## Precision

Measures prediction correctness.

\[
Precision = \frac{TP}{TP+FP}
\]

Higher precision indicates fewer false positive triplets.

---

## Recall

Measures how many ground-truth triplets are recovered.

\[
Recall = \frac{TP}{TP+FN}
\]

Higher recall indicates fewer missing relationships.

---

## F1 Score

Balances precision and recall.

\[
F1 =
\frac{2PR}{P+R}
\]

---

# Model 1

**Implementation**

`gnn_decompose.py`

### Prediction Strategy

- Relational Graph Convolution Encoder
- Pairwise MLP triplet scorer
- BCE link prediction
- Relation classification
- Fixed probability threshold during inference

---

## Sample Prediction

### Query Node

```
node0
```

### Ground Truth

```
node0
 ├── edge2 → node1
 ├── edge10 → node3
 ├── edge8 → node3
 └── color → black
```

Ground-truth representation

```
[[0,2,1],
 [0,10,3],
 [0,8,3],
 [0,26,-108]]
```

### Predicted

```
node0-edge2-node1
node0-edge2-node3
node0-edge26-attr104
node0-edge35-attr166
```

Observations

- Correctly predicts one object relationship.
- Misses multiple true relations.
- Produces incorrect relation labels.
- Generates false attribute predictions.

---

## Quantitative Results

| Metric | Value |
|---------|------:|
| Precision | **0.501** |
| Recall | **0.746** |
| F1 Score | **0.600** |
| True Positives | 4019 |
| False Positives | 3996 |
| False Negatives | 1367 |

Inference threshold

```
0.40
```

### Interpretation

The threshold-based model achieves **high recall**, successfully recovering most ground-truth triplets.

However, the relatively low precision indicates that many additional incorrect triplets are predicted.

This behaviour is expected from probability thresholding, where lowering the threshold increases recall at the cost of more false positives.

---

# Model 2

**Implementation**

`gnn_decompose_new.py`

### Improvements

Compared with Model 1, this implementation introduces

- DistMult bilinear triplet scorer
- Degree prediction head
- Top-K decoding based on predicted node degree
- Degree regression loss
- Ranking-based inference instead of fixed thresholding

---

## Sample Prediction

### Query Node

```
node0
```

Ground Truth

```
[[0,2,1],
 [0,10,3],
 [0,8,3],
 [0,26,-108]]
```

Predicted

```
node0-edge10-node3
node0-edge2-node1
node0-edge26-attr104
```

Observations

- Predicts two correct relationships.
- Produces fewer false positives.
- Misses one relationship and one attribute.
- Predictions are noticeably cleaner than Model 1.

---

## Quantitative Results

| Metric | Value |
|---------|------:|
| Precision | **0.819** |
| Recall | **0.659** |
| F1 Score | **0.730** |
| True Positives | 3547 |
| False Positives | 785 |
| False Negatives | 1839 |

Margin

```
0
```

---

# Comparison

| Metric | Model 1 | Model 2 |
|---------|---------:|---------:|
| Precision | 0.501 | **0.819** |
| Recall | **0.746** | 0.659 |
| F1 Score | 0.600 | **0.730** |
| False Positives | 3996 | **785** |
| False Negatives | **1367** | 1839 |

---

# Discussion

The two models exhibit different prediction characteristics.

### Model 1

Advantages

- Higher recall
- Recovers more ground-truth triplets

Disadvantages

- Large number of false positives
- Lower precision
- Threshold selection significantly affects performance

---

### Model 2

Advantages

- Significantly higher precision
- Better overall F1 score
- More reliable triplet predictions
- Adaptive prediction count through degree estimation

Disadvantages

- Slightly lower recall
- Misses some valid triplets due to Top-K decoding

---

# Conclusion

The enhanced architecture (`gnn_decompose_new.py`) provides a substantial improvement in overall prediction quality.

Compared with the baseline model:

- **Precision improved by approximately 63%** (0.501 → 0.819)
- **F1 score improved by approximately 22%** (0.600 → 0.730)
- **False positives reduced by approximately 80%** (3996 → 785)

Although recall decreases slightly, the ranking-based inference strategy produces cleaner and more reliable scene graph decompositions, making the improved model better suited for practical deployment.

---

# Future Work

Possible directions for further improvement include

- Attention-based graph neural networks (GAT / Graph Transformer)
- Learnable edge embeddings
- Multi-head relation prediction
- Contrastive node representation learning
- Dynamic negative sampling
- Beam-search triplet decoding
- End-to-end scene graph generation from images
- Evaluation using Recall@K and Mean Reciprocal Rank (MRR)