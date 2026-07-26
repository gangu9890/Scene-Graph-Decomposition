# Dataset Documentation

This project operates on **scene graph annotations** extracted from indoor images. Each image is represented as a graph consisting of objects (nodes), relationships between objects (edges), and object attributes.

The raw dataset is converted into a compact graph representation using `prepare_data.py`, which is then used for training and evaluation of the Scene Graph Decomposition model.

---

# Dataset Files

| File | Description |
|------|-------------|
| `consolidated_indoor_top_sg_train.json` | Raw training scene graphs |
| `consolidated_indoor_top_sg_test.json` | Raw testing scene graphs |
| `consolidated_indoor_dicts.json` | Vocabulary mappings for object classes, predicates, and attributes |
| `prepare_data.py` | Converts raw scene graphs into compact graph representation |
| `train_graphs.json` | Processed training graphs (generated) |
| `test_graphs.json` | Processed testing graphs (generated) |

Large raw files are excluded from the repository and can be regenerated using the preprocessing script.

---

# Raw Scene Graph Format

Each image is stored using its image id as the key.

Example:

```json
{
  "2400861": {
    "objs": [...],
    "triplets": {...}
  }
}
```

Each image consists of two components:

- **Objects**
- **Triplets**

---

# Objects

Objects represent every detected entity in the image.

Example:

```json
{
    "idx": 100,
    "bbox": [223,188,49,63],
    "connected": true
}
```

## Fields

| Field | Description |
|--------|-------------|
| `idx` | Integer object category id |
| `bbox` | Bounding box `[x, y, width, height]` |
| `connected` | Whether the object participates in at least one scene graph triplet |

---

# Bounding Boxes

Bounding boxes are stored in pixel coordinates.

```
[x,
 y,
 width,
 height]
```

During preprocessing they are converted into

```
[cx,
 cy,
 width,
 height]
```

where

```
cx = x + width / 2
cy = y + height / 2
```

This provides a more convenient representation for the GNN while preserving object size information.

---

# Triplets

Each triplet represents either

- an object-object relationship
- or an object attribute.

Example:

```json
{
    "type":2,
    "subj":100,
    "rel":2,
    "obj":59,
    "subj_sg_idx":0,
    "obj_sg_idx":1
}
```

---

## Relationship Triplets

Relationship triplets connect two objects.

Example

```
(radiator)
      |
 to the right of
      |
(bowl)
```

Stored as

```json
{
    "type":2,
    "subj":100,
    "rel":2,
    "obj":59
}
```

where

| Field | Meaning |
|--------|---------|
| `subj` | subject object id |
| `rel` | relationship id |
| `obj` | object id |

---

## Attribute Triplets

Attributes describe properties of a single object.

Example

```
chair

↓

color

↓

white
```

Stored as

```json
{
    "type":1,
    "subj":59,
    "rel":26,
    "obj":104
}
```

where

```
color -> white
```

---

# Vocabulary Mapping

The file

```
consolidated_indoor_dicts.json
```

contains all vocabulary mappings used throughout the project.

It consists of

```
idx_to_label
idx_to_predicate
idx_to_attr
```

---

## Object Labels

Example

```
1 → man
2 → chair
3 → table
...
```

---

## Predicates

Example

```
1 → on
2 → to the right of
3 → to the left of
4 → in
...
```

---

## Attributes

Instead of storing strings directly, attributes are represented by indices.

Example

```
26 → color
27 → material
28 → pose
29 → activity
31 → size
```

Each attribute has a corresponding value range.

Example

```
color

101 → brown

104 → white

108 → black
```

---

# Preprocessing

The script

```
prepare_data.py
```

converts the raw scene graph annotations into a compact graph representation suitable for GNN training.

It performs the following operations.

---

## 1. Convert Bounding Boxes

```
[x,y,w,h]

↓

[cx,cy,w,h]
```

---

## 2. Build Node List

Each object becomes a node.

Node format

```
[class_idx,
 center_x,
 center_y,
 width,
 height,
 connected]
```

Example

```
[45,324.5,245.5,99,489,1]
```

---

## 3. Extract Relationship Edges

Relationship triplets become

```
[subj_node,
 relation,
 obj_node]
```

Example

```
[3,5,7]
```

meaning

```
node3 --wearing--> node7
```

Duplicate edges are removed during preprocessing.

---

## 4. Extract Attribute Edges

Attribute triplets become

```
[subj_node,
 attribute,
 value]
```

Example

```
[2,26,104]
```

meaning

```
node2

↓

color

↓

white
```

Duplicate attribute edges are also removed.

---

## 5. Build Query Targets

The decomposition task predicts all triplets incident to a queried node.

For every node, preprocessing generates

```
query_targets
```

which contains every relationship and attribute connected to that node.

Example

```json
{
    "2":[
        [2,5,1],
        [2,26,-104]
    ]
}
```

Negative values indicate attribute-value ids rather than graph nodes.

```
-104

↓

attribute value

↓

white
```

---

# Processed Graph Format

Each processed graph has the following structure.

```json
{
    "nodes":[...],
    "rel_edges":[...],
    "attr_edges":[...],
    "query_targets":{...}
}
```

---

## Nodes

```
[class,
 cx,
 cy,
 width,
 height,
 connected]
```

---

## Relationship Edges

```
[
    subject_node,
    relation,
    object_node
]
```

---

## Attribute Edges

```
[
    subject_node,
    attribute,
    value
]
```

---

## Query Targets

Maps every node to the set of ground-truth triplets that should be predicted during decomposition.

This enables node-centric supervision during training.

---

# Training Pipeline

```
Raw Scene Graphs
        │
        ▼
prepare_data.py
        │
        ▼
Processed Graphs
        │
        ▼
Graph Encoder
        │
        ▼
Query Node Selection
        │
        ▼
Predict Incident Triplets
```

---

# Why Preprocessing?

The raw annotations are optimized for storage and visualization.

The GNN requires a graph-centric representation with

- compact node features
- deduplicated edges
- explicit relationship lists
- explicit attribute lists
- node-level supervision

`prepare_data.py` performs this conversion and generates the graph representation used throughout training and evaluation.