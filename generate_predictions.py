"""
Run a trained GraphDecomposer over every graph (and every node in each
graph) and write ALL predicted triplets, ALONGSIDE the ground-truth
triplets, to one JSON output file -- so you can eyeball predicted-vs-actual
per node without cross-referencing separately.

Output format:
{
  "<image_id>": {
    "nodes": ["man", "bed", "cat", ...],          # class name per sg_idx, for reference
    "predicted_triplets_raw": ["node0-edge4-node1", "node2-edge26-attr108", ...],
    "predicted_triplets_readable": ["man-in-bed", "cat-color-black", ...],
    "ground_truth_triplets_raw": ["node0-edge4-node1", ...],
    "ground_truth_triplets_readable": ["man-in-bed", ...]
  },
  ...
}

Usage
-----
    python generate_predictions.py \
        --model model.pt \
        --graphs test_graphs.json \
        --dicts consolidated_indoor_dicts.json \
        --out predictions.json \
        --margin 0

Notes
-----
- One forward pass per (graph, query node) pair -- for each node, every
  candidate in that graph is scored simultaneously, then decode_triplets_topk
  keeps the model's own predicted-degree number of top candidates. This
  script just loops that over every node in every graph and collects the
  results; it does not change how triplets are produced (see infer() in
  gnn_decompose_fix123.py for that).
- Uses fix123's architecture (DistMult scorer + degree-based top-k decode).
  If you trained with gnn_decompose_fix1.py instead, see the bottom of this
  file for the one-line swap needed (different scorer/decode signature).
"""
import argparse
import json
import importlib.util
import sys
from pathlib import Path

import torch


def load_module(module_path, name):
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ground_truth_triplets_for_node(graph, query_idx):
    """Build the same 'nodeX-edgeR-nodeY' / 'nodeX-edgeR-attrV' string format
    that decode_triplets_topk produces, but from the real labeled edges --
    so predicted and ground-truth triplets are directly comparable strings."""
    triplets = []
    for s, r, o in graph["rel_edges"]:
        if s == query_idx:
            triplets.append(f"node{s}-edge{r}-node{o}")
        elif o == query_idx:
            # query is the object side of a stored (subj, rel, obj) edge;
            # report it from the query's own perspective, same as predictions do
            triplets.append(f"node{o}-edge{r}-node{s}")
    for s, r, v in graph["attr_edges"]:
        if s == query_idx:
            triplets.append(f"node{s}-edge{r}-attr{v}")
    return triplets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to trained model .pt (state_dict)")
    ap.add_argument("--graphs", required=True, help="path to train_graphs.json or test_graphs.json")
    ap.add_argument("--dicts", required=True, help="path to consolidated_indoor_dicts.json")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--margin", type=int, default=0, help="degree margin from find_best_margin()")
    ap.add_argument("--gnn-module", default="gnn_decompose_fix123.py",
                     help="path to the gnn_decompose_fix123.py file to import")
    ap.add_argument("--label-lookup-module", default="label_lookup.py",
                     help="path to label_lookup.py")
    ap.add_argument("--limit", type=int, default=None, help="optional cap on number of graphs, for a quick test run")
    args = ap.parse_args()

    gd = load_module(args.gnn_module, "gnn_decompose_fix123")
    ll = load_module(args.label_lookup_module, "label_lookup")

    device = gd.DEVICE
    model = gd.GraphDecomposer().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    lut = ll.LabelLookup(args.dicts)

    with open(args.graphs) as f:
        all_graphs = json.load(f)

    items = list(all_graphs.items())
    if args.limit:
        items = items[: args.limit]

    output = {}
    for i, (image_id, g) in enumerate(items):
        n = len(g["nodes"])
        if n < 2:
            output[image_id] = {
                "nodes": [lut.class_name(nd[0]) for nd in g["nodes"]],
                "predicted_triplets_raw": [],
                "predicted_triplets_readable": [],
                "ground_truth_triplets_raw": [],
                "ground_truth_triplets_readable": [],
            }
            continue

        raw_triplets = []
        gt_triplets = []
        for query_idx in range(n):
            raw_triplets.extend(gd.infer(model, g, query_idx, margin=args.margin))
            gt_triplets.extend(ground_truth_triplets_for_node(g, query_idx))

        readable_triplets = lut.translate_triplets(raw_triplets, g)
        readable_gt = lut.translate_triplets(gt_triplets, g)

        output[image_id] = {
            "nodes": [lut.class_name(nd[0]) for nd in g["nodes"]],
            "predicted_triplets_raw": raw_triplets,
            "predicted_triplets_readable": readable_triplets,
            "ground_truth_triplets_raw": gt_triplets,
            "ground_truth_triplets_readable": readable_gt,
        }

        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(items)} graphs...", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"wrote {len(output)} graphs' predictions -> {args.out}")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------
# If your trained model came from gnn_decompose_fix1.py (threshold-based,
# not fix123's top-k), swap two things above:
#   1. --gnn-module should point at gnn_decompose_fix1.py
#   2. replace:
#        raw_triplets.extend(gd.infer(model, g, query_idx, margin=args.margin))
#      with:
#        raw_triplets.extend(gd.infer(model, g, query_idx, threshold=0.4))
#      (fix1's infer() takes `threshold=`, not `margin=`)
# -----------------------------------------------------------------------
