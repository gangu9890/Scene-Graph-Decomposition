"""
Preprocess consolidated indoor scene-graph JSONs into a compact graph format
+ per-node decomposition targets for GNN training.

Output schema per graph (keyed by image id):
{
  "nodes": [[class_idx, cx, cy, w, h, connected(0/1)], ...]   # index = sg_idx
  "rel_edges":  [[subj_sg_idx, rel_idx, obj_sg_idx], ...]      # deduped, type=2
  "attr_edges": [[subj_sg_idx, attr_idx, value_idx], ...]      # deduped, type=1
}
"""
import json, sys
from pathlib import Path

def load(path):
    with open(path) as f:
        return json.load(f)

def bbox_norm(b):
    # bbox = [x, y, w, h] in pixel space -> keep as-is (scale-free training uses raw ratios)
    x, y, w, h = b
    return [round(x + w/2, 2), round(y + h/2, 2), round(w, 2), round(h, 2)]

def build_graph(g):
    objs = g["objs"]
    nodes = [[o["idx"], *bbox_norm(o["bbox"]), int(o["connected"])] for o in objs]
    rel_seen, attr_seen = set(), set()
    rel_edges, attr_edges = [], []
    for t in g["triplets"].values():
        if t["type"] == 2 and t["subj_sg_idx"] >= 0 and t["obj_sg_idx"] >= 0:
            key = (t["subj_sg_idx"], t["rel"], t["obj_sg_idx"])
            if key not in rel_seen:
                rel_seen.add(key)
                rel_edges.append(list(key))
        elif t["type"] == 1 and t["subj_sg_idx"] >= 0:
            key = (t["subj_sg_idx"], t["rel"], t["obj"])
            if key not in attr_seen:
                attr_seen.add(key)
                attr_edges.append(list(key))
    return {"nodes": nodes, "rel_edges": rel_edges, "attr_edges": attr_edges}

def make_query_targets(graph):
    """For every sg_idx, list the triplets touching it -> matches user's example format."""
    targets = {}
    for s, r, o in graph["rel_edges"]:
        targets.setdefault(s, []).append([s, r, o])
        targets.setdefault(o, []).append([o, r, s])  # reverse direction, own perspective
    for s, r, v in graph["attr_edges"]:
        targets.setdefault(s, []).append([s, r, -v])  # negative marks attribute-value, not a node
    return targets

def process(in_path, out_path):
    raw = load(in_path)
    out = {}
    for img_id, g in raw.items():
        graph = build_graph(g)
        graph["query_targets"] = make_query_targets(graph)
        out[img_id] = graph
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"{in_path} -> {out_path}: {len(out)} graphs")

if __name__ == "__main__":
    here = Path(__file__).parent
    process(str(here / "consolidated_indoor_top_sg_train.json"), str(here / "train_graphs.json"))
    process(str(here / "consolidated_indoor_top_sg_test.json"), str(here / "test_graphs.json"))
