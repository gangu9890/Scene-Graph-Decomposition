"""
GNN-based scene-graph decomposition.

Given a graph and a query node, predict all triplets (edges + attributes)
incident to that node -- i.e. reconstruct:
    node1-edge1-node2, node1-edge3-node3, ...

Architecture
------------
1. Encoder: each object node gets an embedding from (class_idx, bbox) and is
   refined by a few layers of relational message passing (RGCN-style: one
   weight matrix per predicate bucket, summed with a self-loop) over the
   object-object relation edges. Attribute values are embedded too and
   pulled in as extra "neighbor" messages so node embeddings absorb their
   own attributes.
2. Query head: for a query node, concat its embedding with every other
   node's/attribute value's embedding, pass through an MLP scorer that
   jointly predicts:
     - link_prob : does an edge/attr exist between query and this candidate
     - rel_logits: which predicate/attribute-type it is (multi-class)
   Trained with negative sampling against all non-adjacent nodes/attrs
   present in the same graph.
3. Decoding: at inference, take the query node embedding, score it against
   all candidates in the graph, threshold link_prob, and emit the
   predicted triplet strings.

This keeps the model graph-size-agnostic and reuses the same tiny MLP for
every node, so it scales to any node count without retraining.
"""
import json, random, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 178      # idx_to_label ids go up to 177 (+pad 0)
NUM_RELS = 41          # idx_to_predicate ids go up to 40 (+pad 0)
EMB_DIM = 64
HID_DIM = 128


class RelGraphConv(nn.Module):
    """One relational message-passing layer over object-object edges."""
    def __init__(self, dim, num_rels, num_bases=8):
        super().__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bases = nn.Parameter(torch.randn(num_bases, dim, dim) * 0.05)
        self.rel_coeff = nn.Parameter(torch.randn(num_rels, num_bases) * 0.1)
        self.self_loop = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def rel_weight(self, r):
        # combine bases with per-relation coefficients -> [dim, dim]
        return torch.einsum('b,bij->ij', self.rel_coeff[r], self.bases)

    def forward(self, h, edges):
        # h: [N, dim]; edges: list of (s, r, o) int tuples (already both directions)
        out = self.self_loop(h)
        if edges:
            agg = torch.zeros_like(h)
            deg = torch.zeros(h.size(0), device=h.device) + 1e-6
            for s, r, o in edges:
                W = self.rel_weight(r)
                agg[o] += h[s] @ W
                deg[o] += 1
            out = out + agg / deg.unsqueeze(-1)
        return self.norm(F.relu(out))


class SceneGraphEncoder(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, hid_dim=HID_DIM, num_layers=2, num_rels=NUM_RELS):
        super().__init__()
        self.class_emb = nn.Embedding(NUM_CLASSES, emb_dim, padding_idx=0)
        self.bbox_mlp = nn.Sequential(nn.Linear(4, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.in_proj = nn.Linear(emb_dim * 2, hid_dim)
        self.layers = nn.ModuleList([RelGraphConv(hid_dim, num_rels) for _ in range(num_layers)])
        # attribute-value vocabulary shares the same class embedding space (ids 101-177)
        self.attr_proj = nn.Linear(emb_dim, hid_dim)

    def encode_nodes(self, nodes):
        # nodes: [N,6] -> class_idx, cx, cy, w, h, connected
        cls = torch.tensor([n[0] for n in nodes], dtype=torch.long)
        bbox = torch.tensor([n[1:5] for n in nodes], dtype=torch.float32) / 500.0
        h0 = torch.cat([self.class_emb(cls), self.bbox_mlp(bbox)], dim=-1)
        return self.in_proj(h0)

    def encode_attr_values(self, value_ids):
        cls = torch.tensor(value_ids, dtype=torch.long)
        return self.attr_proj(self.class_emb(cls))

    def forward(self, nodes, rel_edges):
        h = self.encode_nodes(nodes)
        bidir = [(s, r, o) for s, r, o in rel_edges] + [(o, r, s) for s, r, o in rel_edges]
        for layer in self.layers:
            h = layer(h, bidir)
        return h


class TripletScorer(nn.Module):
    """Scores (query_node, candidate) pairs -> link prob + relation logits."""
    def __init__(self, hid_dim=HID_DIM, num_rels=NUM_RELS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
        )
        self.link_head = nn.Linear(hid_dim, 1)
        self.rel_head = nn.Linear(hid_dim, num_rels)

    def forward(self, q_emb, cand_emb):
        # q_emb: [D]; cand_emb: [K,D] -> broadcast
        x = torch.cat([q_emb.unsqueeze(0).expand_as(cand_emb), cand_emb], dim=-1)
        feat = self.mlp(x)
        return self.link_head(feat).squeeze(-1), self.rel_head(feat)


class GraphDecomposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SceneGraphEncoder()
        self.scorer = TripletScorer()

    def forward(self, nodes, rel_edges, attr_edges, query_idx):
        h = self.encoder(nodes, rel_edges)  # [N, D]
        n = len(nodes)
        node_candidates = list(range(n))
        attr_values = sorted({v for _, _, v in attr_edges})
        attr_emb = self.encoder.encode_attr_values(attr_values) if attr_values else torch.empty(0, HID_DIM)
        cand_emb = torch.cat([h, attr_emb], dim=0) if attr_values else h
        q_emb = h[query_idx]
        link_logits, rel_logits = self.scorer(q_emb, cand_emb)
        return link_logits, rel_logits, node_candidates, attr_values


def build_labels(nodes, rel_edges, attr_edges, query_idx, node_candidates, attr_values):
    n = len(node_candidates)
    m = len(attr_values)
    link_y = torch.zeros(n + m)
    rel_y = torch.full((n + m,), -100, dtype=torch.long)  # ignore_index for non-edges
    attr_pos = {v: n + i for i, v in enumerate(attr_values)}
    for s, r, o in rel_edges:
        if s == query_idx:
            link_y[o] = 1.0; rel_y[o] = r
        if o == query_idx:
            link_y[s] = 1.0; rel_y[s] = r
    for s, r, v in attr_edges:
        if s == query_idx:
            idx = attr_pos[v]
            link_y[idx] = 1.0; rel_y[idx] = r
    if query_idx < n:
        link_y[query_idx] = 0.0  # no self-loop target
    return link_y, rel_y


def decode_triplets(query_idx, node_candidates, attr_values, link_logits, rel_logits, threshold=0.5):
    """Turn model scores back into human-readable triplet strings."""
    probs = torch.sigmoid(link_logits)
    rel_pred = rel_logits.argmax(-1)
    n = len(node_candidates)
    triplets = []
    for i in range(len(probs)):
        if i == query_idx:
            continue
        if probs[i] >= threshold:
            r = rel_pred[i].item()
            if i < n:
                triplets.append(f"node{query_idx}-edge{r}-node{i}")
            else:
                v = attr_values[i - n]
                triplets.append(f"node{query_idx}-edge{r}-attr{v}")
    return triplets


# ----------------------------- training loop -----------------------------

def load_graphs(path, limit=None):
    with open(path) as f:
        d = json.load(f)
    items = list(d.values())
    if limit:
        items = items[:limit]
    return [g for g in items if len(g["nodes"]) >= 2]


def train(train_path, epochs=20, lr=1e-3, limit=2000, save_path="model.pt"):
    graphs = load_graphs(train_path, limit=limit)
    model = GraphDecomposer()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        random.shuffle(graphs)
        total_loss, steps = 0.0, 0
        for g in graphs:
            nodes, rel_edges, attr_edges = g["nodes"], g["rel_edges"], g["attr_edges"]
            n = len(nodes)
            if n < 2:
                continue
            query_idx = random.randrange(n)
            link_logits, rel_logits, node_c, attr_v = model(nodes, rel_edges, attr_edges, query_idx)
            link_y, rel_y = build_labels(nodes, rel_edges, attr_edges, query_idx, node_c, attr_v)
            loss_link = F.binary_cross_entropy_with_logits(link_logits, link_y)
            pos_mask = rel_y != -100
            loss_rel = (F.cross_entropy(rel_logits[pos_mask], rel_y[pos_mask])
                        if pos_mask.any() else torch.tensor(0.0))
            loss = loss_link + loss_rel
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); steps += 1
        print(f"epoch {epoch+1}/{epochs}  avg_loss={total_loss/max(steps,1):.4f}  graphs={steps}")
    torch.save(model.state_dict(), save_path)
    print(f"saved -> {save_path}")
    return model


@torch.no_grad()
def infer(model, graph, query_idx, threshold=0.5):
    nodes, rel_edges, attr_edges = graph["nodes"], graph["rel_edges"], graph["attr_edges"]
    link_logits, rel_logits, node_c, attr_v = model(nodes, rel_edges, attr_edges, query_idx)
    return decode_triplets(query_idx, node_c, attr_v, link_logits, rel_logits, threshold)


@torch.no_grad()
def evaluate(model, graphs, threshold=0.4, max_graphs=300):
    """Edge-level precision/recall over a random query node per graph."""
    tp = fp = fn = 0
    for g in graphs[:max_graphs]:
        n = len(g["nodes"])
        if n < 2:
            continue
        q = random.randrange(n)
        gt = set()
        for s, r, o in g["rel_edges"]:
            if s == q: gt.add((r, o, "n"))
            if o == q: gt.add((r, s, "n"))
        for s, r, v in g["attr_edges"]:
            if s == q: gt.add((r, v, "a"))
        pred_strs = infer(model, g, q, threshold)
        pred = set()
        for p in pred_strs:
            parts = p.split("-")
            r = int(parts[1].replace("edge", ""))
            tgt = parts[2]
            if tgt.startswith("node"):
                pred.add((r, int(tgt.replace("node", "")), "n"))
            else:
                pred.add((r, int(tgt.replace("attr", "")), "a"))
        tp += len(gt & pred); fp += len(pred - gt); fn += len(gt - pred)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print(f"precision={prec:.3f}  recall={rec:.3f}  (tp={tp} fp={fp} fn={fn})")


if __name__ == "__main__":
    here = Path(__file__).parent
    train_graphs_path = here / "train_graphs.json"
    test_graphs_path  = here / "test_graphs.json"

    # Auto-generate train_graphs.json if prepare_data.py hasn't been run yet.
    if not train_graphs_path.exists():
        import subprocess, sys
        print("train_graphs.json not found - running prepare_data.py first...")
        subprocess.run([sys.executable, str(here / "prepare_data.py")], check=True)

    # Use proper train / test splits (produced by prepare_data.py).
    with open(test_graphs_path) as f:
        eval_graphs = list(json.load(f).values())

    model = train(str(train_graphs_path), epochs=20, lr=2e-3, limit=None)

    g = eval_graphs[0]
    print("nodes:", [n[0] for n in g["nodes"]])
    print("ground truth targets for node0:", g["query_targets"].get("0"))
    print("model prediction for node0:", infer(model, g, 0, threshold=0.4))

    evaluate(model, eval_graphs)
