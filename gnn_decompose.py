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
3. Decoding: at inference, take the query node embedding, score it against
   all candidates in the graph, threshold link_prob, and emit the
   predicted triplet strings.

Changelog
---------
v2 (recall fix): every node in a graph is used as a query per epoch (not one
random node), pos_weight added to the link loss for class imbalance, and a
validation-based threshold search replaces the fixed 0.4 cutoff.

v3 (GPU speed on Kaggle): the previous RelGraphConv looped over edges in
plain Python (`for s, r, o in edges: ...`), which cannot be parallelized by
a GPU -- each edge was a separate tiny op dispatched from Python, so a GPU
run was actually *slower* than CPU due to kernel-launch overhead with no
parallel work to fill it. This version:
  - Vectorizes message passing: edges are stored as a single tensor, all
    per-edge relation matrices are built in one batched einsum, and
    aggregation uses index_add_ / scatter instead of a Python loop.
  - Adds a DEVICE switch (cuda if available) with model + all tensors moved
    onto it, so the vectorized ops actually run on the GPU.
  - Still trains one graph at a time (graphs vary in node count, so true
    batching across graphs would need padding/block-diagonal adjacency --
    see the note at the bottom of this file if you want that next), but
    each graph's own message passing and scoring is now fully vectorized,
    which is what lets the GPU help at all.
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
MAX_POS_WEIGHT = 20.0   # clamp to keep BCE stable on graphs with very few positives

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[gnn_decompose] using device: {DEVICE}")


class RelGraphConv(nn.Module):
    """One relational message-passing layer over object-object edges.
    Vectorized: no Python loop over individual edges."""
    def __init__(self, dim, num_rels, num_bases=8):
        super().__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bases = nn.Parameter(torch.randn(num_bases, dim, dim) * 0.05)
        self.rel_coeff = nn.Parameter(torch.randn(num_rels, num_bases) * 0.1)
        self.self_loop = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index):
        """
        h: [N, dim]
        edge_index: LongTensor [E, 3] of (subj, rel, obj), already both
            directions, or None/empty if there are no edges.
        """
        out = self.self_loop(h)
        if edge_index is not None and edge_index.numel() > 0:
            s_idx, r_idx, o_idx = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
            # combine bases with per-relation coefficients for every edge at once
            # rel_coeff[r_idx]: [E, num_bases] ; bases: [num_bases, dim, dim]
            W_e = torch.einsum('eb,bij->eij', self.rel_coeff[r_idx], self.bases)  # [E, dim, dim]
            msgs = torch.einsum('ei,eij->ej', h[s_idx], W_e)                     # [E, dim]

            agg = torch.zeros_like(h)
            agg.index_add_(0, o_idx, msgs)
            deg = torch.zeros(h.size(0), device=h.device).index_add_(
                0, o_idx, torch.ones(o_idx.size(0), device=h.device)
            ).clamp(min=1e-6)
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
        cls = torch.tensor([n[0] for n in nodes], dtype=torch.long, device=DEVICE)
        bbox = torch.tensor([n[1:5] for n in nodes], dtype=torch.float32, device=DEVICE) / 500.0
        h0 = torch.cat([self.class_emb(cls), self.bbox_mlp(bbox)], dim=-1)
        return self.in_proj(h0)

    def encode_attr_values(self, value_ids):
        cls = torch.tensor(value_ids, dtype=torch.long, device=DEVICE)
        return self.attr_proj(self.class_emb(cls))

    def forward(self, nodes, rel_edges):
        h = self.encode_nodes(nodes)
        if rel_edges:
            fwd = torch.tensor(rel_edges, dtype=torch.long, device=DEVICE)
            bwd = fwd[:, [2, 1, 0]]  # (o, r, s) reverse direction
            edge_index = torch.cat([fwd, bwd], dim=0)
        else:
            edge_index = None
        for layer in self.layers:
            h = layer(h, edge_index)
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

    def encode_graph(self, nodes, rel_edges):
        """Run the encoder once; reuse the result across all queries for this graph."""
        return self.encoder(nodes, rel_edges)

    def score_query(self, h, attr_edges, query_idx):
        n = h.size(0)
        node_candidates = list(range(n))
        attr_values = sorted({v for _, _, v in attr_edges})
        attr_emb = self.encoder.encode_attr_values(attr_values) if attr_values else torch.empty(0, HID_DIM, device=DEVICE)
        cand_emb = torch.cat([h, attr_emb], dim=0) if attr_values else h
        q_emb = h[query_idx]
        link_logits, rel_logits = self.scorer(q_emb, cand_emb)
        return link_logits, rel_logits, node_candidates, attr_values

    def forward(self, nodes, rel_edges, attr_edges, query_idx):
        h = self.encode_graph(nodes, rel_edges)
        return self.score_query(h, attr_edges, query_idx)


def build_labels(nodes, rel_edges, attr_edges, query_idx, node_candidates, attr_values):
    n = len(node_candidates)
    m = len(attr_values)
    link_y = torch.zeros(n + m, device=DEVICE)
    rel_y = torch.full((n + m,), -100, dtype=torch.long, device=DEVICE)  # ignore_index for non-edges
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
    probs = torch.sigmoid(link_logits).cpu()
    rel_pred = rel_logits.argmax(-1).cpu()
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


def train(train_path, epochs=20, lr=1e-3, limit=None, save_path="model.pt",
          max_queries_per_graph=None):
    """
    max_queries_per_graph: cap on how many query nodes to train on per graph
    per epoch (None = use every node). Only needed if you have very large
    graphs and want to bound per-epoch compute; leave None for full coverage.
    """
    graphs = load_graphs(train_path, limit=limit)
    model = GraphDecomposer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        random.shuffle(graphs)
        total_loss, steps = 0.0, 0
        for g in graphs:
            nodes, rel_edges, attr_edges = g["nodes"], g["rel_edges"], g["attr_edges"]
            n = len(nodes)
            if n < 2:
                continue

            # encode the graph once, reuse for every query node in it.
            # All per-query losses below share this one computation graph, so
            # they must be summed and backpropagated in a single .backward()
            # call per graph -- calling .backward() once per query would free
            # the shared encoder graph after the first call and crash on the
            # second (RuntimeError: backward through the graph a second time).
            h = model.encode_graph(nodes, rel_edges)

            query_indices = list(range(n))
            random.shuffle(query_indices)
            if max_queries_per_graph:
                query_indices = query_indices[:max_queries_per_graph]

            graph_loss = 0.0
            for query_idx in query_indices:
                link_logits, rel_logits, node_c, attr_v = model.score_query(h, attr_edges, query_idx)
                link_y, rel_y = build_labels(nodes, rel_edges, attr_edges, query_idx, node_c, attr_v)

                num_pos = link_y.sum()
                num_neg = link_y.numel() - num_pos
                pos_weight = torch.clamp(num_neg / num_pos.clamp(min=1), max=MAX_POS_WEIGHT) if num_pos > 0 else torch.tensor(1.0, device=DEVICE)
                loss_link = F.binary_cross_entropy_with_logits(link_logits, link_y, pos_weight=pos_weight)

                pos_mask = rel_y != -100
                loss_rel = (F.cross_entropy(rel_logits[pos_mask], rel_y[pos_mask])
                            if pos_mask.any() else torch.tensor(0.0, device=DEVICE))
                graph_loss = graph_loss + loss_link + loss_rel
                total_loss += (loss_link.item() + (loss_rel.item() if pos_mask.any() else 0.0))
                steps += 1

            if len(query_indices) > 0:
                opt.zero_grad()
                (graph_loss / len(query_indices)).backward()
                opt.step()
        print(f"epoch {epoch+1}/{epochs}  avg_loss={total_loss/max(steps,1):.4f}  queries={steps}  graphs={len(graphs)}")
    torch.save(model.state_dict(), save_path)
    print(f"saved -> {save_path}")
    return model


@torch.no_grad()
def infer(model, graph, query_idx, threshold=0.5):
    nodes, rel_edges, attr_edges = graph["nodes"], graph["rel_edges"], graph["attr_edges"]
    link_logits, rel_logits, node_c, attr_v = model(nodes, rel_edges, attr_edges, query_idx)
    return decode_triplets(query_idx, node_c, attr_v, link_logits, rel_logits, threshold)


@torch.no_grad()
def _collect_predictions(model, graphs, threshold, max_graphs=None, all_nodes=True):
    """Shared scoring loop used by both evaluate() and find_best_threshold().
    Returns tp/fp/fn counts at the given threshold, evaluated over every node
    in every graph (not just one random node) for a stable estimate."""
    tp = fp = fn = 0
    use_graphs = graphs[:max_graphs] if max_graphs else graphs
    for g in use_graphs:
        n = len(g["nodes"])
        if n < 2:
            continue
        query_range = range(n) if all_nodes else [random.randrange(n)]
        h = model.encode_graph(g["nodes"], g["rel_edges"])
        for q in query_range:
            gt = set()
            for s, r, o in g["rel_edges"]:
                if s == q: gt.add((r, o, "n"))
                if o == q: gt.add((r, s, "n"))
            for s, r, v in g["attr_edges"]:
                if s == q: gt.add((r, v, "a"))

            link_logits, rel_logits, node_c, attr_v = model.score_query(h, g["attr_edges"], q)
            preds = decode_triplets(q, node_c, attr_v, link_logits, rel_logits, threshold)

            pred = set()
            for p in preds:
                parts = p.split("-")
                r = int(parts[1].replace("edge", ""))
                tgt = parts[2]
                if tgt.startswith("node"):
                    pred.add((r, int(tgt.replace("node", "")), "n"))
                else:
                    pred.add((r, int(tgt.replace("attr", "")), "a"))
            tp += len(gt & pred); fp += len(pred - gt); fn += len(gt - pred)
    return tp, fp, fn


def evaluate(model, graphs, threshold=0.4, max_graphs=300, all_nodes=True):
    """Edge-level precision/recall, evaluated over every node per graph by default
    (set all_nodes=False to reproduce the older one-random-node-per-graph metric)."""
    tp, fp, fn = _collect_predictions(model, graphs, threshold, max_graphs, all_nodes)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"threshold={threshold:.2f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  (tp={tp} fp={fp} fn={fn})")
    return prec, rec, f1


def find_best_threshold(model, val_graphs, thresholds=None, max_graphs=300):
    """Sweep thresholds on a held-out split and return the one maximizing F1."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in torch.arange(0.1, 0.91, 0.05).tolist()]
    best = (0.5, 0.0, 0.0, 0.0)  # threshold, precision, recall, f1
    for t in thresholds:
        prec, rec, f1 = evaluate(model, val_graphs, threshold=t, max_graphs=max_graphs)
        if f1 > best[3]:
            best = (t, prec, rec, f1)
    print(f"\nbest threshold={best[0]:.2f}  precision={best[1]:.3f}  recall={best[2]:.3f}  f1={best[3]:.3f}")
    return best[0]


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

    # Split eval_graphs into a threshold-search half and a final-report half
    # so the reported metric isn't tuned on the same data it's measured on.
    random.shuffle(eval_graphs)
    half = len(eval_graphs) // 2
    thresh_search_graphs, final_report_graphs = eval_graphs[:half], eval_graphs[half:]

    best_t = find_best_threshold(model, thresh_search_graphs)
    print(f"\nmodel prediction for node0 (best threshold={best_t:.2f}):",
          infer(model, g, 0, threshold=best_t))

    print("\nfinal held-out metrics:")
    evaluate(model, final_report_graphs, threshold=best_t)

# -----------------------------------------------------------------------
# Further speedup (optional, more work): true cross-graph mini-batching.
# Right now each graph is still processed one at a time -- the GPU wins
# here come from vectorizing *within* a graph's message passing, not from
# batching many graphs into one matmul. To batch across graphs too, you'd
# merge several graphs' node/edge tensors into one block-diagonal graph per
# step (offset each graph's node indices by a running total, concatenate
# their edge_index tensors, and track per-graph node-count boundaries to
# split queries back out afterward). That's the same trick PyTorch
# Geometric's DataLoader does automatically -- worth adopting if per-graph
# GPU utilization still looks low after this change (check with `nvidia-smi`
# or Kaggle's GPU usage graph while training).
# -----------------------------------------------------------------------
