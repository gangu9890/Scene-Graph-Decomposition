import json, random, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 178      # idx_to_label ids go up to 177 (+pad 0)
NUM_RELS = 41          # idx_to_predicate ids go up to 40 (+pad 0)
EMB_DIM = 64
HID_DIM = 128

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
            directions, or None if there are no edges.
        """
        out = self.self_loop(h)
        if edge_index is not None and edge_index.numel() > 0:
            s_idx, r_idx, o_idx = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]
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
        self.attr_proj = nn.Linear(emb_dim, hid_dim)

    def encode_nodes(self, nodes):
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
            bwd = fwd[:, [2, 1, 0]]
            edge_index = torch.cat([fwd, bwd], dim=0)
        else:
            edge_index = None
        for layer in self.layers:
            h = layer(h, edge_index)
        return h


class DistMultScorer(nn.Module):
    """Bilinear (DistMult-style) triplet scorer.
    rel_logits[k, r] = <h_query * R_r, h_candidate_k>   (elementwise then dot)
    link_logits[k]   = max_r rel_logits[k, r]            (does *some* relation fit)
    """
    def __init__(self, hid_dim=HID_DIM, num_rels=NUM_RELS):
        super().__init__()
        self.rel_emb = nn.Embedding(num_rels, hid_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, q_emb, cand_emb):
        R = self.rel_emb.weight                      # [num_rels, D]
        qr = q_emb.unsqueeze(0) * R                   # [num_rels, D]  (q (*) R_r)
        rel_logits = qr @ cand_emb.t()                # [num_rels, K]
        rel_logits = rel_logits.t()                   # [K, num_rels]
        link_logits = rel_logits.max(dim=-1).values    # [K]
        return link_logits, rel_logits


class DegreeHead(nn.Module):
    """Predicts a query node's own out-degree (# of triplets it appears in),
    used at inference to decide how many top-scoring candidates to keep."""
    def __init__(self, hid_dim=HID_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2), nn.ReLU(),
            nn.Linear(hid_dim // 2, 1)
        )

    def forward(self, q_emb):
        return self.mlp(q_emb).squeeze(-1)  # predicts log1p(degree)


class GraphDecomposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SceneGraphEncoder()
        self.scorer = DistMultScorer()
        self.degree_head = DegreeHead()

    def encode_graph(self, nodes, rel_edges):
        return self.encoder(nodes, rel_edges)

    def score_query(self, h, attr_edges, query_idx):
        n = h.size(0)
        node_candidates = list(range(n))
        attr_values = sorted({v for _, _, v in attr_edges})
        attr_emb = self.encoder.encode_attr_values(attr_values) if attr_values else torch.empty(0, HID_DIM, device=DEVICE)
        cand_emb = torch.cat([h, attr_emb], dim=0) if attr_values else h
        q_emb = h[query_idx]
        link_logits, rel_logits = self.scorer(q_emb, cand_emb)
        pred_log_degree = self.degree_head(q_emb)
        return link_logits, rel_logits, node_candidates, attr_values, pred_log_degree

    def forward(self, nodes, rel_edges, attr_edges, query_idx):
        h = self.encode_graph(nodes, rel_edges)
        return self.score_query(h, attr_edges, query_idx)


def build_labels(nodes, rel_edges, attr_edges, query_idx, node_candidates, attr_values):
    n = len(node_candidates)
    m = len(attr_values)
    link_y = torch.zeros(n + m, device=DEVICE)
    rel_y = torch.full((n + m,), -100, dtype=torch.long, device=DEVICE)
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
        link_y[query_idx] = 0.0
    true_degree = link_y.sum()
    return link_y, rel_y, true_degree


def decode_triplets_topk(query_idx, node_candidates, attr_values, link_logits, rel_logits,
                          pred_log_degree, margin=0):
    """Rank candidates by score, keep the top predicted-degree (+margin) of them."""
    k = max(0, round(math.expm1(pred_log_degree.item())) + margin)
    k = min(k, link_logits.numel())
    if k == 0:
        return []
    scores = link_logits.cpu()
    rel_pred = rel_logits.argmax(-1).cpu()
    n = len(node_candidates)

    order = torch.argsort(scores, descending=True).tolist()
    order = [i for i in order if i != query_idx][:k]

    triplets = []
    for i in order:
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
          max_queries_per_graph=None, degree_loss_weight=0.1):
    """
    limit=None trains on the FULL file (fix #1). max_queries_per_graph caps
    per-graph query count only if you need to bound compute on very large
    graphs; leave None for full node coverage per epoch (fix from v2).
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

            h = model.encode_graph(nodes, rel_edges)

            query_indices = list(range(n))
            random.shuffle(query_indices)
            if max_queries_per_graph:
                query_indices = query_indices[:max_queries_per_graph]

            graph_loss = 0.0
            for query_idx in query_indices:
                link_logits, rel_logits, node_c, attr_v, pred_log_deg = model.score_query(h, attr_edges, query_idx)
                link_y, rel_y, true_degree = build_labels(nodes, rel_edges, attr_edges, query_idx, node_c, attr_v)

                num_pos = link_y.sum()
                num_neg = link_y.numel() - num_pos
                pos_weight = torch.clamp(num_neg / num_pos.clamp(min=1), max=20.0) if num_pos > 0 else torch.tensor(1.0, device=DEVICE)
                loss_link = F.binary_cross_entropy_with_logits(link_logits, link_y, pos_weight=pos_weight)

                pos_mask = rel_y != -100
                loss_rel = (F.cross_entropy(rel_logits[pos_mask], rel_y[pos_mask])
                            if pos_mask.any() else torch.tensor(0.0, device=DEVICE))

                loss_degree = F.mse_loss(pred_log_deg, torch.log1p(true_degree))

                q_loss = loss_link + loss_rel + degree_loss_weight * loss_degree
                graph_loss = graph_loss + q_loss
                total_loss += q_loss.item()
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
def infer(model, graph, query_idx, margin=0):
    nodes, rel_edges, attr_edges = graph["nodes"], graph["rel_edges"], graph["attr_edges"]
    link_logits, rel_logits, node_c, attr_v, pred_log_deg = model(nodes, rel_edges, attr_edges, query_idx)
    return decode_triplets_topk(query_idx, node_c, attr_v, link_logits, rel_logits, pred_log_deg, margin)


@torch.no_grad()
def evaluate(model, graphs, margin=0, max_graphs=300, all_nodes=True):
    """Edge-level precision/recall using adaptive top-k decoding (fix #3),
    evaluated over every node per graph by default."""
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

            link_logits, rel_logits, node_c, attr_v, pred_log_deg = model.score_query(h, g["attr_edges"], q)
            preds = decode_triplets_topk(q, node_c, attr_v, link_logits, rel_logits, pred_log_deg, margin)

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
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"margin={margin}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  (tp={tp} fp={fp} fn={fn})")
    return prec, rec, f1


def find_best_margin(model, val_graphs, margins=range(-1, 3), max_graphs=300):
    """Sweep the +/- slack applied to the predicted degree and pick the best F1.
    margin=0 trusts the degree head exactly; negative/positive shifts it."""
    best = (0, 0.0, 0.0, 0.0)
    for m in margins:
        prec, rec, f1 = evaluate(model, val_graphs, margin=m, max_graphs=max_graphs)
        if f1 > best[3]:
            best = (m, prec, rec, f1)
    print(f"\nbest margin={best[0]}  precision={best[1]:.3f}  recall={best[2]:.3f}  f1={best[3]:.3f}")
    return best[0]


if __name__ == "__main__":
    here = Path(__file__).parent
    train_graphs_path = here / "train_graphs.json"
    test_graphs_path  = here / "test_graphs.json"

    if not train_graphs_path.exists():
        import subprocess, sys
        print("train_graphs.json not found - running prepare_data.py first...")
        subprocess.run([sys.executable, str(here / "prepare_data.py")], check=True)

    with open(test_graphs_path) as f:
        eval_graphs = list(json.load(f).values())

    model = train(str(train_graphs_path), epochs=20, lr=2e-3, limit=None)

    g = eval_graphs[0]
    print("nodes:", [n[0] for n in g["nodes"]])
    print("ground truth targets for node0:", g["query_targets"].get("0"))

    random.shuffle(eval_graphs)
    half = len(eval_graphs) // 2
    margin_search_graphs, final_report_graphs = eval_graphs[:half], eval_graphs[half:]

    best_m = find_best_margin(model, margin_search_graphs)
    print(f"\nmodel prediction for node0 (margin={best_m}):", infer(model, g, 0, margin=best_m))

    print("\nfinal held-out metrics:")
    evaluate(model, final_report_graphs, margin=best_m)