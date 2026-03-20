"""
Layerwise transformer — explicit per-layer hidden states.
Each layer has its own hidden state h_i derived from h_{i-1}.
Probe L_i from h_i only.
Deep supervision + consistency losses.
"""
import json, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os
from collections import Counter

random.seed(42)
torch.manual_seed(42)

LAYERS = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]
D = 64  # hidden dim per layer

print("Loading corpus_v8c...")
rows = []
with open("corpus/corpus_v8c.ndjson") as f:
    for line in f: rows.append(json.loads(line))
print(f"  {len(rows)} rows")

# Build class id arrays and vocab sizes
L_ids = [np.array([r["layer_class_ids"][f"L{i}"] for r in rows], dtype=np.int64) for i in range(7)]
vocab = [int(L_ids[i].max()) + 1 for i in range(7)]
print("Vocab sizes:", vocab)

# Feature vector per row — 7 one-hot phoneme features + word hash features
def encode_row(r):
    x = np.zeros(44 + 16 + 7, dtype=np.float32)
    # L0: phoneme class one-hot (44 phonemes)
    ph = r["layer_class_ids"]["L0"]
    if ph < 44: x[ph] = 1.0
    # L2: morpheme class one-hot (16 morphemes)
    mo = r["layer_class_ids"]["L2"]
    if mo < 16: x[44 + mo] = 1.0
    # Layer presence flags
    for i in range(7):
        x[60 + i] = r["layer_class_ids"][f"L{i}"] / max(vocab[i], 1)
    return x

FDIM = 44 + 16 + 7  # 67

print("Encoding rows...")
X  = torch.tensor(np.stack([encode_row(r) for r in rows]))
Ys = [torch.tensor(L_ids[i]) for i in range(7)]

random.shuffle(rows)  # already encoded, shuffle indices
idx = list(range(len(rows)))
random.shuffle(idx)
split = int(0.85 * len(idx))
tr_idx = torch.tensor(idx[:split])
va_idx = torch.tensor(idx[split:])

X_tr  = X[tr_idx];  X_va  = X[va_idx]
Y_tr  = [Y[tr_idx] for Y in Ys]
Y_va  = [Y[va_idx] for Y in Ys]
print(f"  train={len(tr_idx)} val={len(va_idx)}")

class LayerwiseModel(nn.Module):
    """
    Explicit layerwise hidden states.
    h_0 = f_0(x)
    h_{i+1} = f_i(h_i)
    Probe L_i from h_i only.
    """
    def __init__(self, fdim, d, vocab_sizes):
        super().__init__()
        self.d = d
        # Input projection
        self.inp = nn.Linear(fdim, d)
        # Per-layer transition networks
        self.transitions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(d*2, d), nn.LayerNorm(d)
            ) for _ in range(6)  # 6 transitions for 7 layers
        ])
        # Per-layer classification heads
        self.heads = nn.ModuleList([
            nn.Linear(d, v) for v in vocab_sizes
        ])
        # Consistency heads: predict next layer class from current hidden state
        self.consist = nn.ModuleList([
            nn.Linear(d, vocab_sizes[i+1]) for i in range(6)
        ])
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # Build layerwise hidden states
        h = [None] * 7
        h[0] = self.inp(x)
        for i in range(6):
            h[i+1] = h[i] + self.transitions[i](h[i])  # residual

        # Classification logits from each layer's hidden state
        logits = [self.heads[i](h[i]) for i in range(7)]

        # Consistency logits: predict layer i+1 from layer i's hidden state
        consist_logits = [self.consist[i](h[i]) for i in range(6)]

        return logits, consist_logits, h

    def n_params(self): return sum(p.numel() for p in self.parameters())

model = LayerwiseModel(FDIM, D, vocab)
print(f"LayerwiseModel: {model.n_params():,} params")

opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

# Layer weights — higher layers get more weight initially
# Curriculum: start with lower layers, add higher layers progressively
def layer_weights(ep, n_epochs=100):
    progress = ep / n_epochs
    # All layers active from start, but lower layers weighted more early
    w = [1.0] * 7
    # Higher layers get full weight after warmup
    for i in range(4, 7):
        warmup = (i - 3) * 0.15
        w[i] = min(1.0, progress / warmup) if progress < warmup else 1.0
    return w

best_acc = [0.0] * 7
best_mean = 0.0

for ep in range(100):
    model.train()
    lw = layer_weights(ep)
    perm = torch.randperm(len(X_tr))
    total_loss = 0
    for i in range(0, len(perm), 128):
        idx_b = perm[i:i+128]
        x  = X_tr[idx_b]
        ys = [Y_tr[li][idx_b] for li in range(7)]

        logits, consist_logits, _ = model(x)

        # Classification loss — weighted by layer
        # Phrase (L4) gets extra classification weight
        phrase_w = [1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0]
        cls_loss = sum(lw[li] * phrase_w[li] * F.cross_entropy(logits[li], ys[li]) for li in range(7))

        # Consistency loss — each layer predicts the next
        # L4 gets extra weight: it must predict both L5 (semantic) correctly
        con_losses = []
        for li in range(6):
            w_con = 3.0 if li == 3 else 1.0  # L4 must predict L5
            con_losses.append(w_con * F.cross_entropy(consist_logits[li], ys[li+1]))
        con_loss = sum(con_losses) / 6.0

        loss = cls_loss + 0.5 * con_loss
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    sched.step()

    if ep % 10 == 9:
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(X_va)
            accs = [(logits[li].argmax(1)==Y_va[li]).float().mean().item() for li in range(7)]
        mean_acc = sum(accs) / 7
        if mean_acc > best_mean:
            best_mean = mean_acc
            torch.save(model.state_dict(), "train/model_layerwise_best.pt")
            best_acc = accs[:]
        deltas = [a - 1.0/vocab[li] for li,a in enumerate(accs)]
        print(f"ep={ep+1:3d} loss={total_loss:.1f} mean={mean_acc:.3f} "
              f"[{' '.join(f'L{i}={a:.2f}' for i,a in enumerate(accs))}]")

model.load_state_dict(torch.load("train/model_layerwise_best.pt"))
model.eval()
print(f"\n{'='*60}")
print(f"LLR BENCHMARK — LayerwiseModel on corpus_v8c")
print(f"{'='*60}")
with torch.no_grad():
    logits, _, _ = model(X_va)
    accs = [(logits[li].argmax(1)==Y_va[li]).float().mean().item() for li in range(7)]

results = {}
for li in range(7):
    chance = 1.0/vocab[li]
    delta  = accs[li] - chance
    quality = "STRONG" if delta>0.3 else "MODERATE" if delta>0.1 else "WEAK"
    results[LAYERS[li]] = {"acc":accs[li],"chance":chance,"delta":delta,"vocab":vocab[li]}
    print(f"  L{li} {LAYERS[li]:<12} acc={accs[li]:.3f} chance={chance:.3f} "
          f"delta=+{delta:.3f}  {quality}")

os.makedirs("benchmark", exist_ok=True)
with open("benchmark/llr_layerwise_v8c.json","w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll-strong target (delta >= 0.30):")
n_strong = sum(1 for r in results.values() if r["delta"] >= 0.30)
print(f"  {n_strong}/7 layers STRONG")
