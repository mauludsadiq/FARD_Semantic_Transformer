"""
Stage 5c — Sentence-level layerwise transformer.

Input:  [B, T, 67] — batch of sentences, T words each
Per layer: MLP transition + self-attention over T words
Output: [B, T, vocab_i] per layer

Architecture:
  h_0[t]   = Linear(x[t])                    # word projection
  h_{i+1}[t] = h_i[t] + MLP_i(h_i[t])       # per-word transition
               + Attn_i(h_i)[t]              # cross-word attention
  logit_i[t] = Linear_i(h_i[t])             # per-word classification
"""
import json, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, math
from collections import Counter

random.seed(42)
torch.manual_seed(42)

T      = 8    # words per sentence
FDIM   = 67   # feature dim per word
D      = 512  # hidden dim (start smaller, scale up)
N_HEAD = 4
LAYERS_N = 2  # transformer layers per linguistic level

layer_names = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]
keys = ["phoneme_class","syllable_class","morpheme_class","word_class",
        "phrase_class","semantic_class","discourse_class"]

print("Loading sentence corpus...")
sentences = []
with open("corpus/corpus_sentences_v2.ndjson") as f:
    for line in f: sentences.append(json.loads(line))
print(f"  {len(sentences)} sentences")

# Build vocab
cc = [Counter() for _ in range(7)]
for s in sentences:
    for w in s["words"]:
        for i in range(7): cc[i][w["layer_class_ids"][f"L{i}"]] += 1
vocab = [max(cc[i].keys())+1 for i in range(7)]
print(f"  Vocab: {vocab}")

def encode_sentence(sent):
    x  = np.zeros((T, FDIM), dtype=np.float32)
    ys = np.zeros((7, T), dtype=np.int64)
    for t, w in enumerate(sent["words"][:T]):
        lids = w["layer_class_ids"]
        ph = lids["L0"]; mo = lids["L2"]
        if ph < 44: x[t, ph] = 1.0
        if mo < 16: x[t, 44+mo] = 1.0
        for i in range(7):
            x[t, 60+i] = lids[f"L{i}"] / max(vocab[i], 1)
            ys[i, t]   = lids[f"L{i}"]
    return x, ys

print("Encoding sentences...")
enc = [encode_sentence(s) for s in sentences]
X  = torch.tensor(np.stack([e[0] for e in enc]))   # [N, T, FDIM]
Ys = [torch.tensor(np.stack([e[1][i] for e in enc])) for i in range(7)]  # 7 x [N, T]

random.shuffle(list(range(len(sentences))))
idx = list(range(len(sentences)))
random.shuffle(idx)
split   = int(0.85 * len(idx))
tr_idx  = torch.tensor(idx[:split])
va_idx  = torch.tensor(idx[split:])

X_tr   = X[tr_idx];   X_va   = X[va_idx]
Y_tr   = [Y[tr_idx]  for Y in Ys]
Y_va   = [Y[va_idx]  for Y in Ys]
print(f"  train={len(tr_idx)} val={len(va_idx)} sentences")


class SentenceLayerwiseModel(nn.Module):
    """
    Sentence-level layerwise transformer.
    Each linguistic level gets:
      - A per-word MLP transition
      - A cross-word self-attention block
    """
    def __init__(self, fdim, d, n_heads, vocab_sizes, seq_len):
        super().__init__()
        self.d       = d
        self.seq_len = seq_len
        self.inp     = nn.Linear(fdim, d)
        self.pos_emb = nn.Embedding(seq_len, d)

        # Per-layer: MLP transition + self-attention
        self.mlp_transitions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d*2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(d*2, d), nn.LayerNorm(d)
            ) for _ in range(6)
        ])
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, dropout=0.1, batch_first=True)
            for _ in range(7)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(7)])
        self.attn_drops = nn.ModuleList([nn.Dropout(0.1) for _ in range(7)])

        # Classification heads
        self.heads = nn.ModuleList([nn.Linear(d, v) for v in vocab_sizes])

        # Consistency heads: layer i predicts layer i+1
        self.consist = nn.ModuleList([
            nn.Linear(d, vocab_sizes[i+1]) for i in range(6)
        ])
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def get_hidden(self, x):
        # x: [B, T, FDIM]
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = [None] * 7

        # h_0: project input + positional embedding
        h[0] = self.inp(x) + self.pos_emb(pos)

        # Apply attention at layer 0
        a, _ = self.attn_blocks[0](h[0], h[0], h[0])
        h[0] = self.attn_norms[0](h[0] + self.attn_drops[0](a))

        # h_1..h_6: MLP transition + attention
        for i in range(6):
            # Per-word MLP transition (residual)
            h_next = h[i] + self.mlp_transitions[i](h[i])
            # Cross-word attention
            a, _ = self.attn_blocks[i+1](h_next, h_next, h_next)
            h[i+1] = self.attn_norms[i+1](h_next + self.attn_drops[i+1](a))

        return h

    def forward(self, x):
        h = self.get_hidden(x)
        logits  = [self.heads[i](h[i]) for i in range(7)]    # [B,T,vocab_i]
        consist = [self.consist[i](h[i]) for i in range(6)]  # [B,T,vocab_{i+1}]
        return logits, consist, h

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


model = SentenceLayerwiseModel(FDIM, D, N_HEAD, vocab, T)
print(f"\nSentenceLayerwiseModel: {model.n_params():,} params")
print(f"  Hidden dim: {D}, Heads: {N_HEAD}, Seq len: {T}")

opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
best_mean = 0.0

for ep in range(80):
    model.train()
    perm = torch.randperm(len(X_tr))
    total_loss = 0
    for i in range(0, len(perm), 32):
        idx_b = perm[i:i+32]
        x = X_tr[idx_b]         # [B, T, FDIM]
        ys = [Y_tr[li][idx_b] for li in range(7)]  # [B, T]

        logits, consist, _ = model(x)

        # Classification loss at every position and layer
        cls_loss = sum(
            F.cross_entropy(logits[li].reshape(-1, vocab[li]), ys[li].reshape(-1))
            for li in range(7)
        )

        # Consistency loss: h_i predicts y_{i+1}
        con_loss = sum(
            F.cross_entropy(consist[li].reshape(-1, vocab[li+1]), ys[li+1].reshape(-1))
            for li in range(6)
        ) / 6.0

        loss = cls_loss + 0.5 * con_loss
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    sched.step()

    if ep % 10 == 9:
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(X_va)
            accs = [
                (logits[li].reshape(-1, vocab[li]).argmax(1) == Y_va[li].reshape(-1))
                .float().mean().item()
                for li in range(7)
            ]
        mean_acc = sum(accs) / 7
        if mean_acc > best_mean:
            best_mean = mean_acc
            torch.save(model.state_dict(), "train/model_sentence_512_best.pt")
        print(f"ep={ep+1:3d} loss={total_loss:.1f} mean={mean_acc:.3f} "
              f"[{' '.join(f'L{i}={a:.2f}' for i,a in enumerate(accs))}]")

# Final benchmark
model.load_state_dict(torch.load("train/model_sentence_512_best.pt", map_location="cpu"))
model.eval()
print(f"\n{'='*60}")
print(f"STAGE 5c LLR BENCHMARK — Sentence-level model")
print(f"{'='*60}")
with torch.no_grad():
    logits, _, _ = model(X_va)

results = {}
for li in range(7):
    acc    = (logits[li].reshape(-1,vocab[li]).argmax(1)==Y_va[li].reshape(-1)).float().mean().item()
    chance = 1.0/vocab[li]
    delta  = acc - chance
    q      = "STRONG" if delta>0.3 else "MODERATE" if delta>0.1 else "WEAK"
    results[layer_names[li]] = {"acc":acc,"chance":chance,"delta":delta,"vocab":vocab[li]}
    print(f"  L{li} {layer_names[li]:<12} acc={acc:.3f} chance={chance:.3f} delta=+{delta:.3f}  {q}")

n_strong = sum(1 for r in results.values() if r["delta"]>=0.30)
print(f"\n{n_strong}/7 layers STRONG")
print(f"Model params: {model.n_params():,}")

os.makedirs("benchmark", exist_ok=True)
with open("benchmark/stage5d_512_llr.json","w") as f:
    json.dump(results, f, indent=2)
