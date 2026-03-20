"""
Stage 5e — Train sentence-level model on real UD English EWT corpus.
"""
import json, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os
from collections import Counter

random.seed(42); torch.manual_seed(42)
T=8; FDIM=67; D=256; N_HEAD=4

print("Loading UD EWT corpus...")
sentences = []
with open("corpus/corpus_ud_ewt.ndjson") as f:
    for line in f: sentences.append(json.loads(line))
print(f"  {len(sentences)} sentences")

cc = [Counter() for _ in range(7)]
for s in sentences:
    for w in s["words"]:
        for i in range(7): cc[i][w["layer_class_ids"][f"L{i}"]] += 1
vocab = [max(cc[i].keys())+1 for i in range(7)]
layer_names = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]
print("Vocab:", {layer_names[i]: vocab[i] for i in range(7)})
for i in range(7):
    print(f"  L{i} {layer_names[i]:<12}: {len(cc[i]):5d} unique classes")

def encode_sentence(sent):
    x  = np.zeros((T, FDIM), dtype=np.float32)
    ys = np.zeros((7, T), dtype=np.int64)
    words = sent["words"][:T]
    while len(words) < T: words = words + [words[-1]]
    for t, w in enumerate(words):
        lids = w["layer_class_ids"]
        ph = lids["L0"]; mo = lids["L2"]
        if ph < 44: x[t, ph] = 1.0
        if mo < 16: x[t, 44+mo] = 1.0
        for i in range(7):
            x[t, 60+i] = lids[f"L{i}"] / max(vocab[i], 1)
            ys[i, t]   = lids[f"L{i}"]
    return x, ys

print("Encoding...")
enc = [encode_sentence(s) for s in sentences]
X  = torch.tensor(np.stack([e[0] for e in enc]))
Ys = [torch.tensor(np.stack([e[1][i] for e in enc])) for i in range(7)]

idx = list(range(len(sentences))); random.shuffle(idx)
split = int(0.85*len(idx))
tr_idx = torch.tensor(idx[:split]); va_idx = torch.tensor(idx[split:])
X_tr=X[tr_idx]; X_va=X[va_idx]
Y_tr=[Y[tr_idx] for Y in Ys]; Y_va=[Y[va_idx] for Y in Ys]
print(f"  train={len(tr_idx)} val={len(va_idx)}")

class SentenceLayerwiseModel(nn.Module):
    def __init__(self, fdim, d, n_heads, vocab_sizes, seq_len):
        super().__init__()
        self.inp = nn.Linear(fdim, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        self.mlp_transitions = nn.ModuleList([
            nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(0.1),
                         nn.Linear(d*2,d),nn.LayerNorm(d)) for _ in range(6)])
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(d,n_heads,dropout=0.1,batch_first=True) for _ in range(7)])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(7)])
        self.attn_drops = nn.ModuleList([nn.Dropout(0.1) for _ in range(7)])
        self.heads   = nn.ModuleList([nn.Linear(d,v) for v in vocab_sizes])
        self.consist = nn.ModuleList([nn.Linear(d,vocab_sizes[i+1]) for i in range(6)])

    def forward(self, x):
        B,T,_ = x.shape
        pos = torch.arange(T,device=x.device).unsqueeze(0)
        h = [None]*7
        h[0] = self.inp(x) + self.pos_emb(pos)
        a,_ = self.attn_blocks[0](h[0],h[0],h[0])
        h[0] = self.attn_norms[0](h[0]+self.attn_drops[0](a))
        for i in range(6):
            hn = h[i]+self.mlp_transitions[i](h[i])
            a,_ = self.attn_blocks[i+1](hn,hn,hn)
            h[i+1] = self.attn_norms[i+1](hn+self.attn_drops[i+1](a))
        logits  = [self.heads[i](h[i]) for i in range(7)]
        consist = [self.consist[i](h[i]) for i in range(6)]
        return logits, consist

    def n_params(self): return sum(p.numel() for p in self.parameters())

model = SentenceLayerwiseModel(FDIM, D, N_HEAD, vocab, T)
print(f"Model: {model.n_params():,} params  D={D}")

opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
best_mean = 0.0

for ep in range(150):
    model.train()
    perm = torch.randperm(len(X_tr))
    tloss = 0
    for i in range(0, len(perm), 32):
        idx_b = perm[i:i+32]
        x = X_tr[idx_b]; ys = [Y_tr[li][idx_b] for li in range(7)]
        logits, consist = model(x)
        cls_loss = sum(F.cross_entropy(logits[li].reshape(-1,vocab[li]),ys[li].reshape(-1)) for li in range(7))
        con_loss = sum(F.cross_entropy(consist[li].reshape(-1,vocab[li+1]),ys[li+1].reshape(-1)) for li in range(6))/6.0
        loss = cls_loss + 0.5*con_loss
        opt.zero_grad(); loss.backward(); opt.step()
        tloss += loss.item()
    sched.step()
    if ep % 25 == 24:
        model.eval()
        with torch.no_grad():
            logits,_ = model(X_va)
            accs = [(logits[li].reshape(-1,vocab[li]).argmax(1)==Y_va[li].reshape(-1)).float().mean().item() for li in range(7)]
        mean_acc = sum(accs)/7
        if mean_acc > best_mean:
            best_mean = mean_acc
            torch.save(model.state_dict(),"train/model_ud_ewt_best.pt")
        print(f"ep={ep+1:3d} loss={tloss:.1f} mean={mean_acc:.3f} [{' '.join(f'L{i}={a:.2f}' for i,a in enumerate(accs))}]")

model.load_state_dict(torch.load("train/model_ud_ewt_best.pt",map_location="cpu"))
model.eval()
print(f"\n{'='*60}\nLLR BENCHMARK — UD EWT (real corpus)\n{'='*60}")
with torch.no_grad():
    logits,_ = model(X_va)

results = {}
for li in range(7):
    acc   = (logits[li].reshape(-1,vocab[li]).argmax(1)==Y_va[li].reshape(-1)).float().mean().item()
    n_cls = len(cc[li])
    chance= 1.0/n_cls
    delta = acc-chance
    q     = "STRONG" if delta>0.3 else "MODERATE" if delta>0.1 else "WEAK"
    results[layer_names[li]] = {"acc":acc,"chance":chance,"delta":delta,"n_classes":n_cls}
    print(f"  L{li} {layer_names[li]:<12} acc={acc:.3f} chance={chance:.3f} delta=+{delta:.3f}  {q}")

n_strong = sum(1 for r in results.values() if r["delta"]>=0.30)
print(f"\n{n_strong}/7 layers STRONG on real corpus")
os.makedirs("benchmark",exist_ok=True)
with open("benchmark/stage5e_ud_ewt_llr.json","w") as f:
    json.dump(results,f,indent=2)
print("Saved benchmark/stage5e_ud_ewt_llr.json")
