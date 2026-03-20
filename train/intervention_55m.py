import json, torch, torch.nn as nn, numpy as np, random, os
from collections import Counter

LAYERS = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]
T=8; FDIM=67; D=512; N_HEAD=4

sentences = []
with open("corpus/corpus_sentences_v2.ndjson") as f:
    for line in f: sentences.append(json.loads(line))

cc = [Counter() for _ in range(7)]
for s in sentences:
    for w in s["words"]:
        for i in range(7): cc[i][w["layer_class_ids"][f"L{i}"]] += 1
vocab = [max(cc[i].keys())+1 for i in range(7)]

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

enc = [encode_sentence(s) for s in sentences]
X  = torch.tensor(np.stack([e[0] for e in enc]))
Ys = [torch.tensor(np.stack([e[1][i] for e in enc])) for i in range(7)]
random.seed(42)
idx = list(range(len(sentences))); random.shuffle(idx)
va_idx = torch.tensor(idx[int(0.85*len(idx)):])
X_va = X[va_idx]; Y_va = [Y[va_idx] for Y in Ys]

class SentenceLayerwiseModel(nn.Module):
    def __init__(self, fdim, d, n_heads, vocab_sizes, seq_len):
        super().__init__()
        self.inp = nn.Linear(fdim, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        self.mlp_transitions = nn.ModuleList([
            nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(0.1),
                         nn.Linear(d*2,d),nn.LayerNorm(d)) for _ in range(6)])
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(d, n_heads, dropout=0.1, batch_first=True)
            for _ in range(7)])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(7)])
        self.attn_drops = nn.ModuleList([nn.Dropout(0.1) for _ in range(7)])
        self.heads   = nn.ModuleList([nn.Linear(d, v) for v in vocab_sizes])
        self.consist = nn.ModuleList([nn.Linear(d, vocab_sizes[i+1]) for i in range(6)])

    def get_hidden(self, x):
        B,T,_ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = [None]*7
        h[0] = self.inp(x) + self.pos_emb(pos)
        a,_ = self.attn_blocks[0](h[0],h[0],h[0])
        h[0] = self.attn_norms[0](h[0]+self.attn_drops[0](a))
        for i in range(6):
            hn = h[i] + self.mlp_transitions[i](h[i])
            a,_ = self.attn_blocks[i+1](hn,hn,hn)
            h[i+1] = self.attn_norms[i+1](hn+self.attn_drops[i+1](a))
        return h

    def from_hidden(self, h, patch_layer=None, patch_val=None):
        h = list(h)
        if patch_layer is not None:
            h[patch_layer] = patch_val
            for i in range(patch_layer, 6):
                hn = h[i] + self.mlp_transitions[i](h[i])
                a,_ = self.attn_blocks[i+1](hn,hn,hn)
                h[i+1] = self.attn_norms[i+1](hn+self.attn_drops[i+1](a))
        return [self.heads[i](h[i]) for i in range(7)]

    def forward(self, x):
        h = self.get_hidden(x)
        return self.from_hidden(h), [self.consist[i](h[i]) for i in range(6)], h

print("Loading 55M model...")
model = SentenceLayerwiseModel(FDIM, D, N_HEAD, vocab, T)
model.load_state_dict(torch.load("train/model_sentence_512_best.pt", map_location="cpu"))
model.eval()
print(f"  {sum(p.numel() for p in model.parameters()):,} params")

with torch.no_grad():
    h_clean = model.get_hidden(X_va)
    baseline = [
        (model.from_hidden(h_clean)[li].reshape(-1,vocab[li]).argmax(1)==Y_va[li].reshape(-1))
        .float().mean().item() for li in range(7)]

print(f"Baseline: {' '.join(f'L{i}={a:.2f}' for i,a in enumerate(baseline))}")
print()

all_ok = True
results = {"baseline": baseline, "vocab": vocab, "params": 55274628, "interventions": {}}
for k in range(7):
    with torch.no_grad():
        h_c = [h.clone() for h in h_clean]
        h_c[k] = torch.zeros_like(h_c[k])
        for i in range(k, 6):
            hn = h_c[i] + model.mlp_transitions[i](h_c[i])
            a,_ = model.attn_blocks[i+1](hn,hn,hn)
            h_c[i+1] = model.attn_norms[i+1](hn+model.attn_drops[i+1](a))
        accs_c = [(model.from_hidden(h_c)[li].reshape(-1,vocab[li]).argmax(1)==Y_va[li].reshape(-1)).float().mean().item() for li in range(7)]
        accs_p = [(model.from_hidden(h_c,patch_layer=k,patch_val=h_clean[k])[li].reshape(-1,vocab[li]).argmax(1)==Y_va[li].reshape(-1)).float().mean().item() for li in range(7)]

    damaged_below = [li for li in range(0,k) if accs_c[li] < baseline[li]-0.10]
    ok = len(damaged_below)==0
    if not ok: all_ok = False
    print(f"zero h_{k} ({LAYERS[k]:<10}): {' '.join(f'{a:.2f}' for a in accs_c)}  {'✓' if ok else '✗'}")
    print(f"patch h_{k}            : {' '.join(f'{a:.2f}' for a in accs_p)}")
    results["interventions"][f"h{k}_{LAYERS[k]}"] = {"corrupt": accs_c, "patched": accs_p, "localized": ok}

print(f"\n{'✓ ALL LOCALIZED — 55M hierarchy is causal' if all_ok else '✗ NOT ALL LOCALIZED'}")
os.makedirs("benchmark", exist_ok=True)
with open("benchmark/stage5d_intervention.json","w") as f:
    json.dump(results, f, indent=2)
print("Saved benchmark/stage5d_intervention.json")
