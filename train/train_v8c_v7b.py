import json, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys
from collections import defaultdict, Counter

FDIM   = 33
LAYERS = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]
GT     = [(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,6),(5,6),(6,6)]

print("Loading corpus_v7b...")
seqs = []
buf  = []
attn_counts = defaultdict(Counter)

with open("corpus/corpus_v7b.ndjson") as f:
    for line in f:
        r  = json.loads(line)
        bi = r["block_idx"] % 12
        li = LAYERS.index(r["src_layer"]) if r["src_layer"] in LAYERS else 0
        attn_counts[li][r["attn_top"]] += 1
        if bi == 0:
            if len(buf) == 12: seqs.append(buf)
            buf = [r]
        else:
            buf.append(r)
    if len(buf) == 12: seqs.append(buf)

print(f"  {len(seqs)} sequences")

attn_vocab = {}
for li in range(7):
    top = [w for w,_ in attn_counts[li].most_common(256)]
    attn_vocab[li] = {w:i for i,w in enumerate(top)}
    print(f"  L{li} {LAYERS[li]:<12}: {len(attn_vocab[li])} classes")

random.shuffle(seqs)
split    = int(0.9*len(seqs))
tr_seqs  = seqs[:split]
va_seqs  = seqs[split:]

def encode_seq(records):
    x  = np.zeros((12, FDIM), dtype=np.float32)
    al = np.zeros(7, dtype=np.int64)
    prev_op = 0
    for i, r in enumerate(records):
        bi = r["block_idx"] % 12
        li = LAYERS.index(r["src_layer"]) if r["src_layer"] in LAYERS else 0
        tau = r.get("tau",1.0); tk = r.get("top_k",3)
        tb  = 0 if tau<=0.5 else 1 if tau<=1.0 else 2 if tau<=2.0 else 3
        x[i,0]=bi/60.0; x[i,12+li]=1.0; x[i,19]=bi/60.0
        x[i,20+tb]=1.0; x[i,24]=tk/10.0; x[i,25+prev_op]=1.0
        prev_op = r["op_class"]
        al[li]  = attn_vocab[li].get(r["attn_top"], 0)
    return x, al

def build(seqs):
    enc = [encode_seq(s) for s in seqs]
    return (
        torch.tensor(np.stack([e[0] for e in enc])),
        torch.tensor(np.stack([e[1] for e in enc])),
        torch.tensor(np.array([[r["op_class"]  for r in s] for s in seqs], dtype=np.int64)),
        torch.tensor(np.array([[r["tgt_class"] for r in s] for s in seqs], dtype=np.int64)),
    )

print("Building tensors...")
tr = build(tr_seqs)
va = build(va_seqs)
print(f"  train={len(tr[0])} val={len(va[0])}")

class ProposerV8c(nn.Module):
    def __init__(self, d=128, heads=4, nl=2, fdim=FDIM, avs=None):
        super().__init__()
        self.inp  = nn.Linear(fdim, d)
        self.pos  = nn.Embedding(12, d)
        self.blks = nn.ModuleList([self._b(d,heads) for _ in range(nl)])
        self.op   = nn.Linear(d, 8)
        self.tgt  = nn.Linear(d, 8)
        self.lp   = nn.ModuleList([nn.Linear(d,d) for _ in range(7)])
        self.ah   = nn.ModuleList([nn.Linear(d, avs[li]) for li in range(7)])

    def _b(self, d, h):
        return nn.ModuleDict({
            "a": nn.MultiheadAttention(d, h, dropout=0.1, batch_first=True),
            "f": nn.Sequential(nn.Linear(d,d*4), nn.GELU(), nn.Dropout(0.1), nn.Linear(d*4,d)),
            "n1": nn.LayerNorm(d), "n2": nn.LayerNorm(d), "d": nn.Dropout(0.1),
        })

    def forward(self, x):
        B,T,_ = x.shape
        h = self.inp(x) + self.pos(torch.arange(T, device=x.device).unsqueeze(0))
        m = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for b in self.blks:
            a,_ = b["a"](h,h,h, attn_mask=m, is_causal=True)
            h   = b["n1"](h + b["d"](a))
            h   = b["n2"](h + b["d"](b["f"](h)))
        gt = torch.tensor([0,0,0,0,1,2,3,4,5,6,6,6], device=x.device)
        ls = torch.zeros(B,7,h.shape[-1], device=x.device)
        for li in range(7):
            mk = (gt==li).unsqueeze(0).expand(B,-1)
            ls[:,li,:] = (h*mk.unsqueeze(-1).float()).sum(1) / mk.float().sum(1,keepdim=True).unsqueeze(-1).clamp(min=1).squeeze(-1)
        proj = torch.stack([self.lp[i](ls[:,i,:]) for i in range(7)], dim=1)
        return self.op(h), self.tgt(h), proj, [self.ah[li](proj[:,li,:]) for li in range(7)]

    def n_params(self): return sum(p.numel() for p in self.parameters())

avs   = [len(attn_vocab[li]) for li in range(7)]
model = ProposerV8c(avs=avs)
print(f"ProposerV8c: {model.n_params():,} params")

opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
best  = 0.0

for ep in range(60):
    model.train()
    perm = torch.randperm(len(tr[0]))
    tloss = 0
    for i in range(0, len(perm), 64):
        idx = perm[i:i+64]
        x,al,oy,ty = tr[0][idx],tr[1][idx],tr[2][idx],tr[3][idx]
        op_l,tgt_l,_,atl = model(x)
        loss = F.cross_entropy(op_l.reshape(-1,8), oy.reshape(-1)) + \
               0.5*F.cross_entropy(tgt_l.reshape(-1,8), ty.reshape(-1)) + \
               0.5*sum(F.cross_entropy(atl[li], al[:,li]) for li in range(7))/7.0
        opt.zero_grad(); loss.backward(); opt.step()
        tloss += loss.item()
    sched.step()
    if ep % 10 == 9:
        model.eval()
        with torch.no_grad():
            op_l,_,_,atl = model(va[0])
            oa = (op_l.reshape(-1,8).argmax(1)==va[2].reshape(-1)).float().mean().item()
            aa = [(atl[li].argmax(1)==va[1][:,li]).float().mean().item() for li in range(7)]
        if oa > best:
            best = oa
            torch.save({"sd":model.state_dict(),"av":attn_vocab,"avs":avs},
                       "train/model_v8c_v7b_best.pt")
        print(f"ep={ep+1:3d} loss={tloss:.1f} op={oa:.4f} [{' '.join(f'L{i}={a:.2f}' for i,a in enumerate(aa))}]")

model.eval()
print(f"\n{'='*60}\nLLR BENCHMARK\n{'='*60}")
with torch.no_grad():
    op_l,_,_,atl = model(va[0])
    oa = (op_l.reshape(-1,8).argmax(1)==va[2].reshape(-1)).float().mean().item()
print(f"Op accuracy: {oa:.4f}\nLayer probe:")
res = {}
for li in range(7):
    a  = (atl[li].argmax(1)==va[1][:,li]).float().mean().item()
    c  = 1.0/avs[li]
    d  = a-c
    q  = "STRONG" if d>0.3 else "MODERATE" if d>0.1 else "WEAK"
    res[LAYERS[li]] = {"acc":a,"chance":c,"delta":d,"classes":avs[li]}
    print(f"  L{li} {LAYERS[li]:<12} acc={a:.3f} chance={c:.3f} delta=+{d:.3f}  {q}")

os.makedirs("benchmark", exist_ok=True)
with open("benchmark/llr_v8c_v7b_results.json","w") as f:
    json.dump(res, f, indent=2)
print("Saved benchmark/llr_v8c_v7b_results.json")
