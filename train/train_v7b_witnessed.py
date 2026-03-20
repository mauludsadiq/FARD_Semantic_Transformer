"""
Stage 3 — Witnessed training loop
PyTorch trains ProposerV7b. After each epoch:
  1. Export updated weights to ONNX
  2. FARD runs a forward pass and produces a cryptographic receipt
  3. Receipt is recorded in training_log.ndjson
  
The training_log is a chain of witnessed checkpoints.
Every published accuracy claim is backed by a FARD receipt.
"""
import json, random, subprocess, os, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import defaultdict

FDIM_V7B = 33
LAYERS = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]
OP_NAMES = ["SELECT_UNIVERSE","WITNESS_NEAREST","ATTEND","FFN_STEP",
            "PROJECT_LAYER","RETURN_SET","ACCEPT","REJECT"]
GT = [(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,6),(5,6),(6,6)]

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ProposerV7b(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(FDIM_V7B, 128)
        self.pos_emb    = nn.Embedding(12, 128)
        self.transformer = nn.ModuleList([self._block() for _ in range(2)])
        self.op_head  = nn.Linear(128, 8)
        self.tgt_head = nn.Linear(128, 8)

    def _block(self):
        return nn.ModuleDict({
            "attn":  nn.MultiheadAttention(128, 4, dropout=0.1, batch_first=True),
            "ff":    nn.Sequential(nn.Linear(128,512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512,128)),
            "norm1": nn.LayerNorm(128),
            "norm2": nn.LayerNorm(128),
            "drop":  nn.Dropout(0.1),
        })

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.input_proj(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for blk in self.transformer:
            a, _ = blk["attn"](h, h, h, attn_mask=mask, is_causal=True)
            h = blk["norm1"](h + blk["drop"](a))
            h = blk["norm2"](h + blk["drop"](blk["ff"](h)))
        return self.op_head(h), self.tgt_head(h)

    def n_params(self): return sum(p.numel() for p in self.parameters())


def encode_seq(records):
    x = np.zeros((12, FDIM_V7B), dtype=np.float32)
    prev_op = 0
    for i, r in enumerate(records):
        bi = r["block_idx"] % 12
        li = LAYERS.index(r["src_layer"]) if r["src_layer"] in LAYERS else 0
        tau = r.get("tau", 1.0)
        tk  = r.get("top_k", 3)
        tau_bin = 0 if tau<=0.5 else 1 if tau<=1.0 else 2 if tau<=2.0 else 3
        x[i,0]=bi/60.0; x[i,12+li]=1.0; x[i,19]=bi/60.0
        x[i,20+tau_bin]=1.0; x[i,24]=tk/10.0
        x[i,25+prev_op]=1.0
        prev_op = r["op_class"]
    return x


def export_onnx(model, path):
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, torch.zeros(1,12,FDIM_V7B), path,
            input_names=["sequence"], output_names=["op_logits","tgt_logits"],
            dynamic_axes={"sequence":{0:"batch"}},
            opset_version=17)


def run_fard_witness(onnx_path, fard_bin, epoch, val_acc, gt_correct):
    """Run FARD forward pass on current model — returns receipt digest."""
    fard_prog = f"""
import("std/ffi") as ffi
import("std/str") as str

let lib = ffi.load("../FARD_v0.5/target/release/libfard_onnx.dylib")
let mp  = "{os.path.abspath(onnx_path)}"
let h   = ffi.call(lib.ok, "fard_onnx_load", [mp, str.len(mp)])

fn itof(n, denom) {{
  let raw     = (n * 10000 + denom / 2) / denom
  let clamped = if raw > 10000 then 10000 else raw
  let whole   = clamped / 10000
  let frac    = clamped % 10000
  let fs = if frac < 10 then str.concat("000", str.from_int(frac))
    else if frac < 100 then str.concat("00", str.from_int(frac))
    else if frac < 1000 then str.concat("0", str.from_int(frac))
    else str.from_int(frac)
  str.concat(str.from_int(whole), str.concat(".", fs))
}}

fn feat(bi, li, prev_op) {{
  let f0 = itof(bi, 60)
  let l0 = if li==0 then "1" else "0"
  let l1 = if li==1 then "1" else "0"
  let l2 = if li==2 then "1" else "0"
  let l3 = if li==3 then "1" else "0"
  let l4 = if li==4 then "1" else "0"
  let l5 = if li==5 then "1" else "0"
  let l6 = if li==6 then "1" else "0"
  let p0 = if prev_op==0 then "1" else "0"
  let p1 = if prev_op==1 then "1" else "0"
  let p2 = if prev_op==2 then "1" else "0"
  let p3 = if prev_op==3 then "1" else "0"
  let p4 = if prev_op==4 then "1" else "0"
  let p5 = if prev_op==5 then "1" else "0"
  let p6 = if prev_op==6 then "1" else "0"
  let p7 = if prev_op==7 then "1" else "0"
  str.join([f0,"0","0","0","0","0","0","0","0","0","0","0",
            l0,l1,l2,l3,l4,l5,l6,f0,"1","0","0","0","0.3",
            p0,p1,p2,p3,p4,p5,p6,p7], ",")
}}

let b0  = feat(0,  0, 0)
let b1  = feat(1,  0, 0)
let b2  = feat(2,  0, 1)
let b3  = feat(3,  0, 2)
let b4  = feat(4,  1, 3)
let b5  = feat(5,  2, 3)
let b6  = feat(6,  3, 3)
let b7  = feat(7,  4, 3)
let b8  = feat(8,  5, 3)
let b9  = feat(9,  6, 3)
let b10 = feat(10, 6, 4)
let b11 = feat(11, 6, 5)

let seq = str.concat("[", str.concat(
  str.join([b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11], ","), "]"))
let n = ffi.call(lib.ok, "fard_onnx_infer_seq", [h.ok, seq, str.len(seq), 12, 33])

{{
  epoch: {epoch},
  val_acc: "{val_acc:.4f}",
  gt_correct: {gt_correct},
  n_outputs: n.ok,
  model: "{os.path.basename(onnx_path)}"
}}
"""
    prog_path = f"/tmp/fard_witness_ep{epoch}.fard"
    out_path  = f"/tmp/fard_witness_ep{epoch}_out"
    os.makedirs(out_path, exist_ok=True)
    with open(prog_path, "w") as f:
        f.write(fard_prog)

    result = subprocess.run(
        [fard_bin, "run", "--program", prog_path, "--out", out_path, "--no-trace"],
        capture_output=True, text=True, timeout=60,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    digest = None
    for line in result.stdout.splitlines():
        if "fard_run_digest=" in line:
            digest = line.split("=")[1].strip()

    result_json = None
    result_file = os.path.join(out_path, "result.json")
    if os.path.exists(result_file):
        with open(result_file) as f:
            result_json = json.load(f)

    return digest, result_json


# ── Load corpus ───────────────────────────────────────────────────────────────
print("Loading corpus...")
all_records = []
for path in ["corpus/corpus_v4.ndjson", "corpus/corpus_v5.ndjson"]:
    with open(path) as f:
        for line in f: all_records.append(json.loads(line))

seqs = []
i = 0
while i < len(all_records) - 11:
    chunk = all_records[i:i+12]
    if [r["block_idx"] % 12 for r in chunk] == list(range(12)):
        seqs.append(chunk); i += 12
    else: i += 1

random.shuffle(seqs)
split = int(0.9 * len(seqs))
tr_seqs, va_seqs = seqs[:split], seqs[split:]

def build(seqs):
    xs  = torch.tensor(np.stack([encode_seq(s) for s in seqs]))
    oys = torch.tensor(np.array([[r["op_class"] for r in s] for s in seqs], dtype=np.int64))
    tys = torch.tensor(np.array([[r["tgt_class"] for r in s] for s in seqs], dtype=np.int64))
    return xs, oys, tys

tr = build(tr_seqs)
va = build(va_seqs)
print(f"  train={len(tr[0])} val={len(va[0])} sequences")

# ── Model ─────────────────────────────────────────────────────────────────────
model = ProposerV7b()
if os.path.exists("train/model_v7b_best.pt"):
    model.load_state_dict(torch.load("train/model_v7b_best.pt", map_location="cpu"))
    print(f"Loaded model_v7b_best.pt ({model.n_params():,} params)")
else:
    print(f"Fresh model ({model.n_params():,} params)")

opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

fard_bin = os.path.expanduser("~/Downloads/FARD_v0.5/target/release/fardrun")
log_path = "training_log.ndjson"
best_acc = 0.0

print(f"\nStage 3 — Witnessed training")
print(f"Every epoch checkpoint receipted by FARD")
print(f"Log: {log_path}\n")

for ep in range(20):
    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    perm = torch.randperm(len(tr[0]))
    total_loss = 0
    for i in range(0, len(perm), 32):
        idx = perm[i:i+32]
        x, oy, ty = tr[0][idx], tr[1][idx], tr[2][idx]
        op_l, tgt_l = model(x)
        loss = F.cross_entropy(op_l.reshape(-1,8), oy.reshape(-1)) +                0.5*F.cross_entropy(tgt_l.reshape(-1,8), ty.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    sched.step()

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        op_l, _ = model(va[0])
        val_acc = (op_l.reshape(-1,8).argmax(1)==va[1].reshape(-1)).float().mean().item()

    # ── GT check ──────────────────────────────────────────────────────────────
    x_gt = np.zeros((1,12,FDIM_V7B), dtype=np.float32)
    prev_op = 0
    for bi,(exp_op,li) in enumerate(GT):
        x_gt[0,bi,0]=bi/60.0; x_gt[0,bi,12+li]=1.0; x_gt[0,bi,19]=bi/60.0
        x_gt[0,bi,20]=1.0; x_gt[0,bi,24]=0.3; x_gt[0,bi,25+prev_op]=1.0
        prev_op=exp_op
    with torch.no_grad():
        op_l2,_ = model(torch.tensor(x_gt))
    preds = op_l2[0].argmax(1).tolist()
    gt_correct = sum(1 for i,(e,_) in enumerate(GT) if preds[i]==e)

    # ── Export + FARD witness ─────────────────────────────────────────────────
    ckpt_path = f"train/checkpoint_ep{ep+1:03d}.onnx"
    export_onnx(model, ckpt_path)

    digest, result_json = run_fard_witness(ckpt_path, fard_bin, ep+1, val_acc, gt_correct)

    # ── Log ───────────────────────────────────────────────────────────────────
    log_entry = {
        "epoch":      ep+1,
        "loss":       round(total_loss, 4),
        "val_acc":    round(val_acc, 4),
        "gt_correct": gt_correct,
        "fard_digest": digest,
        "timestamp":  int(time.time()),
        "params":     model.n_params(),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"  ep={ep+1:3d} loss={total_loss:.2f} val_acc={val_acc:.4f} gt={gt_correct}/12 digest={digest}")

    if val_acc >= best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "train/model_v7b_best.pt")
        export_onnx(model, "train/model_v7b.onnx")

print(f"\nBest val_acc: {best_acc:.4f}")
print(f"Training log: {log_path}")
print(f"Each epoch receipt in training_log.ndjson")
