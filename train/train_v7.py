"""
Stage 2 training — ProposerV7 on sequence corpus
Reshapes corpus into 12-block sequences for attention training
"""
import json, random, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import defaultdict
from train.model_v7 import ProposerV7, OP_NAMES, TGT_NAMES

LAYERS = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]

def load_sequences(paths):
    """Load corpus and group into 12-block sequences by chain_hash prefix."""
    all_records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                all_records.append(json.loads(line))

    # Group by (phoneme_idx, tau, top_k, chain_hash of first block)
    # Use block_idx % 12 to reconstruct sequences
    sequences = defaultdict(list)
    for r in all_records:
        key = r["chain_hash"] + str(r.get("tau",1.0)) + str(r.get("top_k",3))
        sequences[key].append(r)

    # Build 12-step sequences
    seqs = []
    for key, records in sequences.items():
        records.sort(key=lambda r: r["block_idx"] % 12)
        # Take complete 12-block sequences only
        by_seq = defaultdict(list)
        for r in records:
            seq_id = r["block_idx"] // 12
            by_seq[seq_id].append(r)
        for sid, recs in by_seq.items():
            recs.sort(key=lambda r: r["block_idx"] % 12)
            if len(recs) == 12:
                seqs.append(recs)

    print(f"  Loaded {len(all_records)} records -> {len(seqs)} complete sequences")
    return seqs

def encode_record(r):
    x = np.zeros(25, dtype=np.float32)
    bi = r["block_idx"] % 12
    li = LAYERS.index(r["src_layer"]) if r["src_layer"] in LAYERS else 0
    sc = bi
    tau = r.get("tau", 1.0)
    tk  = r.get("top_k", 3)
    tau_bin = 0 if tau<=0.5 else 1 if tau<=1.0 else 2 if tau<=2.0 else 3
    x[0]       = (bi * 10000 + 30) / 600000  # bi/60 rounded
    x[12+li]   = 1.0
    x[19]      = x[0]
    x[20+tau_bin] = 1.0
    x[24]      = tk / 10.0
    return x

def build_tensors(seqs):
    xs, op_ys, tgt_ys = [], [], []
    for seq in seqs:
        x_seq  = np.stack([encode_record(r) for r in seq])   # [12, 25]
        op_seq = np.array([r["op_class"] for r in seq])      # [12]
        tgt_seq = np.array([r["tgt_class"] for r in seq])    # [12]
        xs.append(x_seq)
        op_ys.append(op_seq)
        tgt_ys.append(tgt_seq)
    return (torch.tensor(np.array(xs)),
            torch.tensor(np.array(op_ys, dtype=np.int64)),
            torch.tensor(np.array(tgt_ys, dtype=np.int64)))

print("Loading corpus...")
seqs = load_sequences(["corpus/corpus_v4.ndjson", "corpus/corpus_v5.ndjson"])
random.shuffle(seqs)
split = int(0.9 * len(seqs))
tr_seqs, va_seqs = seqs[:split], seqs[split:]

print("Building tensors...")
tr = build_tensors(tr_seqs)
va = build_tensors(va_seqs)
print(f"  train: {len(tr[0])} sequences  val: {len(va[0])} sequences")

model = ProposerV7()
print(f"ProposerV7: {model.n_params():,} parameters")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

best_acc = 0.0
for ep in range(60):
    model.train()
    perm = torch.randperm(len(tr[0]))
    total_loss = 0
    for i in range(0, len(perm), 32):
        idx = perm[i:i+32]
        x, oy, ty = tr[0][idx], tr[1][idx], tr[2][idx]
        op_logits, tgt_logits = model(x)
        # op_logits: [B, 12, 8], oy: [B, 12]
        loss = F.cross_entropy(op_logits.reshape(-1,8), oy.reshape(-1)) +                0.5 * F.cross_entropy(tgt_logits.reshape(-1,8), ty.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    sched.step()

    if ep % 10 == 9:
        model.eval()
        with torch.no_grad():
            op_logits, _ = model(va[0])
            acc = (op_logits.reshape(-1,8).argmax(1) == va[1].reshape(-1)).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "train/model_v7_best.pt")
        print(f"  ep={ep+1:3d} loss={total_loss:.2f} val_acc={acc:.4f} best={best_acc:.4f}")

print(f"\nBest val_acc: {best_acc:.4f}")

# Load best and export
model.load_state_dict(torch.load("train/model_v7_best.pt"))
model.eval()

# Verify GT sequence
import onnxruntime as ort
GT = [(0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(4,6),(5,6),(6,6)]

# Build full 12-block sequence input
x_gt = np.zeros((1, 12, 25), dtype=np.float32)
for bi, (_, li) in enumerate(GT):
    x_gt[0, bi, 0]     = (bi * 10000 + 30) / 600000
    x_gt[0, bi, 12+li] = 1.0
    x_gt[0, bi, 19]    = x_gt[0, bi, 0]
    x_gt[0, bi, 21]    = 1.0
    x_gt[0, bi, 24]    = 0.3

with torch.no_grad():
    op_logits, tgt_logits = model(torch.tensor(x_gt))
preds = op_logits[0].argmax(1).tolist()
correct = sum(1 for i,(exp,_) in enumerate(GT) if preds[i]==exp)
print(f"\nGT accuracy: {correct}/12 = {correct/12*100:.1f}%")
for i,(exp,li) in enumerate(GT):
    ok = "✓" if preds[i]==exp else "✗"
    print(f"  blk={i:2d} {OP_NAMES[preds[i]]:<20} {ok} (expected {OP_NAMES[exp]})")

# Export to ONNX — note: sequence model takes [B, 12, 25]
torch.onnx.export(model, torch.zeros(1,12,25), "train/model_v7.onnx",
    input_names=["sequence"], output_names=["op_logits","tgt_logits"],
    dynamic_axes={"sequence":{0:"batch"},"op_logits":{0:"batch"},"tgt_logits":{0:"batch"}},
    opset_version=17)
print("\nexported train/model_v7.onnx")
import os
print(f"model size: {os.path.getsize('train/model_v7.onnx')//1024} KB")
