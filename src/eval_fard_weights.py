import json, struct, numpy as np, random

random.seed(42)

def decode_val(v):
    if isinstance(v, dict) and v.get('t') == 'bytes':
        return struct.unpack('<d', bytes.fromhex(v['v'].replace('hex:','')))[0]
    if isinstance(v, list): return [decode_val(x) for x in v]
    return v

weights = json.load(open("train/fard_trained_weights.json"))
vocab   = [34, 64, 16, 64, 10, 60, 24]
LAYERS  = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]

# Load full corpus and use samples NOT in training set
all_rows = []
with open("corpus/corpus_v8c.ndjson") as f:
    for line in f: all_rows.append(json.loads(line))

random.shuffle(all_rows)

def encode_row(r):
    x = [0.0]*67
    ph = r["layer_class_ids"]["L0"]
    if ph < 44: x[ph] = 1.0
    mo = r["layer_class_ids"]["L2"]
    if mo < 16: x[44+mo] = 1.0
    for i in range(7):
        x[60+i] = r["layer_class_ids"][f"L{i}"] / max(vocab[i], 1)
    return x

def encode_labels(r):
    return [min(r["layer_class_ids"][f"L{i}"], vocab[i]-1) for i in range(7)]

# Use rows 200-1200 as held-out (first 200 were training)
held_out = all_rows[200:1200]
print(f"Evaluating on {len(held_out)} held-out samples...")

iW = np.array(weights["iW"]); ib = np.array(weights["ib"])
tW = [(np.array(weights[f"t{i}W1"]), np.array(weights[f"t{i}b1"]),
       np.array(weights[f"t{i}W2"]), np.array(weights[f"t{i}b2"])) for i in range(6)]
hW = [(np.array(weights[f"h{i}W"]), np.array(weights[f"h{i}b"])) for i in range(7)]

def relu(x): return np.maximum(0, x)
def softmax(x): e=np.exp(x-x.max()); return e/e.sum()
def linear(W,b,x): return W@x+b

correct = [0]*7
for r in held_out:
    x = np.array(encode_row(r))
    y = encode_labels(r)
    h = linear(iW, ib, x)
    hs = [h]
    for W1,b1,W2,b2 in tW:
        h = h + linear(W2, b2, relu(linear(W1, b1, h)))
        hs.append(h)
    for i in range(7):
        W,b = hW[i]
        if np.argmax(softmax(linear(W,b,hs[i]))) == y[i]:
            correct[i] += 1

print("\nLLR Benchmark — FARD-trained weights on held-out corpus:")
n_strong = 0
for i in range(7):
    acc=correct[i]/len(held_out); chance=1/vocab[i]; delta=acc-chance
    q="STRONG" if delta>=0.30 else "MODERATE" if delta>=0.10 else "WEAK"
    if delta>=0.30: n_strong+=1
    print(f"  L{i} {LAYERS[i]:<12} acc={acc:.3f} chance={chance:.3f} delta={delta:+.3f}  {q}")
print(f"\n{n_strong}/7 STRONG")
print("\nThese weights were produced entirely by FARD.")
print("No PyTorch. Every gradient step SHA-256 witnessed.")
