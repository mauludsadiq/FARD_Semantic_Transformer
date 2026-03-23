"""
Regenerate corpus_v8c_remapped.ndjson from corpus_v8c.ndjson.
Remaps class IDs by frequency so high-frequency classes get low IDs,
preventing cap-induced collapse.

Usage: python3 src/remap_corpus.py
"""
import json
from collections import Counter

CAPS=[34,256,256,256,50,256,256]

rows=[]
with open("corpus/corpus_v8c.ndjson") as f:
    for line in f: rows.append(json.loads(line))

remaps=[]
for li in range(7):
    cnt=Counter(r["layer_class_ids"][f"L{li}"] for r in rows)
    freq_order=sorted(cnt.keys(), key=lambda x: -cnt[x])
    remaps.append({old:new for new,old in enumerate(freq_order)})

with open("corpus/corpus_v8c_remapped.ndjson","w") as f:
    for r in rows:
        nr=json.loads(json.dumps(r))
        for li in range(7):
            raw=r["layer_class_ids"][f"L{li}"]
            nr["layer_class_ids"][f"L{li}"]=min(remaps[li][raw],CAPS[li]-1)
        f.write(json.dumps(nr)+"\n")

print(f"Written corpus/corpus_v8c_remapped.ndjson ({len(rows)} rows)")
