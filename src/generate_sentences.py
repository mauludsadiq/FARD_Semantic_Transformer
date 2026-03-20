"""
Sentence corpus generator for Stage 5c.
Groups corpus_v8c words into sentences of T words sharing a phrase structure.
Each sentence = T words that collectively realize one phrase skeleton.
"""
import json, random, os
from collections import defaultdict

random.seed(42)

print("Loading corpus_v8c...")
rows = []
with open("corpus/corpus_v8c.ndjson") as f:
    for line in f: rows.append(json.loads(line))

# Group by phrase_id
phrase_groups = defaultdict(list)
for i, r in enumerate(rows):
    phrase_groups[r["phrase"]["object"]["phrase_id"]].append(r)

print(f"  {len(rows)} rows, {len(phrase_groups)} phrase groups")

# Build sentences: each sentence is T=8 words from the same phrase group
# This ensures all words share a phrase structure (realistic constituency)
T = 8
sentences = []

for phrase_id, phrase_rows in phrase_groups.items():
    random.shuffle(phrase_rows)
    # Create multiple sentences per phrase group
    for i in range(0, len(phrase_rows) - T + 1, T):
        sent = phrase_rows[i:i+T]
        if len(sent) == T:
            sentences.append(sent)

print(f"  {len(sentences)} sentences of T={T} words")

# Write sentence corpus
os.makedirs("corpus", exist_ok=True)
with open("corpus/corpus_sentences.ndjson", "w") as f:
    for sent in sentences:
        f.write(json.dumps({
            "sentence_id": f"sent:{sent[0]['phrase']['object']['phrase_id']}_{len(sentences)}",
            "phrase_id":   sent[0]["phrase"]["object"]["phrase_id"],
            "skeleton_id": sent[0]["phrase"]["object"].get("skeleton_id",""),
            "length":      T,
            "words": [
                {
                    "layer_class_ids": r["layer_class_ids"],
                    "surface_form":    r["word"]["object"]["surface_form"],
                    "phoneme_class":   r["layer_class_ids"]["L0"],
                    "syllable_class":  r["layer_class_ids"]["L1"],
                    "morpheme_class":  r["layer_class_ids"]["L2"],
                    "word_class":      r["layer_class_ids"]["L3"],
                    "phrase_class":    r["layer_class_ids"]["L4"],
                    "semantic_class":  r["layer_class_ids"]["L5"],
                    "discourse_class": r["layer_class_ids"]["L6"],
                }
                for r in sent
            ]
        }) + "\n")

import os
size_kb = os.path.getsize("corpus/corpus_sentences.ndjson") // 1024
print(f"Written corpus/corpus_sentences.ndjson ({size_kb} KB, {len(sentences)} sentences)")

# Stats
from collections import Counter
import math
layer_names = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]
keys = ["phoneme_class","syllable_class","morpheme_class","word_class","phrase_class","semantic_class","discourse_class"]
cc = [Counter() for _ in range(7)]
for sent_data in open("corpus/corpus_sentences.ndjson"):
    sent = json.loads(sent_data)
    for w in sent["words"]:
        for i,k in enumerate(keys):
            cc[i][w[k]] += 1

print("\nSentence corpus stats:")
for i in range(7):
    n = len(cc[i])
    total = sum(cc[i].values())
    H = -sum((c/total)*math.log2(c/total) for c in cc[i].values()) if n > 1 else 0
    print(f"  L{i} {layer_names[i]:<12}: {n:4d} unique  H={H:.2f} bits")
