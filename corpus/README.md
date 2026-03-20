# Corpus

## corpus_v4.ndjson (12.7 MB)
18,730 records. All 7 layers. Includes rejection scenarios.
Source: LLM_Nature_Semantic_Transformer Rust tower.

## corpus_v5.ndjson (75 MB) — not tracked in git
124,848 records. Word/sentence/paragraph sequences.
Generate: python3 -m train.generate_corpus_v5

## corpus_v6.ndjson (129 MB) — not tracked in git
192,000 records. 2000 semantic triples x 4 taus x 2 top_ks.
Diverse content across all 7 layers.
Generate: python3 data/generate_corpus_v6.py

## corpus_v8c.ndjson (131 MB) — not tracked in git
20,000 rows. Full 7-layer derivation chain per row.
Schema: csl.corpus_v8c_row.v1
Generate: python3 src/generate_corpus_v8c.py
