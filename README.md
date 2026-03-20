# FARD Semantic Transformer

A transformer architecture built entirely in FARD — the deterministic, content-addressed scripting language.

Every forward pass produces a cryptographic receipt. Every training step is a witnessed execution. The execution chain is auditable and reproducible on any machine.

This is the demo for FARD. And a serious attempt to build a transformer that rivals mainstream architectures.

## The Idea

Modern transformers produce outputs. This transformer produces outputs with proof.

Probability sits on top of traceability. The model can be stochastic but the computation that produced each output is committed to a SHA-256 chain. The tower root is a single hash that proves every weight, every input, every intermediate state, in order.

## Architecture

7-layer certified linguistic hierarchy: Phoneme -> Syllable -> Morpheme -> Word -> Phrase -> Semantic -> Discourse

12-block canonical sequence per input unit:
- Block  0: SELECT_UNIVERSE
- Block  1: WITNESS_NEAREST
- Block  2: ATTEND
- Blocks 3-8: FFN_STEP (through each layer)
- Block  9: PROJECT_LAYER
- Block 10: RETURN_SET
- Block 11: ACCEPT

## Current State

- Parameters: 11,323,524 (sentence-level transformer, Stage 5c)
- Architecture: Explicit per-layer hidden states h_0..h_6, deep supervision, consistency losses
- Training corpus: 4,850 sentences x 8 words (corpus_v8c_v2 + sentence grouping)
- LLR Benchmark: 7/7 layers STRONG at both word level and sentence level
- Causal intervention: 7/7 interventions perfectly localized
- Tower root (paragraph, 1200 ops): 5f45a5f43af992a49cd4b8961abfd690ecfc202c38f066aa52fcfc0f73b785d6

## The LLR Benchmark

Linguistic Layered Reasoning — tests whether each layer of the hierarchy
specialises in distinct, linearly recoverable information.

Results:

| Layer     | Accuracy | Chance | Delta  | Result   |
|-----------|----------|--------|--------|----------|
| PHONEME   | 1.000    | 0.029  | +0.971 | STRONG   |
| SYLLABLE  | 0.983    | 0.000  | +0.983 | STRONG   |
| MORPHEME  | 1.000    | 0.062  | +0.938 | STRONG   |
| WORD      | 0.987    | 0.000  | +0.987 | STRONG   |
| PHRASE    | 1.000    | 0.100  | +0.900 | STRONG   |
| SEMANTIC  | 0.744    | 0.005  | +0.739 | STRONG   |
| DISCOURSE | 0.951    | 0.010  | +0.941 | STRONG   |

No layer is decorative. Every layer is a necessary information bottleneck.

## The Causal Intervention Test

Zeroing hidden state h_k damages exactly layers k and above.
Layers below k are unaffected.

| Intervention | L0   | L1   | L2   | L3   | L4   | L5   | L6   |
|--------------|------|------|------|------|------|------|------|
| baseline     | 1.00 | 0.98 | 1.00 | 0.99 | 1.00 | 0.93 | 0.98 |
| zero h_0     | 0.07 | 0.00 | 0.05 | 0.00 | 0.10 | 0.00 | 0.01 |
| zero h_1     | 1.00 | 0.00 | 0.05 | 0.00 | 0.10 | 0.01 | 0.01 |
| zero h_2     | 1.00 | 0.98 | 0.07 | 0.00 | 0.09 | 0.01 | 0.01 |
| zero h_3     | 1.00 | 0.98 | 1.00 | 0.01 | 0.10 | 0.00 | 0.01 |
| zero h_4     | 1.00 | 0.98 | 1.00 | 0.99 | 0.11 | 0.00 | 0.01 |
| zero h_5     | 1.00 | 0.98 | 1.00 | 0.99 | 1.00 | 0.01 | 0.01 |
| zero h_6     | 1.00 | 0.98 | 1.00 | 0.99 | 1.00 | 0.93 | 0.01 |

All 7 interventions perfectly localized.

This is impossible to demonstrate with GPT, BERT, or LLaMA. They have no
exposed hierarchy. Randomizing any internal representation in a standard
transformer damages all outputs indiscriminately. The FARD Semantic Tower
hierarchy is functional, localized, and causal.

## Running

Requires FARD v1.6.0 and libfard_onnx.dylib from the FARD repo.

    ./watch_tower.sh

Regenerate the corpus:

    python3 src/generate_corpus_v8c.py

## Roadmap

Stage 1 (done): 21,904 param MLP, 7-layer tower, 12/12 GT accuracy
Stage 2 (done): 403k param causal transformer, prev_op feedback, 12/12 autoregressive
Stage 3 (done): Witnessed training loop — every epoch FARD-receipted, audit chain
Stage 4 (done): LLR benchmark 7/7 STRONG, causal intervention test 7/7 localized
Stage 5 (in progress):
  5a (done): Morpheme inventory 16->140 classes (full English derivational morphology)
  5b (done): Phrase inventory 10->50 skeleton types (full English CFG)
  5c (done): Sentence-level model — 11.3M params, T=8 words, cross-word attention
             Factorized semantic (60 types) and discourse (24 types)
             7/7 layers STRONG at sentence level
  5d: Scale hidden dim 128->512, 50M params
  5e: Real corpus integration
Stage 6: Training loop in FARD — every gradient step a witnessed execution
Stage 7: Competitive on standard benchmarks with full audit trail

## Why This Matters

A transformer where you can verify exactly what training data was used,
exactly what computation produced each output, and that the model running
today is the same model that was benchmarked.

And where you can surgically intervene on any layer and prove the hierarchy
is real — not emergent, not approximate, but causal.

Built with FARD: https://github.com/mauludsadiq/FARD
