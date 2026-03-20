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

- Parameters: 21,904
- Architecture: 2-layer MLP, dual heads (op + layer)
- Training corpus: 143,578 records (phoneme/word/sentence/paragraph)
- GT accuracy: 12/12 (100%)
- Tower root (paragraph, 1200 ops): 5f45a5f43af992a49cd4b8961abfd690ecfc202c38f066aa52fcfc0f73b785d6

## Running

Requires FARD v1.6.0 and libfard_onnx.dylib from the FARD repo.

    ./watch_tower.sh

## Roadmap

Stage 1 (current): 21,904 param MLP, 7-layer tower, phoneme/word/sentence/paragraph corpus
Stage 2: Attention mechanism, multi-head, longer sequences
Stage 3: Training loop written in FARD — every gradient step a witnessed execution
Stage 4: Competitive on standard benchmarks with full audit trail

## Why This Matters

A transformer where you can verify exactly what training data was used, exactly what computation produced each output, and that the model running today is the same model that was benchmarked.

Built with FARD: https://github.com/mauludsadiq/FARD
