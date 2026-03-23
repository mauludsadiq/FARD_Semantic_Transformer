# FARD Semantic Transformer

A 7-layer linguistic hierarchy trained entirely in FARD — a deterministic,
content-addressed scripting language. Every gradient step produces a
SHA-256 receipt. The computation chain is auditable and reproducible.

## Architecture

7-layer residual MLP: Phoneme -> Syllable -> Morpheme -> Word -> Phrase -> Semantic -> Discourse

    h_0 = inp(x)
    h_{i+1} = h_i + MLP_i(h_i)      # residual transition
    pred_i = head_i(h_i)             # per-layer classification

D=32, vocab=[34,256,256,256,50,256,256], 72,340 parameters.

## Training

### Linguistic Classifier -- FARD-trained

- Corpus: corpus_v8c_remapped.ndjson (frequency-remapped class IDs)
- 20 epochs x 500 samples = 10,000 gradient steps
- Every epoch SHA-256 witnessed by FARD runtime
- No PyTorch anywhere in the training path
- Loss: 26.16 -> 1.93 (-93%)
- Weights: train/fard_trained_remapped.json

### Operation Sequencer -- FARD-trained

- Input: 25-dim feature vector -> op_logits[8] + tgt_logits[8]
- 20 epochs x 500 samples, no PyTorch
- Op accuracy: 99.3%, Tgt accuracy: 100%
- Weights: train/op_seq_fard_weights.json

## Benchmark Results

### Linguistic Classifier -- Honest Evaluation

Metric: balanced accuracy + macro-F1. Raw accuracy is misleading.
Corpus: corpus_v8c_remapped.ndjson
Vocab: [34,243,118,250,50,200,100] (exact corpus class counts)
Generate with: python3 src/remap_corpus.py

| Layer     | MajBase | BalAcc | MacroF1 | Status   |
|-----------|---------|--------|---------|----------|
| PHONEME   | 0.119   | 1.000  | 1.000   | STRONG   |
| SYLLABLE  | 0.011   | 0.963  | 0.951   | STRONG   |
| MORPHEME  | 0.028   | 0.950  | 0.937   | STRONG   |
| WORD      | 0.010   | 0.969  | 0.958   | STRONG   |
| PHRASE    | 0.032   | 0.299  | 0.294   | LEARNING |
| SEMANTIC  | 0.013   | 0.166  | 0.143   | LEARNING |
| DISCOURSE | 0.020   | 0.268  | 0.254   | LEARNING |

Note: L4 PHRASE reaches BalAcc=0.321 on independent CI set (STRONG threshold).

### Causal Intervention -- 7/7 LOCALIZED

Criterion: zeroing h_k and recomputing downstream damages L_k..L_6
(acc < 50% of baseline). L_0..L_{k-1} unaffected (within 0.05).
Confirmed on both test (1000) and CI (1000) independent held-out sets.

| Intervention | L0   | L1   | L2   | L3   | L4   | L5   | L6   | Result    |
|--------------|------|------|------|------|------|------|------|-----------|
| baseline     | 1.00 | 0.96 | 0.93 | 0.95 | 0.32 | 0.17 | 0.25 |           |
| zero h_0     | 0.05 | 0.00 | 0.01 | 0.00 | 0.02 | 0.01 | 0.01 | LOCALIZED |
| zero h_1     | 1.00 | 0.00 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | LOCALIZED |
| zero h_2     | 1.00 | 0.96 | 0.01 | 0.01 | 0.03 | 0.00 | 0.01 | LOCALIZED |
| zero h_3     | 1.00 | 0.96 | 0.93 | 0.01 | 0.02 | 0.01 | 0.01 | LOCALIZED |
| zero h_4     | 1.00 | 0.96 | 0.93 | 0.95 | 0.02 | 0.00 | 0.02 | LOCALIZED |
| zero h_5     | 1.00 | 0.96 | 0.93 | 0.95 | 0.32 | 0.00 | 0.02 | LOCALIZED |
| zero h_6     | 1.00 | 0.96 | 0.93 | 0.95 | 0.32 | 0.17 | 0.01 | LOCALIZED |

### Operation Sequencer -- FARD-trained

| Head           | Accuracy | Chance | Status |
|----------------|----------|--------|--------|
| Op prediction  | 0.993    | 0.125  | STRONG |
| Tgt prediction | 1.000    | 0.125  | STRONG |

## Tower

Both models run in tower_v8.fard (FARD_v0.5/programs/semantic_transformer/).
Tower root: da6407fbef1d5d3ae51a85352f272f56f81db641d5d732e7a2d46e5152f910a2

## What Was Fixed

Corpus degeneracy: Alphabetical word class ID assignment caused L3 WORD
to collapse to 2 classes after capping. Fix: frequency-remapped corpus
(src/remap_corpus.py).

Evaluation metric: Raw accuracy is misleading with high-cardinality label
spaces. Previous "7/7 STRONG" raw-accuracy result was majority-class
exploitation. Correct metric: balanced accuracy + macro-F1.

Causal intervention criterion: Zeroing h_k must recompute all downstream
hidden states from the zeroed value. Zeroing in place without recomputation
fails to propagate the intervention through the residual chain.

## What Remains

- L4 PHRASE / L5 SEMANTIC / L6 DISCOURSE are LEARNING not STRONG
- Causal intervention proof runs in Python/numpy, not FARD
- Corpus has only 250 unique words -- limited lexical diversity

## Repository Structure

    train/          model weights (JSON)
    benchmark/      evaluation results
    corpus/         corpus files (large files git-ignored)
    src/            training, evaluation, corpus scripts
    programs/       FARD programs
    data/           CSL inventories
