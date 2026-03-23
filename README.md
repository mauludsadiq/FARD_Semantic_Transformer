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
Generate with: python3 src/remap_corpus.py

| Layer     | MajBase | BalAcc | MacroF1 | Status   |
|-----------|---------|--------|---------|----------|
| PHONEME   | 0.119   | 1.000  | 1.000   | STRONG   |
| SYLLABLE  | 0.011   | 0.823  | 0.788   | STRONG   |
| MORPHEME  | 0.028   | 0.680  | 0.659   | STRONG   |
| WORD      | 0.010   | 0.719  | 0.677   | STRONG   |
| PHRASE    | 0.032   | 0.113  | 0.109   | LEARNING |
| SEMANTIC  | 0.013   | 0.066  | 0.053   | LEARNING |
| DISCOURSE | 0.020   | 0.086  | 0.077   | LEARNING |

### Causal Intervention -- 7/7 LOCALIZED

Criterion: zeroing h_k and recomputing downstream damages L_k..L_6
(acc < 50% of baseline). L_0..L_{k-1} unaffected (within 0.05).
Confirmed on 1000 independent held-out samples.

| Intervention | L0   | L1   | L2   | L3   | L4   | L5   | L6   | Result    |
|--------------|------|------|------|------|------|------|------|-----------|
| baseline     | 1.00 | 0.81 | 0.68 | 0.72 | 0.11 | 0.07 | 0.11 |           |
| zero h_0     | 0.05 | 0.00 | 0.01 | 0.00 | 0.02 | 0.01 | 0.01 | LOCALIZED |
| zero h_1     | 1.00 | 0.01 | 0.01 | 0.00 | 0.02 | 0.01 | 0.01 | LOCALIZED |
| zero h_2     | 1.00 | 0.81 | 0.01 | 0.00 | 0.03 | 0.01 | 0.01 | LOCALIZED |
| zero h_3     | 1.00 | 0.81 | 0.68 | 0.01 | 0.02 | 0.01 | 0.02 | LOCALIZED |
| zero h_4     | 1.00 | 0.81 | 0.68 | 0.72 | 0.02 | 0.01 | 0.01 | LOCALIZED |
| zero h_5     | 1.00 | 0.81 | 0.68 | 0.72 | 0.11 | 0.01 | 0.01 | LOCALIZED |
| zero h_6     | 1.00 | 0.81 | 0.68 | 0.72 | 0.11 | 0.07 | 0.01 | LOCALIZED |

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
