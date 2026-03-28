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

### Linguistic Classifier -- FARD-trained (current)

- Corpus: corpus_sentences_remapped.ndjson (sentence-level, freq-remapped)
- 20 epochs x 500 samples = 10,000 gradient steps, lr=0.0001
- Every epoch SHA-256 witnessed by FARD runtime
- No PyTorch anywhere in the training path
- Loss: 38.11 -> 0.28 (-99%)
- Weights: train/fard_trained_sentences.json

### Linguistic Classifier -- PyTorch baseline (best accuracy)

- Corpus: corpus_sentences_remapped.ndjson
- 200 epochs x 15360 samples, LayerNorm + inter-layer consistency
- Weights: train/normed_consistency_weights.json

### Operation Sequencer -- FARD-trained

- Input: 25-dim feature vector -> op_logits[8] + tgt_logits[8]
- 20 epochs x 500 samples, no PyTorch
- Op accuracy: 99.3%, Tgt accuracy: 100%
- Weights: train/op_seq_fard_weights.json

## Benchmark Results

### Linguistic Classifier -- Honest Evaluation

Metric: balanced accuracy + macro-F1. Raw accuracy is misleading.
Corpus: corpus_sentences_remapped.ndjson (sentence-level, freq-remapped)
Vocab: [34,243,118,250,50,200,100]
Best model: train/normed_consistency_weights.json
Architecture: D=32, LayerNorm on all hidden states, inter-layer consistency loss

Phase 3 additions:
- LayerNorm: fixes hidden state norm explosion (4 -> 336 without it)
- Inter-layer consistency: auxiliary head on h_i predicts layer i+1 labels
- Contrastive loss on L4-L6: supervised contrastive (ablation, marginal gain)

| Layer     | MajBase | BalAcc | MacroF1 | Status | CI BalAcc |
|-----------|---------|--------|---------|--------|-----------|
| PHONEME   | 0.126   | 1.000  | 1.000   | STRONG | 1.000     |
| SYLLABLE  | 0.009   | 0.998  | 0.998   | STRONG | 0.997     |
| MORPHEME  | 0.022   | 0.996  | 0.996   | STRONG | 0.992     |
| WORD      | 0.008   | 0.998  | 0.998   | STRONG | 0.994     |
| PHRASE    | 0.027   | 0.960  | 0.959   | STRONG | 0.955     |
| SEMANTIC  | 0.010   | 0.970  | 0.968   | STRONG | 0.976     |
| DISCOURSE | 0.016   | 0.981  | 0.980   | STRONG | 0.986     |

### Causal Intervention -- 7/7 LOCALIZED (FARD-native, SHA-256 witnessed)

Program: programs/fard_transformer/causal_intervention_certified.fard
Runs entirely in FARD. Every result cryptographically receipted.
fard_run_digest: sha256:b5b7b5cb50a63f422979ff8735d7f815f220b64d982229c45806e0d96b0f6e40

Criterion: zeroing h_k and recomputing downstream damages L_k..L_6
(acc < 50% of baseline). L_0..L_{k-1} unaffected (within 0.05).
100 held-out samples.

| Intervention | L0   | L1   | L2   | L3   | L4   | L5   | L6   | Result    |
|--------------|------|------|------|------|------|------|------|-----------|
| baseline     | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.98 |           |
| zero h_0     | 0.10 | 0.01 | 0.01 | 0.01 | 0.02 | 0.00 | 0.01 | LOCALIZED |
| zero h_1     | 1.00 | 0.01 | 0.01 | 0.00 | 0.02 | 0.01 | 0.01 | LOCALIZED |
| zero h_2     | 1.00 | 1.00 | 0.02 | 0.01 | 0.03 | 0.00 | 0.01 | LOCALIZED |
| zero h_3     | 1.00 | 1.00 | 1.00 | 0.00 | 0.02 | 0.01 | 0.01 | LOCALIZED |
| zero h_4     | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 | 0.00 | 0.01 | LOCALIZED |
| zero h_5     | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.01 | LOCALIZED |
| zero h_6     | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.01 | LOCALIZED |

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
