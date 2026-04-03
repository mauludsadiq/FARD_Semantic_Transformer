# FARD Semantic Transformer

A 7-layer linguistic hierarchy trained entirely in FARD — a deterministic,
content-addressed scripting language. Every gradient step produces a
SHA-256 receipt. The computation chain is auditable and reproducible.

## Architecture

### Current: Contextual Tower (best model)

Weights: train/contextual_tower_weights.json

Principled ambiguity in upper layers via explicit context variable:

    # Token scope (L0-L3): D=32, LayerNorm — deterministic
    h_0 = LayerNorm(inp(x))
    h_{i+1} = LayerNorm(h_i + MLP_i(h_i))    # i in 0..2
    pred_i = head_i(h_i)                       # word-level

    # Scope boundary: attention pooling
    u = sum_i softmax(a^T h_3^{(i)}) * h_3^{(i)}

    # Context injection
    uc = [u ; emb(C)]                          # C in 6 named contexts

    # Sentence scope (L4-L6): D=64, context-conditioned
    z_0 = LayerNorm(proj(uc))
    z_{j+1} = LayerNorm(z_j + MLP_j(z_j))
    pred_{4+j} = head_j(z_j)

101,902 parameters.

Context space: neutral, eventive, locative, descriptive, classificatory, embedded

Ambiguity structure:
  L0-L3: deterministic (token-level, H(L_i|X) = 0)
  L4:    deterministic given sentence (H(L4|X) = 0)
  L5/L6: context-conditioned (H(L6|X,C) = 0, H(L6|X) > 0)

  30% sentences: single context (deterministic upper labels)
  42% sentences: two contexts (principled L5/L6 ambiguity)
  28% sentences: three contexts
  H(L6|X) mean = 0.352

### Previous: Scope-Separated Tower

Weights: train/tower_upper_weights.json — 7/7 STRONG 1.000, 96,622 params

### Previous: Uniform Residual Tower

    h_0 = inp(x)
    h_{i+1} = h_i + MLP_i(h_i)
    pred_i = head_i(h_i)

D=32/64 tapered, vocab=[34,243,118,250,50,200,100]

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

### Causal Intervention -- Contextual Tower Causal Certification

Four claims certified on 240 held-out sentences:

**A. Token locality (4/4 LOCALIZED)**
zero h_k damages token layers k..3 and upper cascade.
Upstream token layers unaffected (within 0.05).

**B. Scope separation (3/3 LOCALIZED)**
zero z_k damages sentence layers k..2.
Token layers completely stable (1.000) -- scope boundary holds.

**C. Context isolation (CERTIFIED)**
Zeroing context embedding: L5 1.000→0.546, L6 0.746→0.346.
Token layers unchanged. Ambiguity lives in the context channel.

**D. Ambiguity control (20/37 sentences context-sensitive)**
Same X, different C → different L6 predictions.
Context causally controls the discourse layer.

Receipt: benchmark/contextual_tower_causal_cert.json

### Prior FARD-native CI Proof (older model)

Program: programs/fard_transformer/causal_intervention_certified.fard
fard_run_digest: sha256:b5b7b5cb50a63f422979ff8735d7f815f220b64d982229c45806e0d96b0f6e40
7/7 LOCALIZED on 100 held-out samples.

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

## Phase 5 — Tower-to-Sequencer Bridge

Tower hidden states export executable control information.

**Dataset:** data/op_seq_semantic.json (4752 instances)
Op labels depend on L4/L5/L6/context at blocks 4-8.
H(op|struct) = 0.78-1.37 at semantic blocks — not recoverable from structure alone.

**Three-way ablation:**

| Model | Op Acc | Semantic blocks 4-8 |
|-------|--------|---------------------|
| A: structural only (25-dim) | 0.892 | 0.65-0.89 — fails |
| B: tower only (32-dim) | 1.000 | 1.000 — resolves |
| C: bridged (57-dim) | 1.000 | 1.000 — perfect |

Tower contribution: op=+0.108

**Conclusion:** Upper-layer semantic/discourse state (L4/L5/L6 + context)
causally determines execution policy where structural scaffold is insufficient.
The tower is not just a classifier — it exports actionable control state.

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

### Completed
- L0-L3: 1.000 STRONG (token-level, deterministic)
- L4/L5/L6: STRONG under sentence-level supervision
- Causal intervention: FARD-native with SHA-256 receipts
- Contextual tower: 4-claim causal certification
- Phase 5 bridge: tower states proven to control execution policy

### Completed
- L0-L3: 1.000 STRONG (token-level, deterministic)
- L4/L5/L6: STRONG under sentence-level supervision
- Causal intervention: FARD-native with SHA-256 receipts
- Contextual tower: 4-claim causal certification
- Phase 5 bridge: tower states proven to control execution policy
- FARD training of contextual tower: 20 epochs, all SHA-256 witnessed
  - Correct attention pooling gradients (softmax jacobian path)
  - FARD strict shape enforcement caught incorrect gradient wiring

### Generative Phase (current)

The project has pivoted from classifier/analyzer to generator.

**Generation Pass 1: MiniGPT** (train/mini_gpt.pt)
- 6.27M params, D=256, 6 layers, 8 heads
- Primary: next-word prediction (autoregressive)
- Auxiliary: UPOS prediction
- Vocabulary: ~8K word forms from UD English EWT
- ep=5 perplexity: 181 (real sentence structure emerging)

**Generation Pass 2: HierarchicalGPT** (in progress)
- Same architecture + full hierarchy supervision
- Token-level: UPOS + XPOS + morphology heads
- Sentence-level: L4 (phrase) + L5 (semantic) + L6 (discourse) heads
- Tests whether explicit hierarchy improves generation over flat LM

### Completed (classifier/analyzer phase)
- Contextual tower: 7/7 STRONG, 4-claim causal certification
- UD English corpus: 9497 sentences, gold UD annotations
- FARD training: 20 epochs witnessed, correct attention pooling gradients
- Phase 5 bridge: tower states proven to control execution policy

### Open
- Compare Pass 1 vs Pass 2 perplexity
- Add subword tokenization (BPE) for better vocabulary coverage
- Scale to larger text corpus (10M+ tokens)
- FARD-witnessed generative path

## Repository Structure

    train/          model weights (JSON)
    benchmark/      evaluation results
    corpus/         corpus files (large files git-ignored)
    src/            training, evaluation, corpus scripts
    programs/       FARD programs
    data/           CSL inventories
