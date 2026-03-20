# CSL Stack Specification

## Architecture Principle

No layer may be decorative. Every layer must be a necessary information bottleneck.

## The Chain

    p -> s -> m -> w -> phi -> sigma -> delta

Each transition T_i:

    z_{i+1} = f_i(z_i, y_i)

NOT z_i = g_i(x) independently for every layer from the same seed.

## Layer Inventories

- L0: phoneme   (44 certified, Rust)
- L1: syllable  (423k certified, Rust-generated)
- L2: morpheme  (16 certified, needs expansion)
- L3: word      (34,487 certified)
- L4: phrase    (CSL JSON, tree-bearing)
- L5: semantic  (CSL JSON, graph, quotients over phrase)
- L6: discourse (CSL JSON, sequential state transitions)

## Corpus v8c Row Format

Each row contains derived objects at all 7 layers:

    layer_class_ids.L{i} = canonical class index under frozen inventory

Derivation trace: 7 steps
    START_PHONEME -> FORM_SYLLABLE -> MATCH_MORPHEME -> COMPOSE_WORD
    -> BUILD_PHRASE -> BUILD_SEMANTIC -> UPDATE_DISCOURSE

Plus: upstream_digests (all 7), trace_chain_hash, row_digest.

## Required Conditions for All-Strong LLR

1. Data is nested, not parallel — each field derived from previous
2. Each layer has sufficient entropy H(Y_i)
3. Morpheme is compositionally explicit — word = f(morphemes)
4. Phrase is tree-bearing, not template-bearing
5. Semantic quotients over phrase — multiple phrases map to same semantic
6. Discourse is sequential — d_t = T(d_{t-1}, sigma_t)
7. Deep supervision at every layer with consistency losses
8. Curriculum: train L0->L1, then L1->L2, ... then joint fine-tune

## Architecture

- Explicit per-layer hidden states h_0..h_6
- h_{i+1} = F_i(h_i, y_i)
- Probe L_i from h_i only, not from shared global state
- Information bottleneck c_i = B_i(h_i)
- Consistency loss C_i = CE(T_i(h_i_hat), y_{i+1})

## Formal Target

    for all i: delta_i = acc(L_i) - chance(L_i) >= 0.30

Not just two spikes. All layers strong.

## Schema

See spec/corpus_v8c_row.schema.json for the full JSON schema.
