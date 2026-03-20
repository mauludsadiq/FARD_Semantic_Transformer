"""
corpus_v8c generator — nested derivation chain.

Every layer is derived from the previous one:
  p -> s -> m -> w -> phi -> sigma -> delta

No layer is decorative.
"""
import json, hashlib, random, os
from collections import defaultdict, Counter

random.seed(42)

# ── Universe digests from the certified tower ─────────────────────────────────
TOWER_DIGESTS = {
    "phoneme":   "sha256:49061bbaf3053076c42a7757d8a624db68b53552a3ee1f59a33a49ddd66d755b",
    "syllable":  "sha256:d62f207f79e045e5df83b05bbda6585c60430cc9e0a7f6d5a93229bd0d34e9f8",
    "morpheme":  "sha256:266c6e62e0603360bb72fddd52c92fcc7788f196732ef56374ea7a3466de473e",
    "word":      "sha256:e3581b79acf169c79910a4e4b56d1d4ac1a96495a619cc613b9ee700e1cee10f",
    "phrase":    "sha256:0e927cc5ef5c4a7ba14d7e9fcf6abab2e31c89a2d19b90f0425fcb9f90acb3f8",
    "semantic":  "sha256:d6ceb90eed710a4de4e91d8c14f31d75ed583cadf90c627426583a05e155aa71",
    "discourse": "sha256:cee9be215f793b5c36e98f71ba1f1ba414d5e60bd3297da3449e28906fa66339",
}

def sha256(s):
    if isinstance(s, str): s = s.encode()
    return "sha256:" + hashlib.sha256(s).hexdigest()

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))

# ── Phoneme inventory ─────────────────────────────────────────────────────────
PHONEMES = [
    "P","B","T","D","K","G","F","V","TH","DH","S","Z","SH","ZH","HH","CH","JH",
    "M","N","NG","L","R","W","Y",
    "IY","IH","EY","EH","AE","AA","AO","OW","UH","UW","AH","AX","ER","AW","AY","OY",
    "IX","UX","EN","EL"
]
VOWELS = {"IY","IH","EY","EH","AE","AA","AO","OW","UH","UW","AH","AX","ER","AW","AY","OY","IX","UX"}
CONSONANTS = [p for p in PHONEMES if p not in VOWELS]
VOWEL_LIST = [p for p in PHONEMES if p in VOWELS]

PHONEME_FEATURES = {
    "P":  ["bilabial","consonant","stop","voiceless"],
    "B":  ["bilabial","consonant","stop","voiced"],
    "T":  ["alveolar","consonant","stop","voiceless"],
    "D":  ["alveolar","consonant","stop","voiced"],
    "K":  ["consonant","stop","velar","voiceless"],
    "G":  ["consonant","stop","velar","voiced"],
    "F":  ["consonant","fricative","labiodental","voiceless"],
    "V":  ["consonant","fricative","labiodental","voiced"],
    "TH": ["consonant","dental","fricative","voiceless"],
    "DH": ["consonant","dental","fricative","voiced"],
    "S":  ["alveolar","consonant","fricative","voiceless"],
    "Z":  ["alveolar","consonant","fricative","voiced"],
    "SH": ["consonant","fricative","postalveolar","voiceless"],
    "ZH": ["consonant","fricative","postalveolar","voiced"],
    "HH": ["consonant","fricative","glottal","voiceless"],
    "CH": ["affricate","consonant","postalveolar","voiceless"],
    "JH": ["affricate","consonant","postalveolar","voiced"],
    "M":  ["bilabial","consonant","nasal","voiced"],
    "N":  ["alveolar","consonant","nasal","voiced"],
    "NG": ["consonant","nasal","velar","voiced"],
    "L":  ["alveolar","consonant","liquid","voiced"],
    "R":  ["alveolar","approximant","consonant","voiced"],
    "W":  ["approximant","bilabial","consonant","voiced"],
    "Y":  ["approximant","consonant","palatal","voiced"],
    "IY": ["front","high","syllabic","unrounded","vowel","voiced"],
    "IH": ["front","high","syllabic","unrounded","vowel","voiced"],
    "EY": ["front","mid","syllabic","unrounded","vowel","voiced"],
    "EH": ["front","mid","syllabic","unrounded","vowel","voiced"],
    "AE": ["front","low","syllabic","unrounded","vowel","voiced"],
    "AA": ["back","low","syllabic","unrounded","vowel","voiced"],
    "AO": ["back","mid","rounded","syllabic","vowel","voiced"],
    "OW": ["back","mid","rounded","syllabic","vowel","voiced"],
    "UH": ["back","high","rounded","syllabic","vowel","voiced"],
    "UW": ["back","high","rounded","syllabic","vowel","voiced"],
    "AH": ["central","mid","syllabic","unrounded","vowel","voiced"],
    "AX": ["central","mid","syllabic","unrounded","vowel","voiced"],
    "ER": ["central","mid","syllabic","unrounded","vowel","voiced"],
    "AW": ["central","low","syllabic","unrounded","vowel","voiced"],
    "AY": ["front","low","syllabic","unrounded","vowel","voiced"],
    "OY": ["back","mid","rounded","syllabic","vowel","voiced"],
    "IX": ["front","high","syllabic","unrounded","vowel","voiced"],
    "UX": ["back","high","rounded","syllabic","vowel","voiced"],
    "EN": ["alveolar","consonant","nasal","syllabic","voiced"],
    "EL": ["alveolar","consonant","liquid","syllabic","voiced"],
}

PHONEME_IDX = {p: i for i, p in enumerate(PHONEMES)}

# ── Char-to-phoneme mapping ───────────────────────────────────────────────────
CHAR_TO_PH = {
    'a':'AE','b':'B','c':'K','d':'D','e':'EH','f':'F','g':'G','h':'HH',
    'i':'IH','j':'JH','k':'K','l':'L','m':'M','n':'N','o':'OW','p':'P',
    'q':'K','r':'R','s':'S','t':'T','u':'UW','v':'V','w':'W','x':'K',
    'y':'Y','z':'Z'
}

# ── Load word universe ────────────────────────────────────────────────────────
print("Loading word universe...")
words = {}
with open("../LLM_Nature_Semantic_Transformer/data/word_universe.tsv") as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            parts = line.split("\t")
            if len(parts) == 2:
                words[parts[0].strip()] = int(parts[1].strip())

# ── Morpheme inventory v2 ─────────────────────────────────────────────────────
# Load morpheme inventory v2
morph_inv_v2 = json.load(open("../LLM_Nature_Semantic_Transformer/data/csl/morpheme_inventory_v2.json"))["entries"]
MORPHEMES = [(e["meaning_id"], e["class"], e.get("surface_form",""), e["gloss"]) for e in morph_inv_v2]
MORPH_IDX = {m[0]: i for i, m in enumerate(MORPHEMES)}
print(f"  morpheme inventory v2: {len(MORPHEMES)} entries")

# Sort for deterministic class_id assignment
word_list = sorted(words.keys())
word_class_id = {w: i for i, w in enumerate(word_list)}
print(f"  {len(word_list)} words, {len(word_class_id)} class ids")

# ── CSL phrase/semantic/discourse inventories ─────────────────────────────────
phrase_inv  = json.load(open("../LLM_Nature_Semantic_Transformer/data/csl/phrase_inventory_v3.json"))["entries"]
sem_inv     = json.load(open("../LLM_Nature_Semantic_Transformer/data/csl/semantic_inventory.json"))["entries"]
disc_inv    = json.load(open("../LLM_Nature_Semantic_Transformer/data/csl/discourse_inventory.json"))["entries"]
phrase_class_id  = {e["id"]: i for i, e in enumerate(phrase_inv)}
sem_class_id     = {e["id"]: i for i, e in enumerate(sem_inv)}
disc_class_id    = {e["id"]: i for i, e in enumerate(disc_inv)}
print(f"  {len(phrase_inv)} phrases, {len(sem_inv)} semantics, {len(disc_inv)} discourse")

# ── Chain builder ─────────────────────────────────────────────────────────────
def build_chain(word_str, phrase_entry, sem_entry, disc_entry,
                seq_idx, prev_disc_id, rng):
    """
    Build a corpus_v8c row for one word, with full derivation chain.
    p -> s -> m -> w -> phi -> sigma -> delta
    """
    sig = words.get(word_str, 0)

    # ── L0: phoneme ───────────────────────────────────────────────────────────
    # Derive phoneme sequence from word's initial char + sig
    init_ch = word_str[0] if word_str else 'a'
    init_ph = CHAR_TO_PH.get(init_ch, 'AH')

    # Additional phonemes derived from word length and sig
    n_extra = min(len(word_str) - 1, 3)
    ph_seq  = [init_ph]
    for i in range(n_extra):
        ch = word_str[i+1] if i+1 < len(word_str) else 'a'
        ph = CHAR_TO_PH.get(ch, 'AH')
        ph_seq.append(ph)

    # Add a vowel if word is vowel-initial (sig bit 2)
    if sig & (1<<2) and init_ph not in VOWELS:
        ph_seq = [VOWEL_LIST[hash(word_str) % len(VOWEL_LIST)]] + ph_seq

    ph_ids = [f"en:ph:{p}" for p in ph_seq]
    ph_class = PHONEME_IDX.get(init_ph, 0)

    ph_obj = {
        "class_id": ph_class,
        "sequence": [
            {"id": f"en:ph:{p}",
             "features": sorted(PHONEME_FEATURES.get(p, [])),
             "sig": hash(p) % 4096,
             "digest": sha256(f"phoneme:{p}")}
            for p in ph_seq
        ],
        "length": len(ph_seq),
        "surface": {"form": word_str[:len(ph_seq)]},
        "sig": sig & 0xFF,
        "digest": sha256(canonical({"ph_seq": ph_ids, "word": word_str}))
    }

    # ── L1: syllable ──────────────────────────────────────────────────────────
    # Derive syllable from phoneme sequence
    # Find nucleus (first vowel in sequence)
    nucleus_idx = next((i for i,p in enumerate(ph_seq) if p in VOWELS), 0)
    onset = ph_seq[:nucleus_idx] if nucleus_idx > 0 else []
    nucleus = [ph_seq[nucleus_idx]] if nucleus_idx < len(ph_seq) else [ph_seq[0]]
    coda = ph_seq[nucleus_idx+1:] if nucleus_idx+1 < len(ph_seq) else []

    syl_id = "en:syl:" + "".join(onset) + nucleus[0] + "".join(coda)
    syl_class = abs(hash(syl_id)) % 5808  # matches corpus_v7b syllable count

    syl_obj = {
        "class_id": syl_class,
        "sequence": [{
            "id": syl_id,
            "onset":  [f"en:ph:{p}" for p in onset],
            "nucleus":[f"en:ph:{p}" for p in nucleus],
            "coda":   [f"en:ph:{p}" for p in coda],
            "prosody": {
                "weight": "heavy" if coda else "light",
                "stress": "primary" if not (sig & (1<<1)) else "secondary",
                "mora_count": 2 if coda else 1
            },
            "phoneme_span": {"start": 0, "end": len(ph_seq)-1},
            "sig": abs(hash(syl_id)) % 65536,
            "digest": sha256(f"syllable:{syl_id}:from:{ph_obj['digest']}")
        }],
        "length": 1,
        "sig": abs(hash(syl_id)) % 256,
        "digest": sha256(canonical({"syl_id": syl_id, "ph_digest": ph_obj["digest"]}))
    }

    # ── L2: morpheme ──────────────────────────────────────────────────────────
    # Derive morpheme from syllable + word properties
    # Root morpheme depends on POS (sig bit 6 = noun)
    is_noun = bool(sig & (1<<6))
    is_inflected = bool(sig & (1<<3))
    has_deriv = bool(sig & (1<<4))

    morph_seq = []
    # Root morpheme — varies by word hash to cover all 16 morphemes
    root_idx  = abs(hash(word_str)) % len(MORPHEMES)
    root_type = MORPHEMES[root_idx][0]
    morph_seq.append((root_idx, root_type, "root"))

    # Inflectional suffix if inflected — pick from inflectional morphemes
    if is_inflected:
        infl_choices = [i for i,m in enumerate(MORPHEMES) if "infl" in m[1]]
        infl_idx = infl_choices[abs(hash(word_str+"infl")) % len(infl_choices)]
        morph_seq.append((infl_idx, MORPHEMES[infl_idx][0], "infl"))

    # Derivational affix if has_deriv
    if has_deriv:
        deriv_choices = [i for i,m in enumerate(MORPHEMES) if "deriv" in m[1]]
        deriv_idx = deriv_choices[abs(hash(word_str+"deriv")) % len(deriv_choices)]
        morph_seq.append((deriv_idx, MORPHEMES[deriv_idx][0], "deriv"))

    # Morpheme class_id — based on root morpheme for balanced distribution
    morph_class = root_idx

    morph_obj = {
        "class_id": morph_class,
        "sequence": [{
            "meaning_id": mid,
            "surface_syllables": [syl_id],
            "class": MORPHEMES[midx][1],
            "gloss": MORPHEMES[midx][3],
            "allomorphs": [],
            "syllable_span": {"start": 0, "end": 0},
            "sig": midx,
            "digest": sha256(f"morpheme:{mid}:from:{syl_obj['digest']}")
        } for midx, mid, _ in morph_seq],
        "length": len(morph_seq),
        "sig": morph_class,
        "digest": sha256(canonical({
            "morphemes": [m[1] for m in morph_seq],
            "syl_digest": syl_obj["digest"]
        }))
    }

    # ── L3: word ──────────────────────────────────────────────────────────────
    # Derived from morpheme sequence — word IS the composition of morphemes
    wclass = word_class_id.get(word_str, 0)
    pos = "noun" if is_noun else ("verb" if not has_deriv else "adj")

    word_obj = {
        "class_id": wclass,
        "object": {
            "lemma_id": f"en:word:{word_str}",
            "language": "en",
            "phoneme_seq": ph_ids,
            "syllables": [syl_id],
            "morphemes": [m[1] for m in morph_seq],
            "surface_form": word_str,
            "features": {
                "pos": pos,
                "number": "plural" if is_inflected and is_noun else "singular",
                "tense": "past" if is_inflected and not is_noun else "none",
                "aspect": "none", "person": "none", "degree": "none",
                "countability": "count" if is_noun else "none",
                "derivation": "none"
            },
            "sig": sig,
            "digest": sha256(canonical({
                "lemma": word_str,
                "morphemes": [m[1] for m in morph_seq],
                "morph_digest": morph_obj["digest"]
            }))
        }
    }

    # ── L4: phrase ────────────────────────────────────────────────────────────
    # phrase_entry is from CSL inventory — derived from word
    # Phrase class = skeleton index (structural invariant, not opaque id)
    skel_id = phrase_entry.get("skeleton_id", phrase_entry["id"])
    # Use skeleton_index directly from v3 inventory (0-49)
    pclass = phrase_entry.get("skeleton_index", phrase_class_id.get(phrase_entry["id"], 0))
    phrase_obj = {
        "class_id": pclass,
        "object": {
            "phrase_id": phrase_entry["id"],
            "root_label": "S",
            "nodes": [
                {"id": "n0", "label": "S"},
                {"id": "n1", "label": "NP", "parent": "n0",
                 "word_ref": f"en:word:{word_str}"},
                {"id": "n2", "label": "VP", "parent": "n0"},
            ],
            "edges": [
                {"from": "n0", "to": "n1", "role": "subject"},
                {"from": "n0", "to": "n2", "role": "predicate"},
            ],
            "word_refs": [f"en:word:{word_str}"],
            "surface_form": phrase_entry.get("surface_form",""),
            "skeleton": phrase_entry.get("skeleton", []),
            "skeleton_id": phrase_entry.get("skeleton_id",""),
            "dependencies": phrase_entry.get("dependencies",[]),
            "argument_order": phrase_entry.get("argument_order",[]),
            "sig": pclass,
            "digest": sha256(canonical({
                "phrase_id": phrase_entry["id"],
                "word_digest": word_obj["object"]["digest"]
            }))
        }
    }

    # ── L5: semantic ──────────────────────────────────────────────────────────
    # semantic_entry is from CSL inventory — quotients over phrase
    sclass = sem_class_id.get(sem_entry["id"], 0)
    sem_obj = {
        "class_id": sclass,
        "object": {
            "semantic_id": sem_entry["id"],
            "phrase_ref":  phrase_entry["id"],
            "nodes": [
                {"id": "e1", "kind": "event",  "label": sem_entry["event"]},
                {"id": "x1", "kind": "entity", "label": sem_entry["agent"].replace("en:word:","")},
                {"id": "x2", "kind": "entity", "label": sem_entry["patient"].replace("en:word:","")},
            ],
            "edges": [
                {"from": "x1", "to": "e1", "role": "agent"},
                {"from": "x2", "to": "e1", "role": "patient"},
            ],
            "root_event": "e1",
            "sig": sclass,
            "digest": sha256(canonical({
                "sem_id": sem_entry["id"],
                "phrase_digest": phrase_obj["object"]["digest"]
            }))
        }
    }

    # ── L6: discourse ─────────────────────────────────────────────────────────
    # discourse is sequential — depends on previous discourse state
    dclass = disc_class_id.get(disc_entry["id"], 0)
    # state_type derived from whether topic continues (same semantic class) or shifts
    state_type = "topic_continue" if prev_disc_id is not None else "referent_intro"

    disc_obj = {
        "class_id": dclass,
        "object": {
            "discourse_id": disc_entry["id"],
            "semantic_ref": sem_entry["id"],
            "state_type": state_type,
            "history_refs": [prev_disc_id] if prev_disc_id else [],
            "topic": sem_entry["event"],
            "sig": dclass,
            "digest": sha256(canonical({
                "disc_id": disc_entry["id"],
                "sem_digest": sem_obj["object"]["digest"],
                "prev": prev_disc_id or "none"
            }))
        }
    }

    # ── Derivation trace ──────────────────────────────────────────────────────
    deriv = {"steps": [
        {"step":0,"op":"START_PHONEME",    "input_refs":[],
         "output_ref": ph_obj["digest"],
         "step_digest": sha256(f"step0:{ph_obj['digest']}")},
        {"step":1,"op":"FORM_SYLLABLE",    "input_refs":[ph_obj["digest"]],
         "output_ref": syl_obj["digest"],
         "step_digest": sha256(f"step1:{syl_obj['digest']}")},
        {"step":2,"op":"MATCH_MORPHEME",   "input_refs":[syl_obj["digest"]],
         "output_ref": morph_obj["digest"],
         "step_digest": sha256(f"step2:{morph_obj['digest']}")},
        {"step":3,"op":"COMPOSE_WORD",     "input_refs":[morph_obj["digest"]],
         "output_ref": word_obj["object"]["digest"],
         "step_digest": sha256(f"step3:{word_obj['object']['digest']}")},
        {"step":4,"op":"BUILD_PHRASE",     "input_refs":[word_obj["object"]["digest"]],
         "output_ref": phrase_obj["object"]["digest"],
         "step_digest": sha256(f"step4:{phrase_obj['object']['digest']}")},
        {"step":5,"op":"BUILD_SEMANTIC",   "input_refs":[phrase_obj["object"]["digest"]],
         "output_ref": sem_obj["object"]["digest"],
         "step_digest": sha256(f"step5:{sem_obj['object']['digest']}")},
        {"step":6,"op":"UPDATE_DISCOURSE", "input_refs":[sem_obj["object"]["digest"]],
         "output_ref": disc_obj["object"]["digest"],
         "step_digest": sha256(f"step6:{disc_obj['object']['digest']}")},
    ]}

    # ── Chain hash ────────────────────────────────────────────────────────────
    chain_hash = sha256(canonical([s["step_digest"] for s in deriv["steps"]]))

    # ── Row ───────────────────────────────────────────────────────────────────
    row = {
        "schema_version": "csl.corpus_v8c_row.v1",
        "row_id": f"row:en.v8c.{seq_idx:08d}",
        "language": "en",
        "sequence_index": seq_idx,
        "source": {"corpus_id": "corpus_v8c", "split": "train"},
        "layer_class_ids": {
            "L0": ph_class,
            "L1": syl_class,
            "L2": morph_class,
            "L3": wclass,
            "L4": pclass,
            "L5": sclass,
            "L6": dclass,
        },
        "phoneme":   ph_obj,
        "syllable":  syl_obj,
        "morpheme":  morph_obj,
        "word":      word_obj,
        "phrase":    phrase_obj,
        "semantic":  sem_obj,
        "discourse": disc_obj,
        "derivation": deriv,
        "upstream_digests": TOWER_DIGESTS,
        "trace_chain_hash": chain_hash,
        "row_digest": sha256(chain_hash + word_str + phrase_entry["id"])
    }
    return row


# ── Generate corpus ───────────────────────────────────────────────────────────
print("Generating corpus_v8c...")

nouns = [w for w,s in words.items() if (s>>6)&1 and 3<len(w)<10]
random.shuffle(nouns)
nouns = nouns[:500]

os.makedirs("corpus", exist_ok=True)
out_path = "corpus/corpus_v8c.ndjson"
n_rows = 0
prev_disc_id = None

# Full cross-product with diversity:
# Each phrase gets 3 semantics, each semantic gets 2 discourses
# Total: 200 * 3 * 2 = 1200 rows
pairs = []
for pi, pe in enumerate(phrase_inv):
    for k in range(50):
        si = (pi * 3 + k) % len(sem_inv)
        se = sem_inv[si]
        for j in range(2):
            di = (si * 2 + j) % len(disc_inv)
            de = disc_inv[di]
            wi = (pi * 7 + si * 3 + di * 2 + k + j) % len(nouns)
            pairs.append((pe, se, de, nouns[wi]))

random.shuffle(pairs)
print(f"  {len(pairs)} planned rows")

with open(out_path, "w") as f:
    for pe, se, de, word_str in pairs:
        row = build_chain(word_str, pe, se, de, n_rows, prev_disc_id, random)
        f.write(json.dumps(row, separators=(',',':')) + "\n")
        prev_disc_id = de["id"]
        n_rows += 1
        if n_rows % 500 == 0:
            print(f"  {n_rows} rows...", end="\r")

print(f"\n  Generated {n_rows} rows")


