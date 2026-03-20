"""
Map Universal Dependencies (UD) English EWT to CSL corpus format.

UD provides per-token:
  - FORM: surface word
  - UPOS: universal POS tag
  - DEPREL: dependency relation
  - HEAD: head token index

We map these to our 7-layer schema:
  L0 PHONEME   <- initial phoneme of word form
  L1 SYLLABLE  <- syllable structure from CMU dict or approximation
  L2 MORPHEME  <- morpheme class from word suffix + UPOS
  L3 WORD      <- word class index from our word universe
  L4 PHRASE    <- skeleton from dependency structure of sentence
  L5 SEMANTIC  <- predicate frame from UPOS + DEPREL pattern
  L6 DISCOURSE <- discourse state from sentence position + topic
"""
import json, re, os, math, hashlib, random
from collections import defaultdict, Counter

random.seed(42)

def sha256(s):
    if isinstance(s, str): s = s.encode()
    return "sha256:" + hashlib.sha256(s).hexdigest()

# ── Load inventories ──────────────────────────────────────────────────────────
phrase_inv  = json.load(open("data/csl/phrase_inventory_v3.json"))["entries"]
sem_inv     = json.load(open("data/csl/semantic_inventory_v3.json"))["entries"]
disc_inv    = json.load(open("data/csl/discourse_inventory_v3.json"))["entries"]

# Skeleton map: UD dependency patterns -> our 50 skeleton types
# Key insight: UD DEPREL gives us the argument structure directly
SKEL_MAP = {
    # root + nsubj + obj = transitive
    frozenset(["nsubj","obj"]):          "S_NP_VP_trans",
    frozenset(["nsubj"]):                "S_NP_VP_intrans",
    frozenset(["nsubj","iobj","obj"]):   "S_NP_VP_ditrans",
    frozenset(["nsubj:pass"]):           "S_NP_VP_passive",
    frozenset(["nsubj:pass","obl:agent"]):"S_NP_VP_passive",
    frozenset(["nsubj","obl"]):          "S_NP_VP_PP_vp",
    frozenset(["expl","nsubj"]):         "S_expletive_VP",
    frozenset(["nsubj","ccomp"]):        "S_NP_VP_SCOMP",
    frozenset(["nsubj","xcomp"]):        "S_NP_VP_INF",
    frozenset(["nsubj","advcl"]):        "S_NP_VP_MANNER",
    frozenset(["cop","nsubj"]):          "S_NP_be_AP",
    frozenset(["cop","nsubj","obl"]):    "S_NP_be_PP",
}
SKEL_ORDER = [e.get("skeleton_id","") for e in phrase_inv]
SKEL_IDX   = {s:i for i,s in enumerate(SKEL_ORDER) if s}

# Semantic frame map: UPOS -> predicate_frame
UPOS_TO_FRAME = {
    "VERB": None,  # determined by deprels
    "NOUN": "existential",
    "ADJ":  "copular",
    "ADP":  "locative",
}
DEPREL_TO_FRAME = {
    frozenset(["nsubj","obj"]):       "transitive",
    frozenset(["nsubj"]):             "intransitive",
    frozenset(["nsubj","iobj","obj"]):"ditransitive",
    frozenset(["nsubj:pass"]):        "passive",
    frozenset(["cop","nsubj"]):       "copular",
    frozenset(["expl"]):              "existential",
    frozenset(["nsubj","xcomp"]):     "causative",
    frozenset(["nsubj","obl"]):       "locative",
}

FRAME_IDX = {e["predicate_frame"]: None for e in sem_inv}
SEM_LOOKUP = {}  # (frame, tense, polarity) -> class_index
for e in sem_inv:
    SEM_LOOKUP[(e["predicate_frame"], e["tense"], e["polarity"])] = e["class_index"]

DISC_LOOKUP = {}  # (state_type, topic_continuity) -> class_index
for e in disc_inv:
    DISC_LOOKUP[(e["state_type"], e["topic_continuity"])] = e["class_index"]

# ── Phoneme/syllable approximation ───────────────────────────────────────────
CHAR_TO_PH = {'a':'AE','b':'B','c':'K','d':'D','e':'EH','f':'F','g':'G','h':'HH',
              'i':'IH','j':'JH','k':'K','l':'L','m':'M','n':'N','o':'OW','p':'P',
              'q':'K','r':'R','s':'S','t':'T','u':'UW','v':'V','w':'W','x':'K',
              'y':'Y','z':'Z'}
PHONEMES   = ["P","B","T","D","K","G","F","V","TH","DH","S","Z","SH","ZH","HH","CH","JH",
              "M","N","NG","L","R","W","Y","IY","IH","EY","EH","AE","AA","AO","OW","UH",
              "UW","AH","AX","ER","AW","AY","OY","IX","UX","EN","EL"]
PHONEME_IDX = {p:i for i,p in enumerate(PHONEMES)}
VOWELS     = {"IY","IH","EY","EH","AE","AA","AO","OW","UH","UW","AH","AX","ER","AW","AY","OY","IX","UX"}

def word_to_ph_class(form):
    ch = form[0].lower() if form else 'a'
    ph = CHAR_TO_PH.get(ch, 'AH')
    return PHONEME_IDX.get(ph, 0)

def word_to_syl_class(form):
    form = form.lower()
    ph_seq = [CHAR_TO_PH.get(c,'AH') for c in form[:4] if c in CHAR_TO_PH]
    if not ph_seq: return 0
    nidx = next((i for i,p in enumerate(ph_seq) if p in VOWELS), 0)
    onset  = ph_seq[:nidx]
    nuc    = [ph_seq[nidx]] if nidx < len(ph_seq) else [ph_seq[0]]
    coda   = ph_seq[nidx+1:]
    syl_id = "".join(onset)+nuc[0]+"".join(coda)
    return abs(hash(syl_id)) % 5808

def word_to_morph_class(form, upos):
    form = form.lower()
    # Suffix-based morpheme class
    if form.endswith("ing"):   return 2   # progressive
    if form.endswith("ed"):    return 6   # past
    if form.endswith("tion"):  return 10  # nominalizer
    if form.endswith("ness"):  return 11  # nominalizer
    if form.endswith("ly"):    return 12  # adverbial
    if form.endswith("er") and upos=="NOUN": return 8   # agent
    if form.endswith("or") and upos=="NOUN": return 9   # agent
    if form.endswith("able"):  return 30  # adj suffix
    if form.endswith("ful"):   return 36  # adj suffix
    if form.endswith("less"):  return 37  # adj suffix
    if form.endswith("ize"):   return 48  # verbal
    if form.startswith("un"):  return 5   # negation
    if form.startswith("re"):  return 64  # directional
    # Root class by initial phoneme
    ph = CHAR_TO_PH.get(form[0] if form else 'a', 'AH')
    return PHONEME_IDX.get(ph, 0) % 140

def word_to_word_class(form):
    # Use hash mod large number — approximates our word_class_id
    return abs(hash(form.lower())) % 34455

# ── Parse CoNLL-U ─────────────────────────────────────────────────────────────
def parse_conllu(path):
    sentences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("#"): continue
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) < 10: continue
            if "." in parts[0] or "-" in parts[0]: continue  # skip MWT
            current.append({
                "id":     int(parts[0]),
                "form":   parts[1],
                "lemma":  parts[2],
                "upos":   parts[3],
                "xpos":   parts[4],
                "feats":  parts[5],
                "head":   int(parts[6]) if parts[6].isdigit() else 0,
                "deprel": parts[7],
                "misc":   parts[9],
            })
    if current: sentences.append(current)
    return sentences

# ── Derive layer labels from sentence ─────────────────────────────────────────
def get_tense(feats):
    if "Tense=Past" in feats:   return "past"
    if "Tense=Pres" in feats:   return "present"
    if "Tense=Fut"  in feats:   return "future"
    return "present"

def get_polarity(feats, deprels):
    if "Polarity=Neg" in feats: return "negative"
    if "advmod" in deprels and any(d == "advmod" for d in deprels): return "affirmative"
    return "affirmative"

def sent_to_phrase_class(tokens):
    root = next((t for t in tokens if t["deprel"] == "root"), None)
    if not root: return 0
    root_id = root["id"]
    dep_rels = frozenset(t["deprel"] for t in tokens if t["head"] == root_id)
    has_nsubj     = bool(dep_rels & {"nsubj","nsubj:pass"})
    has_obj       = "obj" in dep_rels
    has_iobj      = "iobj" in dep_rels
    has_pass      = "nsubj:pass" in dep_rels
    has_cop       = "cop" in dep_rels
    has_expl      = "expl" in dep_rels
    has_csubj     = "csubj" in dep_rels
    has_obl       = bool(dep_rels & {"obl","obl:agent"})
    has_xcomp     = "xcomp" in dep_rels
    has_ccomp     = "ccomp" in dep_rels
    has_advcl     = "advcl" in dep_rels
    has_advmod    = "advmod" in dep_rels
    has_amod      = "amod" in dep_rels
    has_nmod      = "nmod" in dep_rels

    if has_expl:              return SKEL_IDX.get("S_expletive_VP", 0)
    if has_cop and has_nsubj and has_obl: return SKEL_IDX.get("S_NP_be_PP", 0)
    if has_cop and has_nsubj: return SKEL_IDX.get("S_NP_be_AP", 0)
    if has_pass:              return SKEL_IDX.get("S_NP_VP_passive", 0)
    if has_csubj:             return SKEL_IDX.get("S_SCOMP_VP", 0)
    if has_nsubj and has_iobj and has_obj: return SKEL_IDX.get("S_NP_VP_ditrans", 0)
    if has_nsubj and has_obj and has_xcomp: return SKEL_IDX.get("S_NP_VP_NP_INF", 0)
    if has_nsubj and has_obj and has_ccomp: return SKEL_IDX.get("S_NP_VP_SCOMP", 0)
    if has_nsubj and has_obj and has_obl:   return SKEL_IDX.get("S_NP_VP_PP_vp", 0)
    if has_nsubj and has_obj and has_advcl: return SKEL_IDX.get("S_NP_VP_MANNER", 0)
    if has_nsubj and has_obj:               return SKEL_IDX.get("S_NP_VP_trans", 0)
    if has_nsubj and has_xcomp:            return SKEL_IDX.get("S_NP_VP_INF", 0)
    if has_nsubj and has_ccomp:            return SKEL_IDX.get("S_NP_VP_SCOMP", 0)
    if has_nsubj and has_obl:              return SKEL_IDX.get("S_NP_VP_PP_vp", 0)
    if has_nsubj and has_advcl:            return SKEL_IDX.get("S_NP_VP_MANNER", 0)
    if has_nsubj and has_advmod:           return SKEL_IDX.get("S_NP_VP_MANNER", 0)
    if has_nsubj:                          return SKEL_IDX.get("S_NP_VP_intrans", 0)
    if has_obj and has_obl:                return SKEL_IDX.get("VP_v_np_pp", 0)
    if has_obj:                            return SKEL_IDX.get("VP_v_np", 0)
    if has_obl:                            return SKEL_IDX.get("VP_v_pp", 0)
    if has_amod and has_nmod:              return SKEL_IDX.get("NP_det_adj_n", 0)
    if has_amod:                           return SKEL_IDX.get("NP_det_adj_n", 0)
    if has_nmod:                           return SKEL_IDX.get("NP_det_n_pp", 0)
    return SKEL_IDX.get("S_NP_VP_intrans", 0)

def sent_to_sem_class(tokens):
    root = next((t for t in tokens if t["deprel"] == "root"), None)
    if not root: return 0
    root_id  = root["id"]
    dep_rels = frozenset(t["deprel"] for t in tokens if t["head"] == root_id)
    feats    = root.get("feats","")
    tense    = get_tense(feats)
    polarity = get_polarity(feats, dep_rels)
    has_nsubj = bool(dep_rels & {"nsubj","nsubj:pass"})
    has_obj   = "obj" in dep_rels
    has_iobj  = "iobj" in dep_rels
    has_pass  = "nsubj:pass" in dep_rels
    has_cop   = "cop" in dep_rels
    has_expl  = "expl" in dep_rels
    has_xcomp = "xcomp" in dep_rels
    has_obl   = bool(dep_rels & {"obl","obl:agent","obl:tmod"})
    if has_expl:                    frame = "existential"
    elif has_cop:                   frame = "copular"
    elif has_pass:                  frame = "passive"
    elif has_nsubj and has_iobj and has_obj: frame = "ditransitive"
    elif has_nsubj and has_obj and has_xcomp: frame = "causative"
    elif has_nsubj and has_obj:     frame = "transitive"
    elif has_nsubj and has_obl:     frame = "locative"
    elif has_nsubj:                 frame = "intransitive"
    elif has_obj:                   frame = "transitive"
    else:                           frame = "intransitive"
    return SEM_LOOKUP.get((frame, tense, polarity), 0)

def sent_to_disc_class(sent_idx, total_sents, prev_root_lemma, curr_root_lemma):
    # Approximate discourse state from position and topic continuity
    if sent_idx == 0:
        state = "referent_intro"
    elif sent_idx == total_sents - 1:
        state = "elaboration"
    elif sent_idx % 7 == 0:
        state = "topic_shift"
    else:
        state = "topic_continue"

    if prev_root_lemma == curr_root_lemma:
        topic = "same_topic"
    elif prev_root_lemma and curr_root_lemma and prev_root_lemma[0] == curr_root_lemma[0]:
        topic = "related_topic"
    else:
        topic = "new_topic"

    return DISC_LOOKUP.get((state, topic), 0)

# ── Main conversion ───────────────────────────────────────────────────────────
print("Parsing UD English EWT...")
train_sents = parse_conllu("data/ud_english/en_ewt-ud-train.conllu")
dev_sents   = parse_conllu("data/ud_english/en_ewt-ud-dev.conllu")
all_sents   = train_sents + dev_sents
print(f"  {len(train_sents)} train + {len(dev_sents)} dev = {len(all_sents)} total sentences")

# Filter: 4-12 words, has a root verb
def usable(sent):
    if not 4 <= len(sent) <= 12: return False
    if not any(t["deprel"]=="root" for t in sent): return False
    if not any(t["upos"] in ("VERB","AUX") for t in sent): return False
    return True

usable_sents = [s for s in all_sents if usable(s)]
print(f"  {len(usable_sents)} usable sentences (4-12 words, has root verb)")

T = 8
out_rows = []
prev_root_lemma = None

for si, sent in enumerate(usable_sents):
    # Pad or trim to T tokens
    tokens = sent[:T]
    while len(tokens) < T:
        tokens.append(tokens[-1])  # repeat last token for padding

    # Sentence-level labels
    phr_class  = sent_to_phrase_class(sent)
    sem_class  = sent_to_sem_class(sent)
    root_lemma = next((t["lemma"] for t in sent if t["deprel"]=="root"), "")
    disc_class = sent_to_disc_class(si, len(usable_sents), prev_root_lemma, root_lemma)
    prev_root_lemma = root_lemma

    words = []
    for t in tokens:
        form = t["form"]
        words.append({
            "surface_form": form,
            "upos": t["upos"],
            "deprel": t["deprel"],
            "layer_class_ids": {
                "L0": word_to_ph_class(form),
                "L1": word_to_syl_class(form),
                "L2": word_to_morph_class(form, t["upos"]),
                "L3": word_to_word_class(form),
                "L4": phr_class,
                "L5": sem_class,
                "L6": disc_class,
            }
        })

    out_rows.append({
        "sent_id":    si,
        "split":      "train" if si < len(train_sents) else "dev",
        "length":     len(sent),
        "phrase_skeleton": phr_class,
        "words":      words
    })

random.shuffle(out_rows)
os.makedirs("corpus", exist_ok=True)
with open("corpus/corpus_ud_ewt.ndjson","w") as f:
    for r in out_rows:
        f.write(json.dumps(r, separators=(',',':')) + "\n")

size_mb = os.path.getsize("corpus/corpus_ud_ewt.ndjson")/1024/1024
print(f"Written corpus/corpus_ud_ewt.ndjson ({size_mb:.1f} MB, {len(out_rows)} sentences)")

# Stats
from collections import Counter
import math
layer_names = ["PHONEME","SYLLABLE","MORPHEME","WORD","PHRASE","SEMANTIC","DISCOURSE"]
cc = [Counter() for _ in range(7)]
for row in out_rows:
    for w in row["words"]:
        for i in range(7): cc[i][w["layer_class_ids"][f"L{i}"]] += 1

print("\nCorpus stats:")
for i in range(7):
    n = len(cc[i]); total = sum(cc[i].values())
    H = -sum((c/total)*math.log2(c/total) for c in cc[i].values()) if n>1 else 0
    print(f"  L{i} {layer_names[i]:<12}: {n:5d} unique  H={H:.2f}/{math.log2(max(n,1)):.2f} bits")
