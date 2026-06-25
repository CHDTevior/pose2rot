### build_species_info_dict.py ###
"""
Build ``species_info_dict.npy`` for the MocapAnything (V2 pose2rot) zoo1030 dataset
from raw Truebones Zoo BVH files.

This is the per-species STATIC skeleton/graph/semantic table that the V2 training
loader and inference path read once and broadcast over every clip of a species.

================================================================================
WHO CONSUMES THIS FILE (verified line-by-line against repo, not assumed)
================================================================================
data/loader_v2.py
  L41-42  : np.load(os.path.join(os.path.dirname(bvh_dir), "species_info_dict.npy"),
                    allow_pickle=True).item()
            -> the saved object MUST be a python dict, keyed by the SHORT species
               name (the part before '#', e.g. "Spider", "Trex").
  L314    : info = self.species_info_dict[species_name]
  L316    : hop_mat  = info['joints_distance'].astype(np.int64)   # [J,J]
  L317    : edge_mat = info['joint_relation'].astype(np.int64)    # [J,J]
  L318    : joint_t5embed = info['t5_embedding'].astype(np.float32)# [J,Dt5]
  L320    : J = hop_mat.shape[0]   -> J is DEFINED by joints_distance.shape[0]
                                      (UNPADDED, == the species' real joint count)
  L323    : static_rot_joint_ids = info['static_rot_joints']      # index array -> mask
  L328    : static_pos_joint_ids = info['static_joints']          # index array -> mask
inference/video2pose2rot.py
  L214-216: graph_hop/graph_edge/joint_t5embed read identically.
  L218-222: static_rot_joints / static_joints used as fancy-index into a (J,) mask.
  L158-160: parents / rest_pose come from BVHReader(res) PER-CLIP, NOT from this dict.
            -> This dict does NOT need to provide parents/rest_pose for the model to
               run; the loader rebuilds them per clip. We STILL emit 'parents' and
               'rest_pose' (task requirement, and useful for QA / tooling), but they
               are advisory: at runtime the per-clip BVH is authoritative.

Embedding-size constraints that bound the integer matrices (hard limits):
models/v2/pose2rot/graph_attention.py
  L34 : nn.Embedding(max_path_len + 1, d_model)  with max_path_len=5
        -> 'joints_distance' values MUST be in [0, 5]. (6 hop buckets: 0..5)
  L35 : nn.Embedding(6, d_model)
        -> 'joint_relation' values MUST be in [0, 5]. (6 edge types)
  L73 : num_edge_types = 6 (confirms the 6-way edge typing).
data/loader_v2.py collate (padding fill values, informative):
  L701: hop padded with 5   (so 5 == "far / unreachable" bucket)
  L702: edge padded with 4  (so 4 is the value collate uses for padded/no-relation)

models/v2/pose2rot/model.py
  L22/L107/L253/L380: joint_t5proj = nn.Linear(joint_embed_dim, q_dim),
                      joint_embed_dim default 768
  -> 't5_embedding' last dim MUST be 768. t5-base d_model == 768 (verified).

================================================================================
JOINT-ORDER CONTRACT (critical, verified)
================================================================================
The loader's per-clip parents/offsets come from utils/bvh_reader.py -> utils/bvh.load(),
which lists joints in BVH file DECLARATION order (ROOT first, then each JOINT as it
appears, DFS as written). root parent == -1 (verified on Cat/CAT_TPOSE.bvh:
parents[:6]=[-1,0,1,2,3,4]). We build THIS dict from the SAME utils.bvh.load() so the
index ordering of joints_distance/joint_relation/t5_embedding/static_* aligns
index-for-index with the per-clip parents_a/offset_a the loader produces.

================================================================================
PER-SPECIES J INCONSISTENCY (a real data hazard; we fail loud, not paper over)
================================================================================
Some species contain BVH files with DIFFERENT joint counts (e.g. Trex: 49 vs 61,
Ant: 34 vs 41, Centipede: 64 vs 84 -- usually a TPOSE/STILL rig vs the animated rig).
The loader pulls a SINGLE (J,J) graph per species but uses each clip's own J for
slicing (collate: hop_pad[:J,:J] = hop). If a clip's J != this dict's J the collate
will raise a shape error. We therefore:
  - choose the species representative as the BVH whose joint count == the species
    MODAL joint count (ties -> prefer a TPOSE/rest file, else the lexicographically
    first), and
  - print a loud per-species J histogram so off-J clips can be filtered downstream.
We do NOT silently remap/truncate skeletons.

================================================================================
EXACT OUTPUT SCHEMA  (np.save(..., allow_pickle=True) of one python dict)
================================================================================
species_info_dict.npy  ->  dict[str(short_species_name) -> dict] with keys:

  'parents'         : np.ndarray  shape (J,)    dtype int64
                      BVH parent index per joint; root == -1. Advisory (loader
                      rebuilds per clip); index order == this dict's joint order.
  'rest_pose'       : np.ndarray  shape (J, 3)  dtype float32
                      anim.offsets of the representative skeleton (local bone
                      offsets, RAW units -- NOT normalized; loader normalizes
                      per clip with its own global_scale). Advisory.
  'joints_distance' : np.ndarray  shape (J, J)  dtype int64    values in [0,5]
                      undirected tree BFS hop distance, clamped to 5. diag == 0.
  'joint_relation'  : np.ndarray  shape (J, J)  dtype int64    values in [0,5]
                      pairwise edge TYPE (see ENCODING below).
  't5_embedding'    : np.ndarray  shape (J, 768) dtype float32
                      per-joint-name T5-base encoder embedding, mean-pooled over
                      the (mask-weighted) token sequence of the joint name string.
  'static_rot_joints': np.ndarray shape (S_r,)  dtype int64  (may be empty)
                      joint indices whose rotation is held to the reference frame
                      (used by loader L323 to build static_rot_joint_mask).
  'static_joints'   : np.ndarray  shape (S_p,)  dtype int64  (may be empty)
                      joint indices treated as static for position
                      (used by loader L328 to build static_pos_joint_mask).
  'joint_names'     : list[str]   length J      (extra; not read by model, kept
                      for QA / debugging / npy2bvh tooling. Harmless to consumers.)

  J == number of joints of the representative skeleton (UNPADDED). 25 <= J <= 143
  across the 74 species; MAX_JOINTS=150 padding is applied later by the loader's
  collate, NOT here.

================================================================================
ENCODING CONVENTIONS WE DEFINE (repo does NOT specify these; flagged for review)
================================================================================
joint_relation (edge TYPE), value in {0..5}:
    0 = self            (i == j)
    1 = parent          (j is the direct parent of i)
    2 = child           (j is a direct child of i)
    3 = sibling         (i,j share the same parent, i != j)
    4 = none/other      (any other pair: ancestor>1, cousin, cross-limb, ...)
    5 = (reserved/unused here)
  NOTE: collate pads edges with value 4 (L702), i.e. "none/other" -- consistent
  with our choice that 4 is the generic no-special-relation class. We leave 5
  unused on real joints (room for a future type without colliding with padding).
  >>> This 5-way-used / 6-bucket mapping is OUR convention. If the original
      MocapAnything used a different ordering (e.g. parent and child collapsed
      into one "adjacent" class), retraining-from-scratch is unaffected (the
      embeddings are learned), but loading PRETRAINED V2 weights would require
      matching their exact code. Flagged for codex/user confirmation.

joints_distance (hop), value in {0..5}:
    undirected shortest path in the kinematic tree, min(dist, 5). diag 0.
  Matches max_path_len=5 (6 buckets) and collate's pad value 5 ("far").

static_rot_joints / static_joints:
    DEFAULT = empty arrays (no joint is forced static). The repo gives no rule
    for which joints are static, and forcing e.g. fingertip/end-effector joints
    static is a modeling choice. Empty is the safe, behavior-preserving default:
    loader builds an all-False mask, model.py L617-619 then leaves all predicted
    rotations untouched. >>> If a static policy is desired (e.g. mark leaf "Eye"/
    "Tongue"/End-Site-adjacent joints), expose it via --static-policy; flagged
    for codex/user. We implement a couple of opt-in policies but default to none.

================================================================================
WHAT WE DO NOT DO HERE (out of scope)
================================================================================
- No rendering, no training, no per-clip pose extraction (that's extract_bvh_pose.py).
- No scale cache (that's a separate __mesh2pose1002_species_scale_cache.pkl).
- No memory pkl (that's species_fps_memory.py).
This script ONLY builds the static per-species table.
"""

import argparse
import json
import os
import sys
from collections import deque

import numpy as np

# Make the MocapAnything repo importable so we reuse its EXACT bvh parser
# (guarantees identical joint ordering to the training loader).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT_DEFAULT = os.path.abspath(
    os.path.join(_THIS_DIR, "..", "..", "MocapAnything")
)


# =========================================================
# Edge-type / hop encoding constants (see module docstring)
# =========================================================
EDGE_SELF = 0
EDGE_PARENT = 1
EDGE_CHILD = 2
EDGE_SIBLING = 3
EDGE_OTHER = 4
EDGE_RESERVED = 5  # intentionally unused on real joints

MAX_HOP = 5  # clamp, matches graph_attention max_path_len=5 (6 buckets 0..5)
T5_DIM = 768  # t5-base d_model; MUST equal model.py joint_embed_dim default


# =========================================================
# BVH discovery
# =========================================================
def list_species_dirs(raw_root):
    """Each immediate sub-directory of raw_root that contains >=1 .bvh is a species."""
    species = []
    for name in sorted(os.listdir(raw_root)):
        d = os.path.join(raw_root, name)
        if not os.path.isdir(d):
            continue
        has_bvh = any(f.lower().endswith(".bvh") for f in os.listdir(d))
        if has_bvh:
            species.append(name)
    return species


def list_species_bvhs(species_dir):
    return sorted(
        os.path.join(species_dir, f)
        for f in os.listdir(species_dir)
        if f.lower().endswith(".bvh")
    )


def _is_rest_like(path):
    """Heuristic: prefer TPOSE / rest / still files as the representative skeleton."""
    b = os.path.basename(path).lower()
    return ("tpose" in b) or ("t-pose" in b) or ("rest" in b) or ("still" in b)


# =========================================================
# Skeleton parsing (delegate to repo's utils.bvh.load for ordering fidelity)
# =========================================================
def parse_skeleton(bvh_path, bvh_module):
    """
    Returns (parents:int64[J], offsets:float32[J,3], names:list[str] len J).
    Joint order == utils.bvh.load() == BVH file declaration order (root parent -1).
    """
    anim, names, _frametime = bvh_module.load(bvh_path)
    parents = np.asarray(anim.parents, dtype=np.int64).reshape(-1)
    offsets = np.asarray(anim.offsets, dtype=np.float32)
    names = list(names)
    J = parents.shape[0]
    if offsets.shape != (J, 3):
        raise ValueError(
            f"{bvh_path}: offsets shape {offsets.shape} != ({J}, 3)"
        )
    if len(names) != J:
        raise ValueError(
            f"{bvh_path}: len(names)={len(names)} != J={J}"
        )
    return parents, offsets, names


def choose_representative(bvh_paths, bvh_module):
    """
    Pick the representative BVH whose joint count == the species MODAL joint count.
    Returns (rep_path, modal_J, j_histogram dict[J -> count]).
    Fails loud (raises) if no BVH parses.
    """
    j_of = {}
    for p in bvh_paths:
        try:
            parents, _off, _nm = parse_skeleton(p, bvh_module)
            j_of[p] = int(parents.shape[0])
        except Exception as e:  # noqa: BLE001 - report and skip unparsable file
            print(f"    [WARN] failed to parse {os.path.basename(p)}: {e}")
    if not j_of:
        raise RuntimeError("no parseable BVH in this species")

    hist = {}
    for j in j_of.values():
        hist[j] = hist.get(j, 0) + 1
    # modal J: highest count, tie -> larger J (the animated rig usually has more joints)
    modal_J = sorted(hist.items(), key=lambda kv: (kv[1], kv[0]))[-1][0]

    candidates = [p for p, j in j_of.items() if j == modal_J]
    # prefer a rest/TPOSE file among modal-J candidates, else lexicographically first
    rest_cands = [p for p in candidates if _is_rest_like(p)]
    rep = sorted(rest_cands)[0] if rest_cands else sorted(candidates)[0]
    return rep, modal_J, hist


# =========================================================
# Graph features: hop distance + edge type
# =========================================================
def build_adjacency(parents):
    """Undirected adjacency list of the kinematic tree."""
    J = parents.shape[0]
    adj = [[] for _ in range(J)]
    for i in range(J):
        p = int(parents[i])
        if p >= 0:
            adj[i].append(p)
            adj[p].append(i)
    return adj


def build_hop_matrix(parents, max_hop=MAX_HOP):
    """
    Undirected BFS shortest-path hop count between every joint pair, clamped to
    [0, max_hop]. Disconnected pairs (shouldn't happen in a tree, but defensive)
    -> max_hop. dtype int64, shape (J, J).
    """
    J = parents.shape[0]
    adj = build_adjacency(parents)
    hop = np.full((J, J), max_hop, dtype=np.int64)
    for src in range(J):
        hop[src, src] = 0
        seen = {src}
        dq = deque([(src, 0)])
        while dq:
            node, d = dq.popleft()
            if d >= max_hop:
                # neighbours would be >max_hop; they stay clamped at max_hop
                continue
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    hop[src, nb] = d + 1
                    dq.append((nb, d + 1))
    return hop


def build_edge_matrix(parents):
    """
    Pairwise edge TYPE in {0..5}; see module docstring for the encoding.
      0 self, 1 parent(j of i), 2 child(j of i), 3 sibling, 4 other, 5 reserved.
    dtype int64, shape (J, J).
    """
    J = parents.shape[0]
    edge = np.full((J, J), EDGE_OTHER, dtype=np.int64)

    # children lists for sibling detection
    children = [[] for _ in range(J)]
    for i in range(J):
        p = int(parents[i])
        if p >= 0:
            children[p].append(i)

    for i in range(J):
        edge[i, i] = EDGE_SELF
        pi = int(parents[i])
        # parent of i
        if pi >= 0:
            edge[i, pi] = EDGE_PARENT
            edge[pi, i] = EDGE_CHILD  # symmetric counterpart
        # siblings: share parent pi, exclude self
        if pi >= 0:
            for s in children[pi]:
                if s != i and edge[i, s] == EDGE_OTHER:
                    edge[i, s] = EDGE_SIBLING
    return edge


# =========================================================
# Static joint policy
# =========================================================
def build_static_joints(parents, names, policy, species=None, geo_map=None):
    """
    Return (static_rot_joints int64[Sr], static_joints int64[Sp]).
    policy:
      'none'  -> both empty (DEFAULT; behavior-preserving, model leaves all rots free)
      'leaves'-> mark leaf joints (no children) as static for ROTATION only
                 (position static stays empty). Opt-in; flagged for review.
      'geo'   -> read per-species static_rot_joints from a codex-verified JSON map
                 (per-clip worst-case-ref geodesic <2deg "empirically static" rot
                 joints). Position static stays empty. Needs `species` + `geo_map`.
    """
    J = parents.shape[0]
    empty = np.zeros((0,), dtype=np.int64)

    if policy == "none":
        return empty.copy(), empty.copy()

    if policy == "leaves":
        has_child = np.zeros((J,), dtype=bool)
        for i in range(J):
            p = int(parents[i])
            if p >= 0:
                has_child[p] = True
        leaves = np.where(~has_child)[0].astype(np.int64)
        # static ROTATION on leaves; position static left empty (positions of
        # leaves still vary, so we do NOT mark them position-static).
        return leaves, empty.copy()

    if policy == "geo":
        if geo_map is None:
            raise ValueError("static policy 'geo' requires geo_map (load it from --static-geo-json)")
        if species not in geo_map:
            # Distinguish 'author-truth genuinely empty' (species present, []) from
            # 'species missing from the map' (we cannot infer -> WARN + empty).
            print(f"[WARN] static_policy=geo: species '{species}' not in geo map -> empty static_rot")
            return empty.copy(), empty.copy()
        sr = np.array(geo_map[species], dtype=np.int64)
        # fail-loud: an out-of-range index means species/J misalignment; crash
        # rather than silently masking a wrong joint.
        if sr.size and (sr.min() < 0 or sr.max() >= J):
            raise ValueError(
                f"static_policy=geo: species '{species}' has static_rot index "
                f"out of [0,{J}) (J={J}); indices={sr.tolist()}"
            )
        sr = np.unique(sr).astype(np.int64)  # dedupe + sort
        return sr, empty.copy()

    raise ValueError(f"unknown static policy: {policy}")


# =========================================================
# T5 joint-name embedding
# =========================================================
def load_t5(model_name, device):
    """
    Lazy import + load T5 encoder & tokenizer. Returns (tokenizer, model, device).
    Raises a clear, actionable error if sentencepiece is missing (T5 tokenizer needs it).
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, T5EncoderModel
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"transformers/torch import failed: {e}. "
            f"Activate the 'mocapanything' conda env."
        )

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load T5 tokenizer '{model_name}': {e}\n"
            f">>> T5 tokenizers require sentencepiece. Install with:\n"
            f"      pip install sentencepiece\n"
            f"    (env 'mocapanything'; t5-base is already in the HF cache.)"
        )

    import torch
    model = T5EncoderModel.from_pretrained(model_name)
    model = model.to(device).eval()
    return tok, model, device


def embed_joint_names(names, tokenizer, model, device, batch_size=64):
    """
    Mask-weighted mean-pooled T5 encoder embedding per joint name.
    Returns float32 [J, 768]. Empty/blank names embed as the encoding of "" (a
    single eos token) -> a deterministic, non-NaN vector.
    """
    import torch

    # Normalize names a little: T5 sees raw strings; we keep the exact joint
    # string (including weird Biped/Japanese-romaji tokens). We only replace an
    # empty string with a single space so the tokenizer always yields >=1 token.
    proc = [n if (isinstance(n, str) and len(n.strip()) > 0) else " " for n in names]

    out = np.zeros((len(proc), T5_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(proc), batch_size):
            chunk = proc[start:start + batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt",
            ).to(device)
            hidden = model(**enc).last_hidden_state  # [b, L, 768]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [b, L, 1]
            summed = (hidden * mask).sum(dim=1)  # [b, 768]
            counts = mask.sum(dim=1).clamp(min=1.0)  # [b, 1]
            pooled = (summed / counts).float().cpu().numpy()  # [b, 768]
            out[start:start + len(chunk)] = pooled.astype(np.float32)

    if out.shape[1] != T5_DIM:
        raise ValueError(
            f"T5 embedding dim {out.shape[1]} != expected {T5_DIM} "
            f"(model.py joint_embed_dim). Wrong T5 variant?"
        )
    return out


# =========================================================
# Per-species build
# =========================================================
def build_one_species(
    short_name,
    bvh_paths,
    bvh_module,
    tokenizer,
    model,
    device,
    static_policy,
    geo_map=None,
):
    rep, modal_J, hist = choose_representative(bvh_paths, bvh_module)
    parents, offsets, names = parse_skeleton(rep, bvh_module)
    J = parents.shape[0]

    if not (1 <= J <= 150):
        raise ValueError(f"{short_name}: J={J} outside sane [1,150]")

    hop = build_hop_matrix(parents, max_hop=MAX_HOP)
    edge = build_edge_matrix(parents)
    t5 = embed_joint_names(names, tokenizer, model, device)
    static_rot, static_pos = build_static_joints(
        parents, names, static_policy, species=short_name, geo_map=geo_map
    )

    # ---- invariants (fail loud) ----
    assert hop.shape == (J, J) and hop.dtype == np.int64
    assert edge.shape == (J, J) and edge.dtype == np.int64
    assert t5.shape == (J, T5_DIM) and t5.dtype == np.float32
    assert hop.min() >= 0 and hop.max() <= MAX_HOP, f"{short_name}: hop out of [0,{MAX_HOP}]"
    assert edge.min() >= 0 and edge.max() <= 5, f"{short_name}: edge out of [0,5]"
    assert np.all(np.diag(hop) == 0), f"{short_name}: hop diag != 0"
    assert np.all(np.diag(edge) == EDGE_SELF), f"{short_name}: edge diag != self"
    if static_rot.size:
        assert static_rot.min() >= 0 and static_rot.max() < J
    if static_pos.size:
        assert static_pos.min() >= 0 and static_pos.max() < J

    info = {
        "parents": parents.astype(np.int64),
        "rest_pose": offsets.astype(np.float32),
        "joints_distance": hop.astype(np.int64),
        "joint_relation": edge.astype(np.int64),
        "t5_embedding": t5.astype(np.float32),
        "static_rot_joints": static_rot.astype(np.int64),
        "static_joints": static_pos.astype(np.int64),
        "joint_names": names,
    }
    return info, rep, modal_J, hist


# =========================================================
# Driver
# =========================================================
def build_all(
    raw_root,
    out_path,
    repo_root,
    t5_model,
    device,
    static_policy,
    static_geo_json=None,
    limit_species=None,
    allow_partial=False,
):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from utils import bvh as bvh_module  # noqa: WPS433 - intentional repo import

    # geo policy: load the codex-verified per-species static_rot map ONCE.
    # none/leaves leave geo_map None (no effect on those paths).
    geo_map = None
    if static_policy == "geo":
        if not static_geo_json or not os.path.isfile(static_geo_json):
            raise RuntimeError(
                f"static_policy='geo' needs an existing --static-geo-json; got {static_geo_json!r}"
            )
        with open(static_geo_json, "r") as f:
            geo_map = json.load(f)
        print(f"[INFO] static_policy=geo: loaded geo map ({len(geo_map)} species) from {static_geo_json}")

    species_dirs = list_species_dirs(raw_root)
    if limit_species:
        keep = set(limit_species)
        species_dirs = [s for s in species_dirs if s in keep]
    if not species_dirs:
        raise RuntimeError(f"No species (dirs with .bvh) found under {raw_root}")

    print(f"[INFO] raw_root={raw_root}")
    print(f"[INFO] {len(species_dirs)} species: {species_dirs}")

    tokenizer, model, device = load_t5(t5_model, device)
    print(f"[INFO] T5='{t5_model}' on {device}; static_policy='{static_policy}'")

    species_info_dict = {}
    fail = []
    for name in species_dirs:
        sp_dir = os.path.join(raw_root, name)
        bvh_paths = list_species_bvhs(sp_dir)
        try:
            info, rep, modal_J, hist = build_one_species(
                short_name=name,
                bvh_paths=bvh_paths,
                bvh_module=bvh_module,
                tokenizer=tokenizer,
                model=model,
                device=device,
                static_policy=static_policy,
                geo_map=geo_map,
            )
            species_info_dict[name] = info
            multi = " <<< MULTI-J (off-J clips will break collate; filter them!)" \
                if len(hist) > 1 else ""
            print(
                f"[OK] {name:16s} J={modal_J:3d}  rep={os.path.basename(rep)}  "
                f"J-hist={dict(sorted(hist.items()))}{multi}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {e}")
            fail.append((name, str(e)))

    if not species_info_dict:
        raise RuntimeError("Built zero species; aborting (nothing to save).")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, species_info_dict, allow_pickle=True)
    print(f"\n[DONE] saved {len(species_info_dict)} species -> {out_path}")
    if fail:
        print(f"[WARN] {len(fail)} species failed:")
        for n, e in fail:
            print(f"    {n}: {e}")

    # one-line schema echo for the next reader
    any_info = next(iter(species_info_dict.values()))
    Jx = any_info["joints_distance"].shape[0]
    print(
        "[SCHEMA] dict[species] = {"
        "parents int64[J], rest_pose f32[J,3], joints_distance int64[J,J] in[0,5], "
        "joint_relation int64[J,J] in[0,5], t5_embedding f32[J,768], "
        "static_rot_joints int64[Sr], static_joints int64[Sp], joint_names list[str]}"
        f"  (example J={Jx})"
    )

    # partial-build abort: a species that failed here is SILENTLY DROPPED from
    # the dict, and the loader then drops every clip of that species (no entry in
    # species_info_dict -> loader_v2.py:306-308 skip). Fail loud by default so a
    # partial table is never used unknowingly; --allow-partial opts into it.
    if fail and not allow_partial:
        print(
            f"[FATAL] {len(fail)} species failed to build and are MISSING from "
            f"{out_path}. The loader will silently drop these species. "
            f"Pass --allow-partial to accept a partial table. Failed species:"
        )
        for n, e in fail:
            print(f"    {n}: {e}")
        sys.exit(1)

    return species_info_dict


# =========================================================
# Selftest (synthetic, no real data, no T5 download/runtime)
# =========================================================
def _selftest():
    """
    Validate hop/edge/static builders + the full per-species dict shape/dtype
    contract on a tiny SYNTHETIC skeleton. Does NOT touch real data and STUBS T5
    so it runs with zero network / zero sentencepiece / zero GPU.

    Synthetic tree (J=5):
        0 (root)
        |- 1
        |  |- 3
        |  '- 4
        '- 2
      parents = [-1, 0, 0, 1, 1]
    """
    parents = np.array([-1, 0, 0, 1, 1], dtype=np.int64)
    names = ["root", "Hips_L", "Hips_R", "Knee_L", "", "extra"][: len(parents)]
    J = parents.shape[0]

    # --- hop ---
    hop = build_hop_matrix(parents, max_hop=MAX_HOP)
    assert hop.shape == (J, J) and hop.dtype == np.int64
    assert np.all(np.diag(hop) == 0)
    # 0-1 adjacent => 1 ; 3-4 are siblings (both children of 1) => hop 2
    assert hop[0, 1] == 1 and hop[1, 0] == 1
    assert hop[3, 4] == 2, f"sibling hop expected 2, got {hop[3,4]}"
    # 2 (child of 0) to 3 (grandchild via 1): 2-0-1-3 = 3
    assert hop[2, 3] == 3, f"expected 3, got {hop[2,3]}"
    assert hop.max() <= MAX_HOP and hop.min() >= 0

    # clamp check on a deep chain
    deep = np.array([-1, 0, 1, 2, 3, 4, 5, 6], dtype=np.int64)  # chain length 8
    hop_deep = build_hop_matrix(deep, max_hop=MAX_HOP)
    assert hop_deep[0, 7] == MAX_HOP, f"deep hop should clamp to {MAX_HOP}, got {hop_deep[0,7]}"

    # --- edge ---
    edge = build_edge_matrix(parents)
    assert edge.shape == (J, J) and edge.dtype == np.int64
    assert np.all(np.diag(edge) == EDGE_SELF)
    assert edge[1, 0] == EDGE_PARENT, "joint1's parent is 0"
    assert edge[0, 1] == EDGE_CHILD, "0's child is 1"
    assert edge[1, 2] == EDGE_SIBLING and edge[2, 1] == EDGE_SIBLING, "1,2 siblings"
    assert edge[3, 4] == EDGE_SIBLING, "3,4 siblings"
    assert edge[3, 2] == EDGE_OTHER, "3,2 unrelated -> other"
    assert edge.min() >= 0 and edge.max() <= 5

    # --- static policies ---
    sr_none, sp_none = build_static_joints(parents, names, "none")
    assert sr_none.shape == (0,) and sp_none.shape == (0,)
    assert sr_none.dtype == np.int64 and sp_none.dtype == np.int64
    sr_leaf, sp_leaf = build_static_joints(parents, names, "leaves")
    # leaves of the synthetic tree: 2, 3, 4 (no children)
    assert set(sr_leaf.tolist()) == {2, 3, 4}, f"leaves wrong: {sr_leaf.tolist()}"
    assert sp_leaf.shape == (0,)

    # --- T5 embedding with a STUB (offline) ---
    class _StubTok:
        def __call__(self, chunk, **kw):
            import torch
            b = len(chunk)
            L = 3
            return _StubEnc(
                {"input_ids": torch.ones(b, L, dtype=torch.long),
                 "attention_mask": torch.ones(b, L, dtype=torch.long)}
            )

    class _StubEnc(dict):
        def to(self, _device):
            return self

    class _StubModel:
        def __call__(self, **enc):
            import torch
            b, L = enc["attention_mask"].shape
            return type("O", (), {"last_hidden_state": torch.ones(b, L, T5_DIM)})()

    import torch  # noqa: F401 - ensure torch present for stub
    t5 = embed_joint_names(names, _StubTok(), _StubModel(), device="cpu")
    assert t5.shape == (J, T5_DIM) and t5.dtype == np.float32
    assert np.isfinite(t5).all()
    # mean-pool of all-ones hidden states with all-ones mask == 1.0 everywhere
    assert np.allclose(t5, 1.0), "stub mean-pool should be 1.0"

    # --- full per-species dict shape/dtype (using the same stubs) ---
    info = {
        "parents": parents.astype(np.int64),
        "rest_pose": np.zeros((J, 3), np.float32),
        "joints_distance": hop.astype(np.int64),
        "joint_relation": edge.astype(np.int64),
        "t5_embedding": t5.astype(np.float32),
        "static_rot_joints": sr_none.astype(np.int64),
        "static_joints": sp_none.astype(np.int64),
        "joint_names": names,
    }
    # exact consumer-contract assertions (mirrors loader_v2.py L316-330)
    assert info["joints_distance"].astype(np.int64).shape[0] == J  # loader L320 derives J
    m_rot = np.zeros((J,), dtype=np.bool_); m_rot[info["static_rot_joints"]] = True
    m_pos = np.zeros((J,), dtype=np.bool_); m_pos[info["static_joints"]] = True
    assert m_rot.shape == (J,) and m_pos.shape == (J,)
    # collate-style padding to MAX_JOINTS must not raise
    J_MAX = 150
    hop_pad = np.full((J_MAX, J_MAX), 5, np.int64); hop_pad[:J, :J] = info["joints_distance"]
    edge_pad = np.full((J_MAX, J_MAX), 4, np.int64); edge_pad[:J, :J] = info["joint_relation"]
    t5_pad = np.concatenate(
        [info["t5_embedding"], np.zeros((J_MAX - J, T5_DIM), np.float32)], axis=0
    )
    assert hop_pad.shape == (J_MAX, J_MAX)
    assert edge_pad.shape == (J_MAX, J_MAX)
    assert t5_pad.shape == (J_MAX, T5_DIM)

    # --- round-trip np.save/np.load of the dict ---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "species_info_dict.npy")
        np.save(p, {"SynthSpecies": info}, allow_pickle=True)
        loaded = np.load(p, allow_pickle=True).item()
        assert "SynthSpecies" in loaded
        li = loaded["SynthSpecies"]
        for k in [
            "parents", "rest_pose", "joints_distance", "joint_relation",
            "t5_embedding", "static_rot_joints", "static_joints", "joint_names",
        ]:
            assert k in li, f"missing key after round-trip: {k}"
        assert li["t5_embedding"].dtype == np.float32
        assert li["joints_distance"].dtype == np.int64
        assert li["joint_relation"].dtype == np.int64
        assert li["parents"].dtype == np.int64

    print("[SELFTEST] all assertions passed (synthetic, offline).")


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build species_info_dict.npy (static per-species skeleton/graph/T5 "
            "table) for MocapAnything V2 from raw Truebones Zoo BVH."
        )
    )
    p.add_argument(
        "--raw-root",
        type=str,
        default=(
            "/scratch/ts1v23/workspace/motion_representation_study/data/raw/"
            "Truebones_Zoo/New-FBX-BVH_Z-OO_full/Truebone_Z-OO"
        ),
        help="Dir containing one sub-dir per species, each with *.bvh files.",
    )
    p.add_argument(
        "--out",
        type=str,
        required=False,
        default=None,
        help=(
            "Output .npy path. The loader expects it at "
            "<base_dir>/species_info_dict.npy (sibling of the 'bvh' dir). "
            "If omitted, writes ./species_info_dict.npy in CWD."
        ),
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=_REPO_ROOT_DEFAULT,
        help="Path to the MocapAnything repo (for utils.bvh import).",
    )
    p.add_argument(
        "--t5-model",
        type=str,
        default="t5-base",
        help="HF model id for the joint-name encoder (d_model MUST be 768).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="torch device for T5 (falls back to cpu if cuda unavailable).",
    )
    p.add_argument(
        "--static-policy",
        type=str,
        default="none",
        choices=["none", "leaves", "geo"],
        help=(
            "Which joints to mark static. 'none' (default, behavior-preserving) "
            "leaves all rotations free; 'leaves' marks leaf joints rot-static "
            "(opt-in, flagged for review); 'geo' reads per-species static_rot "
            "from --static-geo-json (codex-verified geodesic-<2deg map)."
        ),
    )
    p.add_argument(
        "--static-geo-json",
        type=str,
        default=os.path.join(
            _THIS_DIR, "..", "static_analysis",
            "proposed_static_rot_perclip_maxref_2deg.json",
        ),
        help=(
            "JSON map dict[species -> list[int] static_rot_joint_indices] used by "
            "--static-policy geo. Indices are unpadded per-species joint indices."
        ),
    )
    p.add_argument(
        "--limit-species",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of species names to build (debug).",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Accept a partial species_info_dict if some species fail to build. "
            "By default a non-empty failure list aborts with exit code 1 (the "
            "loader silently drops species absent from this table)."
        ),
    )
    p.add_argument(
        "--selftest",
        action="store_true",
        help="Run synthetic offline self-test (no real data / no T5) and exit.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        _selftest()
        return

    # resolve device (cpu fallback) without importing torch at module top
    device = args.device
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[WARN] {device} unavailable, falling back to cpu")
            device = "cpu"
    except Exception:  # noqa: BLE001
        device = "cpu"

    out_path = args.out or os.path.join(os.getcwd(), "species_info_dict.npy")

    build_all(
        raw_root=args.raw_root,
        out_path=out_path,
        repo_root=args.repo_root,
        t5_model=args.t5_model,
        device=device,
        static_policy=args.static_policy,
        static_geo_json=args.static_geo_json,
        limit_species=args.limit_species,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
