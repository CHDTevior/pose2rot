import collections, json
from utils.config_utils import load_yaml_config
from data.loader_v2 import AnySpeciesPoseDataset
cfg = load_yaml_config("configs/train/train_pose2rot_v9_fk10ramp.yaml")
d = cfg["data"]
ds = AnySpeciesPoseDataset(bvh_dir=d["bvh_dir"], split_json=d.get("split_json"),
        split_mode="train", memory_pkl_path=d["train_memory_pkl_path"], preload_all=False)
# rel = 'Species#Motion/yAngle' ; motion = 'Species#Motion' ; angle = 'yAngle'
sp_motions = collections.defaultdict(set)    # species -> set(motion)
motion_angles = collections.defaultdict(set) # motion -> set(angle)
for it in ds.items:
    rel = it['rel']
    motion = rel.split('/')[0]
    angle = rel.split('/')[-1] if '/' in rel else 'y0'
    sp = motion.split('#')[0]
    sp_motions[sp].add(motion)
    motion_angles[motion].add(angle)
n_motions = sum(len(m) for m in sp_motions.values())
print(f"TOTAL clips={len(ds.items)}  species={len(sp_motions)}  motions={n_motions}", flush=True)
# angles per motion distribution
ang_dist = collections.Counter(len(a) for a in motion_angles.values())
print(f"angles-per-motion distribution: {dict(sorted(ang_dist.items()))}", flush=True)
print("\n=== species: n_motions / n_clips (sorted by motions) ===", flush=True)
rows = sorted(sp_motions.items(), key=lambda kv: -len(kv[1]))
for sp, ms in rows:
    nclip = sum(len(motion_angles[m]) for m in ms)
    print(f"  {sp:22s} motions={len(ms):3d}  clips={nclip:3d}", flush=True)
# dump full structure for split building
out = {sp: sorted(list(ms)) for sp, ms in sp_motions.items()}
ang = {m: sorted(list(a)) for m, a in motion_angles.items()}
json.dump({"species_motions": out, "motion_angles": ang},
          open("../zoo1030_build/dataset_motions.json", "w"))
print("\ndumped -> zoo1030_build/dataset_motions.json", flush=True)
