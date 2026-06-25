import torch
from utils.config_utils import load_yaml_config
from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded
from utils.rotation import rot6d_to_fk_positions_correct

cfg = load_yaml_config("configs/train/train_pose2rot_v7d_ddp_lr2e4.yaml"); d = cfg["data"]
ds = AnySpeciesPoseDataset(bvh_dir=d["bvh_dir"], window=d["seq_len"], mmap=d.get("mmap", True),
    cache_scale=d.get("cache_scale", True), split_json=d["split_json"], split_mode="train",
    memory_pkl_path=d.get("train_memory_pkl_path"), preload_all=False)
idxs = [0, 200, 500, 1000, 2000]
b = collate_anyspecies_padded([ds[i] for i in idxs])
gt_rot6d = b["rot6d_a"].float()
fk_offset = b["fk_offset"].float()
parents = b["parent_a"]
gt_pos = b["position"].float()       # already root-centered + /gscale
Jv = b["J_valid"]; species = b["species"]
print("fk_offset shape", tuple(fk_offset.shape))
fk_pos = rot6d_to_fk_positions_correct(gt_rot6d, fk_offset, parents).float()  # root-centered
fail = 0
print("=== FK_correct(GT, fk_offset) vs batch['position']  (修前: 0.16~0.59) ===")
for i in range(len(idxs)):
    J = int(Jv[i])
    a = fk_pos[i, :, :J]; g = gt_pos[i, :, :J]
    err = (a - g).abs().mean().item()
    ok = err < 0.01
    fail += (not ok)
    print(f"{species[i][:16]:16s} J={J:3d}  mean_abs_err={err:.6f}  {'PASS' if ok else 'FAIL'}")
print("OVERALL:", "PASS (<0.01)" if fail == 0 else f"FAIL ({fail}/{len(idxs)})")
