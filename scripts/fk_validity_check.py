"""Root-cause: is the TRAINING fk loss target valid?
loss_fk = SmoothL1(rot6d_to_fk_positions(pred_rot6d, offset_a, parents, gscale), batch["position"]).
For this to be a VALID loss, FK(GT_rot6d, offset_a) must reconstruct batch["position"].
If they mismatch (scale/convention), minimizing loss_fk drives the model to a WRONG target.
"""
import torch
from utils.config_utils import load_yaml_config
from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded
from utils.rotation import rot6d_to_fk_positions

cfg = load_yaml_config("configs/train/train_pose2rot_v7d_ddp_lr2e4.yaml")
d = cfg["data"]
ds = AnySpeciesPoseDataset(
    bvh_dir=d["bvh_dir"], window=d["seq_len"], mmap=d.get("mmap", True),
    cache_scale=d.get("cache_scale", True), split_json=d["split_json"], split_mode="train",
    memory_pkl_path=d.get("train_memory_pkl_path"), preload_all=False,
)
# a few diverse samples
idxs = [0, 200, 500, 1000]
b = collate_anyspecies_padded([ds[i] for i in idxs])
gt_rot6d = b["rot6d_a"].float()
offsets = b["offset_a"].float()
parents = b["parent_a"]
gscale = b["global_scale"].float()
gt_pos = b["position"].float()
Jv = b["J_valid"]
species = b["species"]

fk_pos = rot6d_to_fk_positions(gt_rot6d, offsets, parents, gscale).float()  # [B,T,J,3]
# both should be root-centered already; root-center to be safe
fk_c = fk_pos - fk_pos[:, :, 0:1, :]
gt_c = gt_pos - gt_pos[:, :, 0:1, :]

print("=== FK(GT_rot6d, offset_a)  VS  batch['position']  (valid joints) ===")
for i in range(len(idxs)):
    J = int(Jv[i])
    a = fk_c[i, :, :J]; g = gt_c[i, :, :J]
    err = (a - g).abs().mean().item()
    # per-sample best-fit scale (if it's a pure scale mismatch, this ratio explains it)
    num = (a * g).sum().item(); den = (a * a).sum().item() + 1e-9
    scale = num / den
    err_scaled = (a * scale - g).abs().mean().item()
    print(f"{species[i][:14]:14s} J={J:3d} gscale={float(gscale[i]):.4f} | "
          f"mean_abs_err={err:.5f}  fk_rng=[{a.min():.2f},{a.max():.2f}] pos_rng=[{g.min():.2f},{g.max():.2f}] | "
          f"best_scale={scale:.4f} err_after_scale={err_scaled:.5f}")
print("=== 解读: err≈0 → fk loss 目标有效(损坏另有原因); err 大但 err_after_scale≈0 → 纯 scale 不匹配(global_scale 没除); err 大且 scale 救不回 → convention 不匹配 ===")
