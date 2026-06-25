"""Per-term GRADIENT-norm attribution: for the stable(52) and onset(53) ckpts, backward
EACH loss term ALONE and measure the gradient L2 norm it produces. The term with the
largest grad norm is what DROVE the weight updates (cause), not just the largest loss (symptom).
grad_clip=1.0 clips the TOTAL grad to norm 1.0, so a term whose grad norm >> others dominates the clipped step.
Run from repo root via srun.
"""
import torch
from torch.utils.data import DataLoader
from utils.config_utils import load_yaml_config, instantiate_from_config
from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded
from utils.loss import compute_rot_loss, get_loss_fn

cfg = load_yaml_config("configs/train/train_pose2rot_v8b_fkramp.yaml")
d, tc = cfg["data"], cfg["train"]
CKDIR = "checkpoints/pose2rot/exp_pose2rot_v8b_fkramp"
W = tc["weight"]
TERMS = ["rot_wt", "vel_wt", "acc_wt", "fk_wt", "root_wt", "tvar_wt"]
rot_fn = get_loss_fn("smooth_l1"); vel_fn = get_loss_fn("smooth_l1"); acc_fn = get_loss_fn("smooth_l1")

ds = AnySpeciesPoseDataset(bvh_dir=d["bvh_dir"], window=d["seq_len"], mmap=d.get("mmap", True),
    cache_scale=d.get("cache_scale", True), split_json=d["split_json"], split_mode="train",
    memory_pkl_path=d.get("train_memory_pkl_path"), preload_all=False)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                    collate_fn=collate_anyspecies_padded, drop_last=True)
batch = next(iter(loader))
bd = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

def load_model(ep):
    m = instantiate_from_config(cfg["model"]).cuda().eval()
    sd = torch.load(f"{CKDIR}/pose2rot_ckpt_epoch{ep}.pt", map_location="cuda")
    m.load_state_dict(sd["model_state"])
    return m

print("=== per-term GRADIENT NORM (each term alone) — driver vs symptom ===", flush=True)
for ep in [52, 53]:
    m = load_model(ep)
    print(f"\n--- epoch{ep} ---", flush=True)
    for term in TERMS:
        m.zero_grad(set_to_none=True)
        w = {t: (W[t] if t == term else 0.0) for t in TERMS}
        out = m(bd)
        loss, ld = compute_rot_loss(out, bd, w, rot_fn, vel_fn, acc_fn)
        if loss.requires_grad and float(loss) != 0:
            loss.backward()
            gn = sum((p.grad.float().norm().item() ** 2) for p in m.parameters() if p.grad is not None) ** 0.5
        else:
            gn = 0.0
        print(f"    {term:8s}: weighted_loss={float(loss):.5f}  grad_norm={gn:.4f}", flush=True)
    del m; torch.cuda.empty_cache()
print("\n=== DONE (grad_clip=1.0: 谁的grad_norm远大于其它, 谁就主导被clip后的step方向) ===", flush=True)
