"""Diagnose the v8b epoch54 divergence: WHICH loss term blew up, WHICH submodule's
weights diverged, and is the raw 6D output exploding. Compare epoch52(good)/53(onset)/54(blown).
Run from repo root via srun: python zoo1030_build/scripts/diagnose_divergence.py
"""
import torch
from torch.utils.data import DataLoader
from utils.config_utils import load_yaml_config, instantiate_from_config
from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded
from utils.loss import compute_rot_loss, get_loss_fn

cfg = load_yaml_config("configs/train/train_pose2rot_v8b_fkramp.yaml")
d, tc = cfg["data"], cfg["train"]
CKDIR = "checkpoints/pose2rot/exp_pose2rot_v8b_fkramp"
wcfg = tc["weight"]  # real weights (fk_wt=30 etc.)
rot_fn = get_loss_fn("smooth_l1"); vel_fn = get_loss_fn("smooth_l1"); acc_fn = get_loss_fn("smooth_l1")

ds = AnySpeciesPoseDataset(
    bvh_dir=d["bvh_dir"], window=d["seq_len"], mmap=d.get("mmap", True),
    cache_scale=d.get("cache_scale", True), split_json=d["split_json"], split_mode="train",
    memory_pkl_path=d.get("train_memory_pkl_path"), preload_all=False,
)
# FIXED batch (same data for all 3 ckpts) so differences are purely the weights
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4,
                    collate_fn=collate_anyspecies_padded, drop_last=True)
batch = next(iter(loader))
bd = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

def load_model(ep):
    m = instantiate_from_config(cfg["model"]).cuda().eval()
    sd = torch.load(f"{CKDIR}/pose2rot_ckpt_epoch{ep}.pt", map_location="cuda")
    msd = sd["model_state"]
    msd = { (k[7:] if k.startswith("module.") else k): v for k, v in msd.items() }
    m.load_state_dict(msd)
    return m, sd

print("=== v8b divergence diagnosis: epoch52(good) / 53(onset) / 54(blown) ===", flush=True)
for ep in [52, 53, 54]:
    m, sd = load_model(ep)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = m(bd)
        loss, ld = compute_rot_loss(out, bd, wcfg, rot_fn, vel_fn, acc_fn)
    pred = out["pred_rot6d"].float()
    g = lambda k: float(ld[k]) if k in ld and torch.is_tensor(ld[k]) else -1
    print(f"\n--- epoch{ep}: TOTAL={float(loss):.4f} | pred6d |max|={pred.abs().max():.3f} std={pred.std():.4f} mean={pred.mean():.4f}", flush=True)
    print(f"    per-term RAW: rot={g('loss_rot'):.5f} vel={g('loss_vel'):.5f} acc={g('loss_acc'):.5f} "
          f"fk={g('loss_fk'):.5f} root={g('loss_root'):.5f} tvar={g('loss_tvar'):.5f}", flush=True)
    print(f"    per-term WEIGHTED(x权重): rot={g('loss_rot')*wcfg['rot_wt']:.4f} vel={g('loss_vel')*wcfg['vel_wt']:.4f} "
          f"acc={g('loss_acc')*wcfg['acc_wt']:.4f} fk={g('loss_fk')*wcfg['fk_wt']:.4f} "
          f"root={g('loss_root')*wcfg['root_wt']:.4f} tvar={g('loss_tvar')*wcfg['tvar_wt']:.4f}", flush=True)
    # per-submodule weight norms + max abs (which component's weights diverged)
    for name in ["rest_encoder", "pose_encoder", "memory_encoder", "decoder"]:
        sub = getattr(m, name, None)
        if sub is None: continue
        wn = sum(p.detach().float().norm().item()**2 for p in sub.parameters()) ** 0.5
        wmax = max(p.detach().abs().max().item() for p in sub.parameters())
        print(f"    W[{name:14s}] L2norm={wn:9.3f}  max|w|={wmax:8.4f}", flush=True)
    # Adam 2nd-moment health (if optimizer saved) — exploding exp_avg_sq = instability signature
    opt = sd.get("optimizer_state")
    if opt and "state" in opt:
        sqs = [s["exp_avg_sq"].max().item() for s in opt["state"].values() if "exp_avg_sq" in s]
        avgs = [s["exp_avg"].abs().max().item() for s in opt["state"].values() if "exp_avg" in s]
        if sqs:
            print(f"    Adam: max(exp_avg_sq)={max(sqs):.3e}  max|exp_avg|={max(avgs):.3e}", flush=True)
    del m; torch.cuda.empty_cache()
print("\n=== DONE ===", flush=True)
