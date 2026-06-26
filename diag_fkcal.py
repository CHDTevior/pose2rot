"""Diagnostic for pose2rot v11 FK-loss over-dominance (TASK 1).

(a) Unweighted training-loss components + weighted contributions (fk_wt=10) at ep15/ep19,
    matching utils/loss.compute_rot_loss term-by-term (criterion types preserved).
(b) Per-tier geodesic angle_l1 (seen/rare/unseen) at ep15/ep19 via loader split_group.

Single-GPU. Run on swarma1004 via srun. Does NOT modify any training artifact.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded
from utils.config_utils import load_yaml_config, instantiate_from_config
from utils.loss import (
    masked_loss, rot6d_vel_loss, rot6d_acc_loss, angle_L1,
)
from utils.rotation import rot6d_to_fk_positions_correct
from torch.utils.data import DataLoader

CFG = os.environ.get("DIAG_CFG", "configs/train/train_pose2rot_v11_fresh.yaml")
CKPT_DIR = os.environ.get("DIAG_CKPT_DIR", "checkpoints/pose2rot/exp_pose2rot_v11_fresh")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(cfg):
    model = instantiate_from_config(cfg["model"]).to(DEVICE)
    model.eval()
    return model


def load_ckpt(model, epoch):
    ck = torch.load(os.path.join(CKPT_DIR, f"pose2rot_ckpt_epoch{epoch}.pt"),
                    map_location="cpu", weights_only=False)
    sd = ck["model_state"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  [ep{epoch}] loaded; missing={len(missing)} unexpected={len(unexpected)}")
    return model


def make_loader(cfg, split_group=None):
    data_cfg = cfg["data"]
    ds = AnySpeciesPoseDataset(
        bvh_dir=data_cfg["bvh_dir"],
        window=data_cfg["seq_len"],
        mmap=data_cfg.get("mmap", True),
        cache_scale=data_cfg.get("cache_scale", True),
        limit_species_debug=data_cfg.get("limit_species_debug", []),
        split_json=data_cfg["split_json"],
        split_mode="test",
        split_group=split_group,
        memory_pkl_path=data_cfg.get("test_memory_pkl_path", data_cfg.get("train_memory_pkl_path")),
        preload_all=False,
        image_embed_mode=data_cfg.get("image_embed_mode", None),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=collate_anyspecies_padded, drop_last=False)
    return loader


def to_dev(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(DEVICE)
    return batch


# ---- criterion factory matching training exactly ----
def crit(kind):
    if kind == "smooth_l1":
        return nn.SmoothL1Loss(reduction="none")
    if kind == "l1":
        return nn.L1Loss(reduction="none")
    if kind == "l2":
        return nn.MSELoss(reduction="none")
    raise ValueError(kind)


@torch.no_grad()
def measure_components(model, loader, weight, max_batches=None):
    """Replicate utils/loss.compute_rot_loss term-by-term, accumulate UNWEIGHTED magnitudes.
    Criterion types match training: rot/vel/acc/root=smooth_l1, fk=SmoothL1, tvar=L1."""
    rot_c = crit("smooth_l1")   # rot_loss_type in cfg
    vel_c = crit("smooth_l1")
    acc_c = crit("smooth_l1")
    rotmse_c = crit("l2")       # for rot_l2 reporting

    acc = {k: 0.0 for k in ["loss_rot", "loss_rot_l2", "loss_vel", "loss_acc",
                            "loss_fk", "loss_root", "loss_tvar"]}
    n = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = to_dev(batch)
        out = model(batch)
        pred_rot6d = out["pred_rot6d"]
        gt_rot6d = batch["rot6d_a"]
        target_pose = batch["position"]

        joint_mask = batch["joint_mask"].bool()
        static_rot_mask = batch["static_rot_joint_mask"].bool()
        static_pos_mask = batch["static_pos_joint_mask"].bool()
        rot_jm = joint_mask & (~static_rot_mask)
        pos_jm = joint_mask & (~static_pos_mask)

        parents = batch["parent_a"]

        bs = gt_rot6d.size(0)

        loss_rot = masked_loss(pred_rot6d, gt_rot6d, rot_jm, rot_c)
        loss_rot_l2 = masked_loss(pred_rot6d, gt_rot6d, rot_jm, rotmse_c)
        loss_vel = rot6d_vel_loss(pred_rot6d, gt_rot6d, rot_jm, vel_c)
        loss_acc = rot6d_acc_loss(pred_rot6d, gt_rot6d, rot_jm, acc_c)

        # FK via the CORRECT fn (matches training utils/loss.py:284)
        pred_pos = rot6d_to_fk_positions_correct(pred_rot6d, batch["fk_offset"], parents)
        loss_fk = masked_loss(pred_pos, target_pose, pos_jm,
                              nn.SmoothL1Loss(reduction="none"))

        # root: smooth_l1 on root joint only
        root_mask = torch.zeros_like(joint_mask)
        root_mask[:, 0] = True
        loss_root = masked_loss(pred_rot6d, gt_rot6d, root_mask, rot_c)

        # tvar: L1 on demeaned-temporal
        pred_dm = pred_rot6d - pred_rot6d.mean(dim=1, keepdim=True)
        gt_dm = gt_rot6d - gt_rot6d.mean(dim=1, keepdim=True)
        loss_tvar = masked_loss(pred_dm, gt_dm, rot_jm, nn.L1Loss(reduction="none"))

        acc["loss_rot"] += float(loss_rot) * bs
        acc["loss_rot_l2"] += float(loss_rot_l2) * bs
        acc["loss_vel"] += float(loss_vel) * bs
        acc["loss_acc"] += float(loss_acc) * bs
        acc["loss_fk"] += float(loss_fk) * bs
        acc["loss_root"] += float(loss_root) * bs
        acc["loss_tvar"] += float(loss_tvar) * bs
        n += bs

    for k in acc:
        acc[k] /= max(n, 1)
    acc["_n"] = n
    return acc


@torch.no_grad()
def measure_angle(model, loader, max_batches=None):
    """Geodesic angle_l1 (deg) over rot_joint_mask, matching evaluate_batch_metrics."""
    tot = 0.0
    n = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = to_dev(batch)
        out = model(batch)
        pred_rot6d = out["pred_rot6d"]
        gt_rot6d = batch["rot6d_a"]
        joint_mask = batch["joint_mask"].bool()
        static_rot_mask = batch["static_rot_joint_mask"].bool()
        rot_jm = (joint_mask & (~static_rot_mask)).detach().cpu().numpy()
        bs = gt_rot6d.size(0)
        a = angle_L1(pred_rot6d, gt_rot6d, mask=rot_jm)
        tot += float(a) * bs
        n += bs
    return (tot / max(n, 1)), n


def report_components(label, comp, weight):
    print(f"\n===== TASK 1(a) loss components @ {label} (n={comp['_n']} clips) =====")
    # weighted contributions
    w = weight
    contrib = {
        "rot":  w["rot_wt"] * comp["loss_rot"],
        "vel":  w["vel_wt"] * comp["loss_vel"],
        "acc":  w["acc_wt"] * comp["loss_acc"],
        "tvar": w["tvar_wt"] * comp["loss_tvar"],
        "root": w["root_wt"] * comp["loss_root"],
        "fk":   w["fk_wt"] * comp["loss_fk"],
    }
    rot_obj = contrib["rot"] + contrib["vel"] + contrib["acc"] + contrib["tvar"] + contrib["root"]
    total = rot_obj + contrib["fk"]
    print(f"  UNWEIGHTED magnitudes:")
    print(f"    loss_rot (smooth_l1) = {comp['loss_rot']:.6f}   (rot_l2={comp['loss_rot_l2']:.6f})")
    print(f"    loss_vel             = {comp['loss_vel']:.6f}")
    print(f"    loss_acc             = {comp['loss_acc']:.6f}")
    print(f"    loss_tvar (L1)       = {comp['loss_tvar']:.6f}")
    print(f"    loss_root            = {comp['loss_root']:.6f}")
    print(f"    loss_fk (SmoothL1, CORRECT fn) = {comp['loss_fk']:.6f}")
    print(f"  WEIGHTED contributions (weights: rot={w['rot_wt']} vel={w['vel_wt']} acc={w['acc_wt']} "
          f"tvar={w['tvar_wt']} root={w['root_wt']} fk={w['fk_wt']}):")
    for k in ["rot", "vel", "acc", "tvar", "root", "fk"]:
        print(f"    {k:5s} = {contrib[k]:.6f}   ({100*contrib[k]/max(total,1e-9):5.1f}% of total)")
    print(f"    --- rotation_objective (rot+vel+acc+tvar+root) = {rot_obj:.6f}")
    print(f"    --- fk weighted contribution                   = {contrib['fk']:.6f}")
    print(f"    --- TOTAL objective                            = {total:.6f}")
    print(f"    >>> FK SHARE of total = {100*contrib['fk']/max(total,1e-9):.1f}%")
    print(f"    >>> FK / rotation_objective ratio = {contrib['fk']/max(rot_obj,1e-9):.2f}x")
    return {"comp": comp, "contrib": contrib, "rot_obj": rot_obj, "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_batches", type=int, default=None,
                    help="cap test clips for speed; None = full test set")
    ap.add_argument("--epochs", type=int, nargs="+", default=[15, 19])
    args = ap.parse_args()

    cfg = load_yaml_config(CFG)
    weight = cfg["train"]["weight"]
    print(f"Device: {DEVICE}")
    print(f"v11 weights: {dict(weight)}")

    model = build_model(cfg)

    # ---- TASK 1(a): full test-set components ----
    full_loader = make_loader(cfg, split_group=None)
    comp_results = {}
    for ep in args.epochs:
        load_ckpt(model, ep)
        comp = measure_components(model, full_loader, weight, max_batches=args.max_batches)
        comp_results[ep] = report_components(f"epoch{ep}", comp, weight)

    # ---- TASK 1(b): per-tier angle ----
    print("\n\n===== TASK 1(b) per-tier geodesic angle_l1 (deg) =====")
    tier_loaders = {t: make_loader(cfg, split_group=t) for t in ["seen", "rare", "unseen"]}
    tier_table = {}
    for ep in args.epochs:
        load_ckpt(model, ep)
        tier_table[ep] = {}
        for t, ld in tier_loaders.items():
            ang, n = measure_angle(model, ld, max_batches=args.max_batches)
            tier_table[ep][t] = (ang, n)
            print(f"  ep{ep} {t:6s}: angle_l1 = {ang:6.2f} deg  (n={n})")

    # ---- summary for calibration ----
    print("\n\n===== CALIBRATION SUMMARY =====")
    for ep in args.epochs:
        r = comp_results[ep]
        print(f"ep{ep}: rotation_objective={r['rot_obj']:.6f}  loss_fk(unweighted)={r['comp']['loss_fk']:.6f}  "
              f"fk_share={100*r['contrib']['fk']/max(r['total'],1e-9):.1f}%")
    print(json.dumps({
        "components": {ep: comp_results[ep]["comp"] for ep in args.epochs},
        "rot_obj": {ep: comp_results[ep]["rot_obj"] for ep in args.epochs},
        "tiers": {ep: {t: tier_table[ep][t][0] for t in tier_table[ep]} for ep in args.epochs},
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
