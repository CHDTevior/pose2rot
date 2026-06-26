"""MOTION visual QA for pose2rot v11c (the "metric can lie, look at the motion" gate).

For each chosen test clip:
  - run v11c -> pred_rot6d from GT positions+skeleton+memory
  - reconstruct PRED joint positions via the validated CORRECT FK
    (utils.rotation.rot6d_to_fk_positions_correct), the SAME fn the training loss uses
  - GT joint positions = batch["position"]
  - render side-by-side articulated-skeleton MOTION animation (GT | PRED | OVERLAY) over the
    full clip, as joints (points) + bones (parent-child lines), same camera/scale, gif.

Renderer faithfulness SELF-CHECK: also render GT-vs-FK(GT) for one species; the two skeleton
panels MUST be visually identical (FK(GT)=position validated to ~2e-7).

Read-only w.r.t. all training artifacts. Single-GPU; share GPU on swarma1003 is fine.
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded
from utils.config_utils import load_yaml_config, instantiate_from_config
from utils.rotation import rot6d_to_fk_positions_correct
from torch.utils.data import DataLoader

CFG = os.environ.get("QA_CFG", "configs/train/train_pose2rot_v11c_lr1e4_fkcal.yaml")
CKPT = os.environ.get("QA_CKPT",
    "checkpoints/pose2rot/exp_pose2rot_v11c_lr1e4_fkcal/pose2rot_ckpt_epoch50.pt")
OUTDIR = os.environ.get("QA_OUT", "v11c_visual_qa")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# (species_short_name, split_group)
TARGETS = [
    ("Horse", "seen"),    # quadruped
    ("Eagle", "seen"),    # bird
    ("Trex", "seen"),     # biped dinosaur
    ("Cat", "unseen"),    # held-out quadruped
    ("Spider", "unseen"), # held-out, 8-legged, very distinct topology
    ("Goat", "unseen"),   # held-out quadruped
]
SELFCHECK_SPECIES = "Horse"


def make_loader(cfg, split_group):
    dc = cfg["data"]
    ds = AnySpeciesPoseDataset(
        bvh_dir=dc["bvh_dir"], window=dc["seq_len"], mmap=dc.get("mmap", True),
        cache_scale=dc.get("cache_scale", True),
        limit_species_debug=dc.get("limit_species_debug", []),
        split_json=dc["split_json"], split_mode="test", split_group=split_group,
        memory_pkl_path=dc.get("test_memory_pkl_path", dc.get("train_memory_pkl_path")),
        preload_all=False, image_embed_mode=dc.get("image_embed_mode", None),
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                      collate_fn=collate_anyspecies_padded, drop_last=False)


def to_dev(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(DEVICE)
    return batch


def get_first_clip(loader, species):
    for batch in loader:
        if batch["species"][0] == species:
            return batch
    return None


def bone_pairs(parents, Jv):
    """list of (child, parent) for valid joints with a valid parent."""
    pairs = []
    for j in range(Jv):
        p = int(parents[j])
        if 0 <= p < Jv:
            pairs.append((j, p))
    return pairs


def render_clip(name, panels, parents, Jv, out_path, fps=12, elev=12, azim=-75):
    """panels: list of (title, pos[W,J,3], color). Renders an animated multi-panel
    3D skeleton gif. Same axis limits & camera across panels. y(height) -> vertical.
    """
    W = panels[0][1].shape[0]
    pairs = bone_pairs(parents, Jv)

    # combined range across all panels (valid joints) -> identical scale, exposes drift
    allpts = np.concatenate([P[:, :Jv, :].reshape(-1, 3) for _, P, _ in panels], axis=0)
    # plot mapping: (x, z, y) so data-y(height) is the vertical plot axis
    px, py, pz = allpts[:, 0], allpts[:, 2], allpts[:, 1]
    def lims(a):
        lo, hi = float(a.min()), float(a.max())
        c, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
        r = max(r, 1e-3) * 1.15
        return c - r, c + r
    xlim, ylim, zlim = lims(px), lims(py), lims(pz)

    n = len(panels)
    fig = plt.figure(figsize=(5.0 * n, 5.2))
    axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]

    def draw_frame(f):
        for ax, (title, P, color) in zip(axes, panels):
            ax.cla()
            pts = P[f, :Jv, :]  # [Jv,3]
            X, Y, Z = pts[:, 0], pts[:, 2], pts[:, 1]  # map height->vertical
            ax.scatter(X, Y, Z, c=color, s=14, depthshade=False)
            for (j, p) in pairs:
                ax.plot([pts[j, 0], pts[p, 0]],
                        [pts[j, 2], pts[p, 2]],
                        [pts[j, 1], pts[p, 1]], c=color, lw=1.4)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
            ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        fig.suptitle(f"{name}   frame {f+1}/{W}", fontsize=12)
        return []

    anim = FuncAnimation(fig, draw_frame, frames=W, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def overlay_panel_frame_count(P):
    return P.shape[0]


def motion_stats(gt, pred, Jv):
    """gt/pred: [W,J,3] (valid joints first Jv). Returns dict of motion diagnostics."""
    g = gt[:, :Jv, :]
    p = pred[:, :Jv, :]
    W = g.shape[0]
    mpjpe = float(np.linalg.norm(g - p, axis=-1).mean())          # normalized units
    maxjpe = float(np.linalg.norm(g - p, axis=-1).max())
    # temporal motion magnitude: std over time per joint coord, averaged
    gt_dyn = float(g.std(axis=0).mean())
    pred_dyn = float(p.std(axis=0).mean())
    ratio = pred_dyn / max(gt_dyn, 1e-9)
    # per-frame velocity magnitude (mean joint displacement between consecutive frames)
    if W > 1:
        gt_vel = float(np.linalg.norm(np.diff(g, axis=0), axis=-1).mean())
        pred_vel = float(np.linalg.norm(np.diff(p, axis=0), axis=-1).mean())
    else:
        gt_vel = pred_vel = 0.0
    # extent (skeleton bounding-box diagonal of GT) for normalizing mpjpe to %
    ext = float(np.linalg.norm(g.reshape(-1, 3).max(0) - g.reshape(-1, 3).min(0)))
    return {
        "W": W, "Jv": Jv,
        "mpjpe": mpjpe, "maxjpe": maxjpe,
        "mpjpe_pct_extent": 100.0 * mpjpe / max(ext, 1e-9),
        "gt_dyn_std": gt_dyn, "pred_dyn_std": pred_dyn, "dyn_ratio": ratio,
        "gt_vel": gt_vel, "pred_vel": pred_vel,
        "vel_ratio": pred_vel / max(gt_vel, 1e-9),
        "extent": ext,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    print("DEVICE", DEVICE, "OUTDIR", os.path.abspath(OUTDIR))
    cfg = load_yaml_config(CFG)
    model = instantiate_from_config(cfg["model"]).to(DEVICE)
    model.eval()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(ck["model_state"], strict=False)
    ep = ck.get("epoch")
    print(f"loaded {CKPT}  epoch_in_ckpt={ep}  missing={len(miss)} unexpected={len(unexp)}")

    # group -> loader (built lazily)
    loaders = {}
    summary = {}

    for species, group in TARGETS:
        if group not in loaders:
            loaders[group] = make_loader(cfg, group)
        batch = get_first_clip(loaders[group], species)
        if batch is None:
            print(f"!! no clip found for {species} ({group})")
            continue
        rel = None
        batch = to_dev(batch)
        with torch.no_grad():
            out = model(batch)
        pred_rot6d = out["pred_rot6d"]
        parents = batch["parent_a"]
        Jv = int(batch["J_valid"][0].item())
        W = int(batch["position"].shape[1])

        gt_pos = batch["position"][0].detach().cpu().numpy()                       # [W,J,3]
        pred_fk = rot6d_to_fk_positions_correct(pred_rot6d, batch["fk_offset"], parents)
        gt_fk = rot6d_to_fk_positions_correct(batch["rot6d_a"], batch["fk_offset"], parents)
        pred_pos = pred_fk[0].detach().cpu().numpy()
        gt_fk_pos = gt_fk[0].detach().cpu().numpy()
        par = parents[0].detach().cpu().numpy()

        # FK self-check error (numbers)
        jm = batch["joint_mask"][0].bool().detach().cpu().numpy()
        sc_err = np.abs((gt_fk_pos - gt_pos)[:, :Jv, :]).max()

        st = motion_stats(gt_pos, pred_pos, Jv)
        st["fk_selfcheck_maxerr"] = float(sc_err)
        st["group"] = group
        st["species"] = species
        summary[f"{species}_{group}"] = st
        print(f"\n[{species} | {group}] Jv={Jv} W={W} "
              f"mpjpe={st['mpjpe']:.4f} ({st['mpjpe_pct_extent']:.1f}% extent) "
              f"maxjpe={st['maxjpe']:.4f} dyn_ratio={st['dyn_ratio']:.3f} "
              f"vel_ratio={st['vel_ratio']:.3f} fk_selfcheck_maxerr={sc_err:.2e}")

        out_path = os.path.join(OUTDIR, f"{species}_{group}.gif")
        # 3-panel render: GT | PRED | OVERLAY
        render_with_overlay(
            f"{species} ({group})  GT vs PRED  [v11c ep{ep}]",
            gt_pos, pred_pos, par, Jv, out_path, fps=args.fps,
        )
        print(f"   saved {os.path.abspath(out_path)}")

        # one self-check render: GT positions vs FK(GT) positions -> must be identical
        if species == SELFCHECK_SPECIES:
            sc_path = os.path.join(OUTDIR, f"_SELFCHECK_{species}_GTpos_vs_FKofGT.gif")
            render_clip(
                f"SELF-CHECK {species}: GT-position vs FK(GT)  (must be identical, err={sc_err:.1e})",
                [("GT position", gt_pos, "tab:blue"),
                 ("FK(GT rot6d)", gt_fk_pos, "tab:green")],
                par, Jv, sc_path, fps=args.fps,
            )
            print(f"   saved SELF-CHECK {os.path.abspath(sc_path)}")

    with open(os.path.join(OUTDIR, "qa_motion_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY (normalized units; mpjpe_pct_extent = %% of GT bbox diagonal) ===")
    for k, st in summary.items():
        print(f"{k:18s} mpjpe={st['mpjpe']:.4f} ({st['mpjpe_pct_extent']:5.1f}%)  "
              f"maxjpe={st['maxjpe']:.4f}  dyn_ratio={st['dyn_ratio']:.3f}  "
              f"vel_ratio={st['vel_ratio']:.3f}")
    print("\nALL DONE. ckpt epoch:", ep)


def render_with_overlay(name, gt_pos, pred_pos, parents, Jv, out_path, fps=12,
                        elev=12, azim=-75):
    """3 panels: GT | PRED | OVERLAY(GT gray + PRED red), shared scale/camera."""
    W = gt_pos.shape[0]
    pairs = bone_pairs(parents, Jv)
    allpts = np.concatenate([gt_pos[:, :Jv].reshape(-1, 3),
                             pred_pos[:, :Jv].reshape(-1, 3)], axis=0)
    px, py, pz = allpts[:, 0], allpts[:, 2], allpts[:, 1]
    def lims(a):
        lo, hi = float(a.min()), float(a.max())
        c, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
        r = max(r, 1e-3) * 1.15
        return c - r, c + r
    xlim, ylim, zlim = lims(px), lims(py), lims(pz)
    box = (xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])

    fig = plt.figure(figsize=(15.0, 5.2))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]

    def draw_skel(ax, pts, color, lw=1.4, s=14, alpha=1.0):
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=color, s=s,
                   depthshade=False, alpha=alpha)
        for (j, p) in pairs:
            ax.plot([pts[j, 0], pts[p, 0]],
                    [pts[j, 2], pts[p, 2]],
                    [pts[j, 1], pts[p, 1]], c=color, lw=lw, alpha=alpha)

    def setup(ax, title):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_box_aspect(box)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    def draw_frame(f):
        g = gt_pos[f, :Jv]; p = pred_pos[f, :Jv]
        axes[0].cla(); draw_skel(axes[0], g, "tab:blue"); setup(axes[0], "GT")
        axes[1].cla(); draw_skel(axes[1], p, "tab:red"); setup(axes[1], "PRED")
        axes[2].cla()
        draw_skel(axes[2], g, "0.55", lw=1.0, s=8, alpha=0.8)
        draw_skel(axes[2], p, "tab:red", lw=1.4, s=12, alpha=0.9)
        setup(axes[2], "OVERLAY  GT(gray) + PRED(red)")
        fig.suptitle(f"{name}   frame {f+1}/{W}", fontsize=12)
        return []

    anim = FuncAnimation(fig, draw_frame, frames=W, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    main()
