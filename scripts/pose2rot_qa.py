#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pose2rot 可视化验收 QA (position -> rotation).

这是对已训练 pose2rot 模型 (Pose2RotMemoryRestModel) 的「动作迁移可视化验收」:
模型吃关节位置 position, 吐每关节 6D 旋转 pred_rot6d; 我们把预测旋转做前向运动学
(FK) 还原成骨架运动, 与 GT 骨架运动并排出多帧 gif, 人眼检查 (用户铁律: CV 任务
可视化 demo 的准确程度 > metric)。

★ 重要诚实标注:
  selected_test_split1010.json 是空的 ({"seen":{},"rare":{},"unseen":{}})。因此本 QA
  里的所有 clip 在训练时模型 *都见过*。这衡量的是 **重建 (reconstruction) 质量, 不是
  泛化 (generalization)**。不要把这里的好结果当作 unseen-topology / 跨物种迁移的证据。

★ FK 约定 linchpin (codex 早指出, 已被三份独立调研 + 实测证实):
  npz 里的 rot6d 存的是旋转矩阵 M 的「前两行」(ROW convention, 见 preprocess/
  extract_bvh_pose.py:92-101)。要从 rot6d 重建 M 必须用 *行* 约定:
      utils/transforms3d.py:578 rotation_6d_to_matrix  ->  stack(...,dim=-2)  [正确]
  仓库里 utils/rotation.py 的 rot6d_to_rotmat_tensor (dim=-1, 列约定) 重建出的是 M^T,
  会让 FK 误差与骨架尺寸同量级 (实测 max~1e2)。它被 rot6d_to_fk_positions 调用,
  train 里因 fk_wt=0.0 从未生效, 所以这个转置 bug 从没被训练 loss 抓到。
  **本脚本禁用 rot6d_to_fk_positions / rot6d_to_rotmat_*, 只用 rotation_6d_to_matrix
  + bvh_forward (后者本身的 parent-on-left 链是对的)。**

★ 自检闸门 (fail-loud, 硬信息, 不许静默跳过):
  对每条 clip, 先用 GT rot6d 跑 FK, 与 npz 的 GT position 比 (都 root-center)。
  通过阈值: FK(GT_rot6d) vs GT position 的 max 误差应到 ~1e-5 量级 (相对 ~1e-8)。
  本脚本用 FK_SELFTEST_ABS_TOL (绝对) 与 FK_SELFTEST_REL_TOL (相对 extent) 双判据:
  任一通过即算 PASS。若两者都不过 -> 打印大警告, 在 gif 标题标 "FK UNTRUSTED",
  该条的 MPJPE 标记为不可信 (仍出图供人眼看, 自检误差写进 summary)。

用法:
  conda activate mocapanything
  cd <repo>/MocapAnything            # 必须在仓库根跑 (import utils.transforms3d 等)
  python <repo>/zoo1030_build/scripts/pose2rot_qa.py \
      --species Horse Buffalo Camel Cat --n_clips 2 --gpu 1

我 (用户) 来跑, 脚本本身不在调研阶段执行。
"""

import os
import sys
import json
import glob
import re
import math
import random
import argparse
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------------
# 路径常量 (绝对路径, 防 cwd 漂移)
# ----------------------------------------------------------------------------
REPO_ROOT = "."
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs/train/train_pose2rot.yaml")
DEFAULT_CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints/pose2rot/exp_pose2rot_v3")
DEFAULT_OUT = ("../"
               "artifacts/20260619_014725_MocapAnything/pose2rot_qa")

# FK 自检阈值 (见 docstring)。绝对阈值按调研给的 ~1e-5 量级放宽一档到 1e-3;
# 相对阈值 1e-4 * extent。任一通过即 PASS。
FK_SELFTEST_ABS_TOL = 1e-3
FK_SELFTEST_REL_TOL = 1e-4


# ----------------------------------------------------------------------------
# matplotlib headless (必须在 import pyplot 之前设 Agg)
# ----------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: F401


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# FK (linchpin) —— 行约定, 已验证正确
# ----------------------------------------------------------------------------
def fk_positions(rot6d_TJ6, offsets_J3, parents_J, torch_mod,
                 rotation_6d_to_matrix, bvh_forward):
    """
    rot6d_TJ6 : torch [T, J, 6]  (J = 有效关节数, 已截断, 无 padding)
    offsets_J3: torch [J, 3]     parent-local rest offsets (= anim.offsets / rest_pose)
    parents_J : torch [J]        root 的 parent = -1
    返回 pos[T,J,3], 已 root-center (减 joint0)。
    用 rotation_6d_to_matrix (行约定, transforms3d.py:578) + bvh_forward (rotation.py:98)。
    禁用 rot6d_to_fk_positions / rot6d_to_rotmat_* (列约定, 转置 bug)。
    """
    Rmat = rotation_6d_to_matrix(rot6d_TJ6)  # [T,J,3,3], 行约定
    T = rot6d_TJ6.shape[0]
    transl = torch_mod.zeros((T, 3), dtype=Rmat.dtype, device=Rmat.device)
    _, pos = bvh_forward(Rmat, transl, offsets_J3, parents_J)  # [T,J,3]
    pos = pos - pos[:, 0:1, :]  # root-center
    return pos


def fk_selftest(gt_rot6d_TJ6, gt_pos_TJ3, offsets_J3, parents_J, torch_mod,
                rotation_6d_to_matrix, bvh_forward):
    """
    硬闸门: FK(GT_rot6d) vs GT position (都 root-center) 比 max 误差。
    返回 (max_err, mean_err, extent, passed_bool)。
    任一判据 (abs<FK_SELFTEST_ABS_TOL 或 rel<FK_SELFTEST_REL_TOL) 通过即 PASS。
    """
    fk_gt = fk_positions(gt_rot6d_TJ6, offsets_J3, parents_J, torch_mod,
                         rotation_6d_to_matrix, bvh_forward)
    gt = gt_pos_TJ3 - gt_pos_TJ3[:, 0:1, :]  # GT 已 root-center, 再减一次无害
    diff = (fk_gt - gt).abs()
    max_err = float(diff.max().item())
    mean_err = float(diff.mean().item())
    # extent = 骨架尺度 (用 GT 位置的逐维跨度的最大值)
    gt_np = gt.detach().float().cpu().numpy()
    extent = float(max(
        gt_np[..., 0].max() - gt_np[..., 0].min(),
        gt_np[..., 1].max() - gt_np[..., 1].min(),
        gt_np[..., 2].max() - gt_np[..., 2].min(),
    ))
    extent = max(extent, 1e-6)
    passed = (max_err < FK_SELFTEST_ABS_TOL) or (max_err < FK_SELFTEST_REL_TOL * extent)
    return max_err, mean_err, extent, passed


# ----------------------------------------------------------------------------
# 渲染: GT vs pred 并排骨架动画 gif (固定相机, 共享坐标范围, J-agnostic)
# ----------------------------------------------------------------------------
def build_chains_from_parents(parents):
    """从 parent 数组建运动学链 (每条链 = 从某叶子回溯到 root 的关节序列, 用于画骨)。
    简化为画 (child, parent) 每条边; 返回 list of (i, parent_i)。"""
    edges = []
    for i in range(len(parents)):
        p = int(parents[i])
        if p >= 0:
            edges.append((i, p))
    return edges


def render_compare_gif(gt_pos, pred_pos, parents, save_path, fps=20, max_frames=60,
                       elev=12, azim=-90, vertical_axis="y", fig_w=22, fig_h=11,
                       title_prefix="", fk_trusted=True):
    """
    gt_pos, pred_pos: np [T, J, 3] (已 root-center)。
    并排 (左 GT 灰, 右 pred 红) 两个 3D 子图, 固定相机 + 共享立方坐标范围。
    fk_trusted=False 时在标题标 "FK UNTRUSTED"。
    """
    gt_pos = np.asarray(gt_pos, dtype=np.float64)
    pred_pos = np.asarray(pred_pos, dtype=np.float64)
    gt_pos = gt_pos - gt_pos[:, 0:1, :]
    pred_pos = pred_pos - pred_pos[:, 0:1, :]

    T = min(gt_pos.shape[0], pred_pos.shape[0])
    if max_frames is not None and T > max_frames:
        idx = np.linspace(0, T - 1, max_frames).astype(int)
        gt_pos = gt_pos[idx]
        pred_pos = pred_pos[idx]
        T = len(idx)

    edges = build_chains_from_parents(parents)

    # 共享立方范围 (两套 pose 一起算, 保证 GT/pred 同尺度可比)
    both = np.concatenate([gt_pos, pred_pos], axis=0)
    mids = [(both[..., d].max() + both[..., d].min()) / 2.0 for d in range(3)]
    half = max(
        both[..., 0].max() - both[..., 0].min(),
        both[..., 1].max() - both[..., 1].min(),
        both[..., 2].max() - both[..., 2].min(),
    ) / 2.0
    half = max(half, 1e-3) * 1.05

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_gt = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")

    trust_tag = "" if fk_trusted else "  [FK UNTRUSTED]"

    def setup(ax, color, sub):
        ax.set_xlim(mids[0] - half, mids[0] + half)
        ax.set_ylim(mids[1] - half, mids[1] + half)
        ax.set_zlim(mids[2] - half, mids[2] + half)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)
        ax.set_title(sub)
        pts, = ax.plot([], [], [], "o", color=color, markersize=3)
        lines = [ax.plot([], [], [], "-", color=color, lw=1.5, alpha=0.8)[0]
                 for _ in edges]
        return pts, lines

    pts_gt, lines_gt = setup(ax_gt, "0.35", "GT")
    pts_pred, lines_pred = setup(ax_pred, "crimson", "pred (FK of pred_rot6d)" + trust_tag)
    sup = fig.suptitle("", fontsize=11)

    def draw(ax_pts, ax_lines, P):
        ax_pts.set_data(P[:, 0], P[:, 1])
        ax_pts.set_3d_properties(P[:, 2])
        for ln, (i, p) in zip(ax_lines, edges):
            ln.set_data([P[i, 0], P[p, 0]], [P[i, 1], P[p, 1]])
            ln.set_3d_properties([P[i, 2], P[p, 2]])

    def update(f):
        draw(pts_gt, lines_gt, gt_pos[f])
        draw(pts_pred, lines_pred, pred_pos[f])
        sup.set_text(f"{title_prefix}  frame {f+1}/{T}{trust_tag}")
        return [pts_gt, pts_pred] + lines_gt + lines_pred + [sup]

    ani = FuncAnimation(fig, update, frames=T, interval=1000.0 / fps, blit=False)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return save_path


# ----------------------------------------------------------------------------
# checkpoint 选择
# ----------------------------------------------------------------------------
def find_latest_epoch_ckpt(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "pose2rot_ckpt_epoch*.pt"))
    if not files:
        raise FileNotFoundError(f"no pose2rot_ckpt_epoch*.pt under {ckpt_dir}")

    def epoch_num(p):
        m = re.search(r"epoch(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    files.sort(key=epoch_num)
    return files[-1], epoch_num(files[-1])


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="pose2rot 动作迁移可视化 QA")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint .pt; 默认取 ckpt_dir 下最新 epoch")
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--species", nargs="+",
                    default=["Horse", "Buffalo", "Camel", "Cat"],
                    help="物种短名列表 (# 前的部分)")
    ap.add_argument("--n_clips", type=int, default=2, help="每物种渲染几条 clip")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--gpu", type=int, default=1,
                    help="rose09 空闲 GPU index (默认 1, 不扰训练)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--elev", type=float, default=12.0, help="相机仰角")
    ap.add_argument("--azim", type=float, default=-90.0, help="相机方位角")
    ap.add_argument("--vertical_axis", default="y", choices=["x", "y", "z"],
                    help="哪个轴朝上(y=Y up, 配合 azim 让 +Z 朝屏幕)")
    ap.add_argument("--fig_w", type=float, default=22.0, help="figure 宽(英寸)")
    ap.add_argument("--fig_h", type=float, default=11.0, help="figure 高(英寸)")
    ap.add_argument("--fp32", action="store_true",
                    help="用 fp32 推理 (默认 bf16, 与训练一致)")
    args = ap.parse_args()

    # cwd 必须是仓库根 (import utils.* / models.* / data.*)
    os.chdir(REPO_ROOT)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    # GPU 绑定: 设 CUDA_VISIBLE_DEVICES 到指定 gpu, 进程内即 cuda:0
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from utils.config_utils import load_yaml_config, instantiate_from_config
    from utils.transforms3d import rotation_6d_to_matrix
    from utils.rotation import bvh_forward
    from utils import bvh as _qa_bvh
    from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded

    if args.device == "cuda" and not torch.cuda.is_available():
        log("WARNING: --device cuda 但 CUDA 不可用, 回退 cpu (fp32)")
        args.device = "cpu"
        args.fp32 = True
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    os.makedirs(args.out, exist_ok=True)

    # ---- 1. config + 模型 + ckpt ----
    cfg = load_yaml_config(args.config)
    model = instantiate_from_config(cfg["model"])
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path, ep = find_latest_epoch_ckpt(args.ckpt_dir)
        log(f"latest ckpt = {ckpt_path} (epoch {ep})")
    else:
        log(f"ckpt = {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd["model_state"] if "model_state" in sd else sd
    missing, unexpected = model.load_state_dict(state, strict=True)
    model.eval().to(device)
    log(f"model loaded. missing={len(missing)} unexpected={len(unexpected)}")

    # ---- 2. dataset (split=train 因 test 拆分为空; limit_species_debug 只扫指定物种) ----
    data_cfg = cfg["data"]
    ds = AnySpeciesPoseDataset(
        bvh_dir=data_cfg["bvh_dir"],
        window=data_cfg["seq_len"],
        mmap=data_cfg.get("mmap", True),
        cache_scale=data_cfg.get("cache_scale", True),
        limit_species_debug=list(args.species),  # 只扫这几个物种, 加速
        split_json=data_cfg["split_json"],
        split_mode="train",                       # test 拆分为空 -> 必用 train
        memory_pkl_path=data_cfg.get("train_memory_pkl_path", None),
        preload_all=False,
    )
    log(f"dataset items = {len(ds.items)} (species={args.species})")

    # 按物种分组 ds.items, 每物种取前 n_clips 条 (rel 排序后, 确定性)
    by_species = {}
    for i, it in enumerate(ds.items):
        sp = it["rel"].split("/")[0].split("#")[0]
        by_species.setdefault(sp, []).append((i, it["rel"]))
    for sp in by_species:
        by_species[sp].sort(key=lambda x: x[1])

    chosen = []  # (ds_idx, species, rel)
    for sp in args.species:
        lst = by_species.get(sp, [])
        if not lst:
            log(f"WARNING: 物种 {sp} 没有可用 clip (ds.items 里没有)")
            continue
        for ds_idx, rel in lst[:args.n_clips]:
            chosen.append((ds_idx, sp, rel))
    if not chosen:
        log("FATAL: 没有任何可渲染的 clip, 退出")
        sys.exit(2)

    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if (device.type == "cuda" and not args.fp32)
                    else _nullcontext())

    summary = {
        "ckpt": ckpt_path,
        "config": args.config,
        "device": str(device),
        "gpu_index": args.gpu if args.device == "cuda" else None,
        "dtype": "bf16" if (device.type == "cuda" and not args.fp32) else "fp32",
        "split_note": ("selected_test_split1010.json 为空 -> 全部 clip 训练时见过; "
                       "这是 reconstruction 质量, 非 generalization"),
        "fk_convention": ("rotation_6d_to_matrix(rows, transforms3d.py:578) + "
                          "bvh_forward(rotation.py:98); rot6d_to_fk_positions 禁用(列约定转置bug)"),
        "fk_selftest_abs_tol": FK_SELFTEST_ABS_TOL,
        "fk_selftest_rel_tol": FK_SELFTEST_REL_TOL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "clips": [],
    }

    n_fail_selftest = 0

    for ds_idx, sp, rel in chosen:
        clip_tag = rel.replace("/", "__")
        log(f"--- {sp} | {rel} ---")
        # 确定性: 固定 random (train 分支用 random 选 window/ref_idx)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        sample = ds[ds_idx]
        batch = collate_anyspecies_padded([sample])
        J = int(batch["J_valid"][0].item())

        batch_dev = {}
        for k, v in batch.items():
            batch_dev[k] = v.to(device) if torch.is_tensor(v) else v

        # GT (有效 J)
        gt_rot6d = batch_dev["rot6d_a"][0, :, :J, :].float()       # [W,J,6]
        # FK offsets: RAW parent-local anim.offsets / gscale (NOT loader's offset_a,
        # which root-subtracts -> corrupts child offsets when root offset != 0; codex fix).
        _anim = _qa_bvh.load(ds.items[ds_idx]["bvh"])[0]
        _gscale = float(ds.species_static_cache[sp]["global_scale"])
        offsets = torch.from_numpy(
            np.asarray(_anim.offsets[:J], dtype=np.float32) / _gscale
        ).to(device).float()                                       # [J,3] parent-local, no root subtract
        parents = batch_dev["parent_a"][0, :J].long()              # [J], root=-1
        gt_pos = batch_dev["position"][0, :, :J, :].float()        # [W,J,3]

        # ---- 3. FK 自检闸门 (fail-loud) ----
        st_max, st_mean, extent, st_pass = fk_selftest(
            gt_rot6d, gt_pos, offsets, parents, torch,
            rotation_6d_to_matrix, bvh_forward)
        if st_pass:
            log(f"  FK self-test PASS: max_err={st_max:.3e} mean={st_mean:.3e} "
                f"extent={extent:.2f} (rel={st_max/extent:.2e})")
        else:
            n_fail_selftest += 1
            log("  " + "!" * 70)
            log(f"  !! FK SELF-TEST FAIL: max_err={st_max:.3e} mean={st_mean:.3e} "
                f"extent={extent:.2f} (rel={st_max/extent:.2e})")
            log("  !! FK 约定与数据不一致 -> 该 clip 的 pred FK / MPJPE 不可信。")
            log("  !! gif 仍会出 (标 FK UNTRUSTED), 误差已写入 summary。")
            log("  " + "!" * 70)

        # ---- 4. forward -> pred_rot6d -> FK ----
        with torch.no_grad():
            with autocast_ctx:
                out = model(batch_dev)
        pred_rot6d = out["pred_rot6d"][0, :, :J, :].float()        # [W,J,6]

        pred_pos = fk_positions(pred_rot6d, offsets, parents, torch,
                                rotation_6d_to_matrix, bvh_forward)

        # ---- 5. metric ----
        # rot6d L1 (pred vs GT), 只在有效 J 上
        rot6d_l1 = float((pred_rot6d - gt_rot6d).abs().mean().item())
        # MPJPE (pred pos vs GT pos, 都 root-center); 仅自检通过时有意义
        gt_pos_c = gt_pos - gt_pos[:, 0:1, :]
        pred_pos_c = pred_pos - pred_pos[:, 0:1, :]
        mpjpe = float((pred_pos_c - gt_pos_c).norm(dim=-1).mean().item())

        gif_path = os.path.join(args.out, f"{clip_tag}.gif")
        render_compare_gif(
            gt_pos_c.detach().cpu().numpy(),
            pred_pos_c.detach().cpu().numpy(),
            parents.detach().cpu().numpy(),
            gif_path,
            fps=args.fps, max_frames=args.max_frames,
            elev=args.elev, azim=args.azim, vertical_axis=args.vertical_axis,
            fig_w=args.fig_w, fig_h=args.fig_h,
            title_prefix=f"{sp} | {rel}",
            fk_trusted=st_pass,
        )
        log(f"  rot6d_L1={rot6d_l1:.4f}  MPJPE={mpjpe:.4f}"
            f"{'' if st_pass else ' (UNTRUSTED)'}  -> {gif_path}")

        summary["clips"].append({
            "species": sp,
            "clip": rel,
            "J": J,
            "fk_selftest_maxerr": st_max,
            "fk_selftest_meanerr": st_mean,
            "fk_selftest_extent": extent,
            "fk_selftest_pass": bool(st_pass),
            "rot6d_l1": rot6d_l1,
            "mpjpe": mpjpe,
            "mpjpe_trusted": bool(st_pass),
            "gif": gif_path,
        })

    summary["n_clips"] = len(summary["clips"])
    summary["n_fk_selftest_fail"] = n_fail_selftest

    summary_path = os.path.join(args.out, "qa_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log("=" * 60)
    log(f"DONE. {len(summary['clips'])} clips -> {args.out}")
    log(f"FK self-test fails: {n_fail_selftest}/{len(summary['clips'])}")
    log(f"summary: {summary_path}")
    if n_fail_selftest > 0:
        log("WARNING: 有 clip FK 自检未过 -> 那些 MPJPE/视觉不可信, 见 summary 与 gif 标注。")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
