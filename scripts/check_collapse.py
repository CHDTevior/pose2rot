"""Quick collapse check: load latest ckpt of an exp, report var(pred)/var(GT) on
non-static (predicted) joints + position sensitivity. ratio_DYN ~0 = collapsed
(frozen constant); >0.3 = collapse broken (model produces time-varying motion).
Usage: python check_collapse.py <config.yaml> <exp_name>
"""
import sys, glob, random
import numpy as np, torch
from utils.config_utils import load_yaml_config, instantiate_from_config
from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded

cfg_path, exp = sys.argv[1], sys.argv[2]
cfg = load_yaml_config(cfg_path)
model = instantiate_from_config(cfg['model']).eval().cuda()
cks = sorted(glob.glob(f'checkpoints/pose2rot/{exp}/pose2rot_ckpt_epoch*.pt'),
             key=lambda p: int(p.split('epoch')[-1].split('.')[0]))
if not cks:
    print(f'NO_CKPT for {exp}'); sys.exit(0)
# optional 3rd arg: explicit ckpt path (else latest)
ck = sys.argv[3] if len(sys.argv) > 3 else cks[-1]
model.load_state_dict(torch.load(ck, map_location='cpu')['model_state'])
d = cfg['data']
ds = AnySpeciesPoseDataset(bvh_dir=d['bvh_dir'], split_json=d.get('split_json'),
                           split_mode='train', memory_pkl_path=d['train_memory_pkl_path'],
                           preload_all=False)  # on-demand: only read the ~8 clips we check (fast)
want = ['Horse#', 'Cat#', 'Camel#', 'Buffalo#']
idxs = [i for i, it in enumerate(ds.items) if any(it['rel'].startswith(w) for w in want)][:8]
random.seed(1); np.random.seed(1); torch.manual_seed(1)
rd, ss, rl = [], [], []
for i in idxs:
    b = collate_anyspecies_padded([ds[i]]); bd = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    J = int(b['J_valid'][0]); stat = bd['static_rot_joint_mask'][0, :J].bool().cpu().numpy()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        pr = model(bd)['pred_rot6d'][0, :, :J].float()
    gt = bd['rot6d_a'][0, :, :J].float()
    vp = pr.var(dim=0).mean(dim=1).cpu().numpy(); vg = gt.var(dim=0).mean(dim=1).cpu().numpy()
    dyn = ~stat
    rdyn = vp[dyn].mean() / (vg[dyn].mean() + 1e-12) if dyn.any() else float('nan')
    bd2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in bd.items()}
    bd2['position'] = torch.zeros_like(bd2['position'])
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        pr2 = model(bd2)['pred_rot6d'][0, :, :J].float()
    ss.append((pr - pr2).abs().mean().item() / (pr.abs().mean().item() + 1e-9))
    rl.append((pr - gt)[:, torch.from_numpy(dyn)].abs().mean().item()); rd.append(rdyn)
verdict = 'BROKEN(good)' if np.nanmean(rd) > 0.3 else ('PARTIAL' if np.nanmean(rd) > 0.05 else 'COLLAPSED')
print(f'{ck.split("/")[-1]} | ratio_DYN={np.nanmean(rd):.3f} pos_sens={np.mean(ss):.3f} rot6d_L1={np.mean(rl):.3f} -> {verdict}')
