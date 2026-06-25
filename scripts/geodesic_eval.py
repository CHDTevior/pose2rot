"""Geodesic angle error (deg) eval on held-out test split, per-species AND per-tier
(seen/rare/unseen), to compare head-to-head with MoCapAnything (V2 unseen 6.54°, V1 ~17°).
Ang(all)=over all valid joints (their likely def; static joints overwritten w/ ref => deflates).
Ang(dyn)=dynamic (non-static) joints only (honest, harder). Self-check: geodesic(GT,GT)~0.
Usage: python geodesic_eval.py <config.yaml> <exp_name> [ckpt_path]
"""
import sys, glob, json, collections
import numpy as np, torch
from utils.config_utils import load_yaml_config, instantiate_from_config
from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded
from utils.transforms3d import rotation_6d_to_matrix

def geodesic_deg(r6_pred, r6_gt):
    Rp = rotation_6d_to_matrix(r6_pred); Rg = rotation_6d_to_matrix(r6_gt)
    Rrel = torch.matmul(Rp.transpose(-1,-2), Rg)
    tr = Rrel[...,0,0]+Rrel[...,1,1]+Rrel[...,2,2]
    return torch.rad2deg(torch.arccos(torch.clamp((tr-1.0)*0.5, -1.0, 1.0)))

cfg_path, exp = sys.argv[1], sys.argv[2]
cfg = load_yaml_config(cfg_path)
model = instantiate_from_config(cfg['model']).eval().cuda()
cks = sorted(glob.glob(f'checkpoints/pose2rot/{exp}/pose2rot_ckpt_epoch*.pt'),
             key=lambda p: int(p.split('epoch')[-1].split('.')[0]))
if not cks: print(f'NO_CKPT for {exp}'); sys.exit(0)
ck = sys.argv[3] if len(sys.argv) > 3 else cks[-1]
model.load_state_dict(torch.load(ck, map_location='cpu')['model_state'])
d = cfg['data']
# motion -> tier map from split json
motion2tier = {}
sj = json.load(open(d['split_json']))
for tier in ['seen','rare','unseen']:
    for m in sj.get(tier, {}): motion2tier[m] = tier
ds = AnySpeciesPoseDataset(bvh_dir=d['bvh_dir'], split_json=d.get('split_json'),
        split_mode='test', memory_pkl_path=d.get('test_memory_pkl_path', d['train_memory_pkl_path']), preload_all=False)
print(f'ckpt={ck.split("/")[-1]}  test clips={len(ds.items)}', flush=True)
b0 = collate_anyspecies_padded([ds[0]]); bd0 = {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in b0.items()}
J0=int(b0['J_valid'][0]); gt0=bd0['rot6d_a'][0,:,:J0].float()
print(f'SELF-CHECK geodesic(GT,GT) max = {geodesic_deg(gt0,gt0).max().item():.6f} deg (must ~0)', flush=True)

per_sp = collections.defaultdict(lambda: {'aa':[],'ad':[],'rl':[]})
per_tier = collections.defaultdict(lambda: {'aa':[],'ad':[]})
for i in range(len(ds.items)):
    motion = ds.items[i]['rel'].split('/')[0]
    sp = motion.split('#')[0]; tier = motion2tier.get(motion, '?')
    b = collate_anyspecies_padded([ds[i]]); bd={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in b.items()}
    J=int(b['J_valid'][0]); stat=bd['static_rot_joint_mask'][0,:J].bool()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        pr = model(bd)['pred_rot6d'][0,:,:J].float()
    gt = bd['rot6d_a'][0,:,:J].float(); ang = geodesic_deg(pr, gt); dyn=~stat
    aa = ang.mean().item(); per_sp[sp]['aa'].append(aa); per_tier[tier]['aa'].append(aa)
    if dyn.any():
        ad = ang[:,dyn].mean().item(); per_sp[sp]['ad'].append(ad); per_tier[tier]['ad'].append(ad)
        per_sp[sp]['rl'].append((pr-gt)[:,dyn].abs().mean().item())

print(f'\n=== PER-TIER (clip-mean) ===', flush=True)
print(f'{"tier":8s} {"clips":>5s} {"Ang(all)":>9s} {"Ang(dyn)":>9s}', flush=True)
for t in ['seen','rare','unseen']:
    s=per_tier[t]
    if s['aa']: print(f'{t:8s} {len(s["aa"]):5d} {np.mean(s["aa"]):8.2f}° {np.mean(s["ad"]):8.2f}°', flush=True)
allaa=[x for t in ['seen','rare','unseen'] for x in per_tier[t]['aa']]
allad=[x for t in ['seen','rare','unseen'] for x in per_tier[t]['ad']]
print(f'{"ALL":8s} {len(allaa):5d} {np.mean(allaa):8.2f}° {np.mean(allad):8.2f}°', flush=True)
print(f'\n=== PER-SPECIES ===\n{"species":16s} {"n":>3s} {"Ang(all)":>9s} {"Ang(dyn)":>9s} {"rot6dL1":>8s}', flush=True)
for sp in sorted(per_sp):
    s=per_sp[sp]; ad=np.mean(s['ad']) if s['ad'] else float('nan'); rl=np.mean(s['rl']) if s['rl'] else float('nan')
    tier=motion2tier.get([m for m in sj.get('seen',{})|sj.get('rare',{})|sj.get('unseen',{}) if m.startswith(sp+'#')][0],'?') if False else ''
    print(f'{sp:16s} {len(s["aa"]):3d} {np.mean(s["aa"]):8.2f}° {ad:8.2f}° {rl:8.4f}', flush=True)
print('--- 对标 MoCapAnything: V2 unseen 6.54° / V1 ~17° (注意他们Ang多半含static关节=偏低; 我们Ang(all)同口径, Ang(dyn)更honest) ---', flush=True)
