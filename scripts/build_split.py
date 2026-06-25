import json
BASE = ".."
data = json.load(open(f"{BASE}/zoo1030_build/dataset_motions.json"))
sp_motions = {k: sorted(v) for k, v in data["species_motions"].items()}
motion_angles = data["motion_angles"]
UNSEEN = ["Coyote","Cat","Goat","Pigeon","Comodoa","Spider"]
SEEN   = ["Trex","Raptor2","Horse","BrownBear","Alligator","Buffalo","Anaconda","Camel","Bird","Elephant","Lion","Eagle"]
RARE   = ["Flamingo","Hamster","Scorpion","Chicken","Rat","Puppy","Parrot2","Skunk","Crow","Tukan","FireAnt","PolarBearB"]
SEEN_HOLD, RARE_HOLD = 2, 1
def ang_nums(m): return [a[1:] for a in sorted(motion_angles[m])]
split = {"seen":{}, "rare":{}, "unseen":{}}
miss = [s for s in UNSEEN+SEEN+RARE if s not in sp_motions]
if miss: print("MISSING:", miss); raise SystemExit(1)
for sp in UNSEEN:
    for m in sp_motions[sp]: split["unseen"][m] = ang_nums(m)
for sp in SEEN:
    for m in sp_motions[sp][:SEEN_HOLD]: split["seen"][m] = ang_nums(m)
for sp in RARE:
    for m in sp_motions[sp][:RARE_HOLD]: split["rare"][m] = ang_nums(m)
out = f"{BASE}/MocapAnything/datasets/zoo1030/test_split_seen_rare_unseen.json"
json.dump(split, open(out,"w"), indent=1)
print(f"WROTE {out}\n")
tm=tc=0
for g in ["seen","rare","unseen"]:
    ms=list(split[g]); nc=sum(len(v) for v in split[g].values()); sps=sorted(set(m.split('#')[0] for m in ms))
    tm+=len(ms); tc+=nc
    print(f"{g:7s}: {len(ms):3d} motions / {nc:4d} clips / {len(sps)} sp  {sps}")
print(f"\nTOTAL TEST: {tm} motions ({tm*100//823}% of 823) / {tc} clips")
print(f"unseen species fully removed from train: {sorted(UNSEEN)}")
