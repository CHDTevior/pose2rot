"""Extract evenly-spaced frames from each QA gif and stack vertically into a montage PNG
so the motion (and GT-vs-PRED tracking) can be eyeballed in a single static image."""
import os, sys, glob
from PIL import Image, ImageSequence

OUTDIR = "v11c_visual_qa"
NFRAMES = 6  # sampled timesteps per gif


def montage(gif_path, n=NFRAMES):
    im = Image.open(gif_path)
    frames = [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)]
    if not frames:
        return None
    total = len(frames)
    idxs = [round(i * (total - 1) / (n - 1)) for i in range(n)] if total > 1 else [0]
    sel = [frames[i] for i in idxs]
    w, h = sel[0].size
    out = Image.new("RGB", (w, h * len(sel)), "white")
    for r, fr in enumerate(sel):
        out.paste(fr, (0, r * h))
    base = os.path.splitext(os.path.basename(gif_path))[0]
    op = os.path.join(OUTDIR, f"_montage_{base}.png")
    # downscale width to keep file manageable
    if out.width > 1500:
        scale = 1500 / out.width
        out = out.resize((1500, int(out.height * scale)))
    out.save(op)
    print(f"{op}  (sampled frames {idxs} of {total})")
    return op


if __name__ == "__main__":
    for g in sorted(glob.glob(os.path.join(OUTDIR, "*.gif"))):
        montage(g)
    print("MONTAGE DONE")
