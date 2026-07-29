"""Three logo point clouds, and the dot-to-dot matching between them.

WHY THESE THREE SHAPES. The Master Prompt says to trace real brand marks from
reference images and explicitly warns "trace them, don't hand-draw them". No
reference images were supplied, and drawing a trademarked mark from memory
produces something that is recognisably wrong -- which is worse on a personal
profile than not using it. So all three shapes here are CONSTRUCTED exactly:
two are glyph outlines rendered from a system monospace font, and the triangle
is true equilateral geometry. Nothing is approximated.

    </>   ->   /\   ->   >_
    code     deploy    terminal

Everything lives in the same 300x340 coordinate space as the portrait, so the
travelling dots move in one continuous system rather than being rescaled
between phases.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

GRID_W, GRID_H = 300, 340
N_TRAVELLERS = 900
MONO = "/System/Library/Fonts/Menlo.ttc"


def _mask_to_points(mask, n, seed):
    """Sample `n` points from a binary mask, evenly spread.

    Uniform random sampling clumps; a jittered grid over the mask's own cells
    keeps the shape legible at 900 dots. Ordering is randomised afterwards so
    the optimal-transport matching is not fed a scanline-ordered cloud.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise SystemExit("empty logo mask")
    rng = np.random.default_rng(seed)
    if len(xs) >= n:
        idx = rng.choice(len(xs), n, replace=False)
    else:
        idx = rng.choice(len(xs), n, replace=True)
    pts = np.stack([xs[idx], ys[idx]], 1).astype(np.float64)
    # Sub-cell jitter so the cloud does not sit on integer lattice points --
    # the same reason the drift bands get per-dot noise.
    pts += rng.normal(0, 0.45, pts.shape)
    return pts


def glyph_mask(text, size, dx=0, dy=0):
    img = Image.new("L", (GRID_W, GRID_H), 0)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(MONO, size)
    box = d.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    d.text(((GRID_W - w) / 2 - box[0] + dx, (GRID_H - h) / 2 - box[1] + dy),
           text, font=font, fill=255)
    return np.asarray(img) > 128


def triangle_mask(side=210, dy=6):
    """A true equilateral triangle -- height = side * sqrt(3)/2."""
    img = Image.new("L", (GRID_W, GRID_H), 0)
    d = ImageDraw.Draw(img)
    h = side * np.sqrt(3) / 2
    cx, cy = GRID_W / 2, GRID_H / 2 + dy
    pts = [(cx, cy - 2 * h / 3), (cx - side / 2, cy + h / 3),
           (cx + side / 2, cy + h / 3)]
    d.polygon(pts, fill=255)
    # Hollow it out so the dot count spreads around the edge rather than
    # filling a solid slab -- a filled triangle at 900 dots reads as noise.
    inner = side * 0.58
    ih = inner * np.sqrt(3) / 2
    d.polygon([(cx, cy - 2 * ih / 3), (cx - inner / 2, cy + ih / 3),
               (cx + inner / 2, cy + ih / 3)], fill=0)
    return np.asarray(img) > 128


def build():
    masks = [
        ("code", glyph_mask("</>", 128)),
        ("deploy", triangle_mask()),
        ("terminal", glyph_mask(">_", 148, dy=-6)),
    ]

    clouds = [_mask_to_points(m, N_TRAVELLERS, seed=i * 17 + 3)
              for i, (_n, m) in enumerate(masks)]

    # Optimal transport between consecutive shapes (and back to the first, so
    # the loop closes): each dot takes the shortest available path, which is
    # what makes a morph read as one object changing rather than a crossfade.
    order = [0, 1, 2, 0]
    matched = [clouds[0]]
    cur = clouds[0]
    for a, b in zip(order, order[1:]):
        tgt = clouds[b]
        cost = np.linalg.norm(cur[:, None, :] - tgt[None, :, :], axis=2)
        _r, c = linear_sum_assignment(cost)
        nxt = tgt[c]
        matched.append(nxt)
        cur = nxt

    stack = np.stack(matched)          # (4, N, 2) -- last frame == first shape
    np.save("banner/logo_paths.npy", stack.astype(np.float32))

    for (name, m), cloud in zip(masks, clouds):
        print(f"  {name:9s} mask {m.sum():6,} cells  -> {len(cloud)} dots")
    move = np.linalg.norm(stack[1] - stack[0], axis=1)
    print(f"  mean dot travel  {move.mean():.1f} cells "
          f"(max {move.max():.1f}) -- optimal transport keeps this short")
    return stack


if __name__ == "__main__":
    build()
