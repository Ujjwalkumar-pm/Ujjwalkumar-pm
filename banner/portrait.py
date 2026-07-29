"""Photo -> 1-bit dithered dot grid.

WHY THIS DOES NOT USE BACKGROUND SEGMENTATION
=============================================
The Master Prompt's dark-mode recipe is "segment the background out (threshold
on colour distance, binary closing, fill holes, keep largest component)". That
recipe assumes the photo it asks for: a flat, uniform backdrop with clear
subject separation. The source here is the opposite -- a low-key studio portrait
on smoke, measured at:

    suit luminance        37.5
    backdrop beside it    39.7      <- 2 levels apart, and the same blue hue
    backdrop range        20..122   <- nowhere near flat

Running the prescribed segmentation across thresholds 18..55 confirmed there is
no operating point: at t=18 the "subject" is 86.9% of the frame (the smoke is
inside the mask); by t=40 the backdrop is gone but only 43% of the torso
survives, and the mask has broken into 14+ components. No threshold separates
this subject.

What DOES work is the photo's own lighting. Keyed on luminance:

    backdrop  0.0% ink        face   57-66% ink
    haze      0.0-0.4% ink    shirt  ~96% ink
                              suit   16-31% ink  (dissolves into the panel)

That is the same visual outcome segmentation was meant to produce -- dots draw
the lit subject, the background stays empty -- reached without a mask this
photo cannot support. The lit right shoulder carries the head-and-shoulders
silhouette, so the result is not a tight face crop.

Everything else follows the prompt: 300x340 grid, 1-bit Floyd-Steinberg in
serpentine order, 1.3x contrast only, autocontrast(cutoff=1) and
UnsharpMask(radius=3, percent=140). Single hue -- all tone comes from dot
density, never from per-dot colour.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SRC = "banner/src/photo.jpg"
GRID_W, GRID_H = 300, 340

# Tone curve. The shirt sits at mean 200 and would clip to a solid slab, which
# reads as a white rectangle rather than fabric; the shoulders sit at ~96 and
# need to survive. `floor` is subtracted first so the backdrop lands hard at
# zero (no stray dots in the panel), then the range is stretched.
FLOOR = 70.0
CEIL = 232.0
GAMMA = 1.05


def load_grid():
    """Photo -> float array in 0..1 on the dot grid, keyed on lit subject."""
    im = Image.open(SRC).convert("L")

    # Head and shoulders, not a tight face crop -- the prompt is explicit that
    # an over-zoomed crop reads aggressive.
    #
    # The source is framed wide: shoulders run nearly edge to edge while the
    # head is small and central. Rendered at the full frame the face lands at
    # roughly a third of the panel and the detail the dither works hardest on
    # is the part you cannot read. Trimming the lower chest and the side margins
    # brings the head up without crossing into a face crop -- both shoulders and
    # the collar still frame it.
    w, h = im.size
    im = im.crop((int(w * 0.07), 0, int(w * 0.93), int(h * 0.74)))

    im = ImageOps.autocontrast(im, cutoff=1)
    im = ImageEnhance.Contrast(im).enhance(1.3)
    im = im.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    im = im.resize((GRID_W, GRID_H), Image.LANCZOS)

    a = np.asarray(im).astype(np.float64)
    a = (a - FLOOR) / (CEIL - FLOOR)
    a = np.clip(a, 0.0, 1.0) ** GAMMA
    return a


def floyd_steinberg(a):
    """1-bit Floyd-Steinberg, serpentine.

    Serpentine matters: raster order pushes error consistently rightward and
    lays down visible diagonal worming across a face. Alternating direction
    cancels it.
    """
    a = a.copy()
    h, w = a.shape
    out = np.zeros((h, w), dtype=bool)
    for y in range(h):
        rng = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        step = 1 if y % 2 == 0 else -1
        for x in rng:
            old = a[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new > 0.5
            err = old - new
            if 0 <= x + step < w:
                a[y, x + step] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - step < w:
                    a[y + 1, x - step] += err * 3 / 16
                a[y + 1, x] += err * 5 / 16
                if 0 <= x + step < w:
                    a[y + 1, x + step] += err * 1 / 16
    return out


def light_grid(tone):
    """Light-mode tone: ink where the subject is DARK, but only inside it.

    The prompt's light-mode rule is "keep the background; dots draw the dark
    parts of the photo". That is written for a photo with a LIGHT backdrop --
    there, the dark parts are the subject. This backdrop is nearly black, so
    applying the rule literally inks the entire background and produces a slab
    with a person-shaped hole in it.

    Reusing the dark-mode dot set instead is no better: it is a tonal negative,
    and the brightest thing in frame (the shirt, mean 200) becomes a solid
    violet block.

    So the rule is applied where it was meant to apply -- inside the subject.
    The lit key says WHERE the subject is; within that region ink runs with
    darkness. Background stays empty, highlights stay open, and the face reads
    as a positive image in both themes.
    """
    # The region test has to be a real silhouette, not "any ink at all". At a
    # 0.02 cut the leftover smoke qualifies, and inverting it sends those cells
    # to ~1.0 -- solid violet blobs, worse than the haze they came from. Cut
    # high, close the gaps, fill, and keep the one big component. This is the
    # prompt's own morphology; it works here because the LIT key gives a single
    # connected subject, which the colour-distance key never did.
    from scipy import ndimage

    def disk(r):
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        return x * x + y * y <= r * r

    # Square structuring elements leave visible stair-steps along the shoulder
    # -- the same "you built a grid" failure the prompt warns about, showing up
    # on the silhouette instead of in the drift. A disk plus a light blur before
    # thresholding keeps the boundary organic.
    smooth = ndimage.gaussian_filter(tone, 1.6)
    region = smooth > 0.15
    region = ndimage.binary_closing(region, disk(5))
    region = ndimage.binary_fill_holes(region)
    lab, n = ndimage.label(region)
    if n > 1:
        region = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    region = ndimage.binary_dilation(region, disk(2))

    inv = np.zeros_like(tone)
    inv[region] = 1.0 - tone[region]
    # Compress so the deep suit shadows keep texture instead of flooding solid
    # and swallowing the lapel structure.
    inv = np.clip((inv - 0.12) / 0.95, 0.0, 1.0)
    inv[~region] = 0.0
    return inv


def build():
    tone = load_grid()
    dots = floyd_steinberg(tone)
    ys, xs = np.nonzero(dots)
    pts = np.stack([xs, ys], 1).astype(np.int16)

    ltone = light_grid(tone)
    ldots = floyd_steinberg(ltone)
    lys, lxs = np.nonzero(ldots)
    lpts = np.stack([lxs, lys], 1).astype(np.int16)

    np.save("banner/portrait_dots.npy", pts)
    np.save("banner/portrait_dots_light.npy", lpts)
    np.save("banner/portrait_tone.npy", tone.astype(np.float32))
    print(f"light dots      {len(lpts):,}  "
          f"({len(lpts)/(GRID_W*GRID_H)*100:.1f}% ink)")

    # Report the numbers the prompt says to verify by measurement, not by eye.
    band = tone[:40, :]
    print(f"grid            {GRID_W}x{GRID_H} = {GRID_W*GRID_H:,} cells")
    print(f"dots            {len(pts):,}  ({len(pts)/(GRID_W*GRID_H)*100:.1f}% ink)")
    print(f"backdrop ink    {dots[:40, :].mean()*100:.2f}%  (top band)")
    print(f"corner ink      {dots[:30, :30].mean()*100:.2f}%")
    print(f"face ink        {dots[60:150, 120:200].mean()*100:.1f}%")
    print(f"shoulder ink    {dots[210:300, 190:270].mean()*100:.1f}%")
    return pts


if __name__ == "__main__":
    build()
