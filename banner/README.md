# Banner generator

`dark.svg` and `light.svg` at the repo root are **build output**. The source of
truth is this folder — the scripts plus the `.npy` data. Edit the SVGs by hand
and the next build overwrites you.

## Rebuild

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy pillow

.venv/bin/python banner/portrait.py    # photo   -> dot grids   (.npy)
.venv/bin/python banner/logos.py       # shapes  -> point clouds (.npy)
.venv/bin/python banner/build_svg.py   # data    -> dark.svg / light.svg
.venv/bin/python banner/shoot.py dark.svg   # screenshots at 5 animation times
```

`shoot.py` renders through headless Chrome, not cairosvg. cairosvg draws only
the first SMIL frame and mishandles `textLength`, so it would report every later
animation phase as identical and every locked row as misaligned.

## What the numbers mean

`build_svg.py` prints two metrics on every run. Both exist to catch a specific
failure that looks fine in a still frame and wrong in motion.

| Metric | Good | Bad | What it catches |
|---|---|---|---|
| intro evenness | ~0.15 | ~0.86 | Groups fading in as spatial blocks or a wipe instead of shimmering in everywhere at once |
| noise moved … between bands | ~70% | ~0% | Drift bands quantised straight off a linear function — a square grid, so the dissolve looks blocky |

The evenness figure is the sampling floor for random interleaving at this dot
count, not a target of 0.05: measured against the same portrait, contiguous
spatial blocks score 0.86 and a vertical wipe 0.90.

## The one thing that would improve this most

A better source photo. `src/photo.jpg` is the 382×382 GitHub avatar — the
Master Prompt asks for 1000px+ on a flat backdrop, and this is neither.

`portrait.py` documents at length why the prompt's background segmentation is
not used here: the suit sits at luminance 37.5 against a backdrop of 39.7 in the
same hue, so no threshold separates them. Keying on the photo's own lighting
gets the same result, but a photo shot against a plain wall would allow the
documented method, a sharper dither, and a stronger light mode.

Replace `src/photo.jpg` and re-run — nothing downstream needs to change.
