"""Assemble dark.svg / light.svg -- the animated terminal banner.

STRUCTURE
    window chrome  -> title bar, traffic lights, `profile.sh --live`
    left ~38%      -> VISUAL.MAP portrait frame
    right          -> SYSTEM.INFO readout, dotted leaders, LIVE badge, handle pill

THREE DOT LAYERS, and they cannot be merged
    intro       60 interleaved random groups, fade in over ~2s, once
    loop        the same dots regrouped into ~94 drift bands, 14.2s, forever
    travellers  ~900 dots morphing </> -> triangle -> >_ , hidden during portrait

The intro and loop layers hold the SAME dots under DIFFERENT groupings, so they
have to be two separate sets of paths -- that is the duplicate portrait layer
the Master Prompt warns about, and merging them would mean one grouping serving
two different animations, which it cannot.

WHY THE DRIFT BANDS GET NOISE FIRST
    Drift is a linear function of position. Quantising a linear function into
    bands mechanically reproduces straight, evenly spaced boundaries -- a square
    grid -- and the dissolve reads blocky rather than organic. Per-dot Gaussian
    noise (sigma 4, about one band width) is added to the projection BEFORE
    quantising, so band boundaries interlock. build() reports how many dots the
    noise moved between bands; if that number is near zero the grid is back.

Dots are <path> runs with shape-rendering="crispEdges" -- never font glyphs,
which mush below ~2px.
"""

import numpy as np
from PIL import ImageFont

W, H = 1180, 610
GRID_W, GRID_H = 300, 340

N_INTRO_GROUPS = 60
N_BANDS = 94
DRIFT_FRACTION = 0.42
NOISE_SIGMA = 4.0

# Loop timeline. Explicitly uneven: evenly spaced keyTimes would force every
# phase to hold the same length, and the portrait needs longer than a logo.
T_PORTRAIT, T_LOGO, T_TRANS = 3.0, 2.0, 1.3
LOOP = T_PORTRAIT + 4 * T_TRANS + 3 * T_LOGO          # 14.2s
INTRO_FADE, INTRO_HOLD = 2.0, 3.2

MONO_STACK = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
MONO_FILE = "/System/Library/Fonts/Menlo.ttc"

THEMES = {
    "dark": dict(bg="#0A101F", panel="#0C1424", stroke="#15304A",
                 chrome="#22D3EE", portrait="#A78BFA", accent="#10B981",
                 text="#C7D2E1", muted="#5A7085", leader="#20364B",
                 title="#8FA3B8", pillbg="#10B981", pilltx="#04140C"),
    "light": dict(bg="#F4F5F8", panel="#FFFFFF", stroke="#D3DCE6",
                  chrome="#0891B2", portrait="#7C3AED", accent="#059669",
                  text="#1F2A37", muted="#64748B", leader="#D9E1EA",
                  title="#475569", pillbg="#059669", pilltx="#FFFFFF"),
}

ROWS = [
    ("Subject",        "Ujjwal Kumar"),
    ("Role",           "Product Manager"),
    ("Origin",         "Bengaluru, India"),
    ("Education",      "PMP · Project Management Institute"),
    ("Status",         "Shipping @ Smartworks · 6+ yrs"),
    ("ToolChain",      "Claude Code, Cursor, Figma, Vercel"),
    (None, None),
    ("Core.Lang",      "TypeScript · SQL · Python"),
    ("Core.Frontend",  "Next.js · React · Tailwind"),
    ("Core.Backend",   "Node · Drizzle ORM"),
    ("Core.Database",  "Postgres (Neon)"),
    ("Core.Infra",     "Vercel · GitHub Actions"),
    (None, None),
    ("Grid.Mail",      "ujjwalkumarjob1@gmail.com"),
    ("Grid.Portfolio", "ujjwalkumarr.vercel.app"),
    ("Grid.LinkedIn",  "/in/ujjwalkumarpm"),
    ("Grid.GitHub",    "Ujjwalkumar-pm"),
    ("Grid.Design",    "ujjwalkumar.designfolio.me"),
]

FS_ROW, FS_HEAD, FS_LIVE, FS_PILL, ROW_STEP = 14, 13, 12, 14, 23

# Geometry
WIN = (16, 16, W - 32, H - 32)
BAR_H = 42
PANEL_Y, PANEL_H = 78, 496
PORT = (40, PANEL_Y, 414, PANEL_H)
INFO = (478, PANEL_Y, W - 478 - 40, PANEL_H)
LX = INFO[0] + 18
RX = INFO[0] + INFO[2] - 18


def measure(text, size):
    """Natural advance width, used as textLength so the row cannot reflow."""
    f = ImageFont.truetype(MONO_FILE, size)
    return f.getlength(text)


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def runs_to_path(pts):
    """Run-length encode a dot set into one path.

    Consecutive cells on a row become `M x y h N v 1 h -N z` instead of N
    separate squares -- roughly a third of the bytes, and identical output.
    """
    if len(pts) == 0:
        return ""
    order = np.lexsort((pts[:, 0], pts[:, 1]))
    p = pts[order]
    out, i, n = [], 0, len(p)
    while i < n:
        x0, y = int(p[i, 0]), int(p[i, 1])
        j = i + 1
        while j < n and int(p[j, 1]) == y and int(p[j, 0]) == int(p[j - 1, 0]) + 1:
            j += 1
        ln = j - i
        out.append(f"M{x0} {y}h{ln}v1h-{ln}z")
        i = j
    return "".join(out)


def keytimes():
    """Uneven keyTimes for the 14.2s loop, as fractions."""
    t = [0.0, T_PORTRAIT]
    for k in range(3):
        t.append(t[-1] + T_TRANS)
        t.append(t[-1] + T_LOGO)
    t.append(t[-1] + T_TRANS)
    return [round(v / LOOP, 5) for v in t], t


def build_bands(pts, logo_centroid, rng):
    """Group the portrait into N_BANDS drift bands, noise applied first."""
    centre = pts.mean(0)
    direction = logo_centroid - centre
    direction = direction / (np.linalg.norm(direction) + 1e-9)

    raw = pts @ direction
    noisy = raw + rng.normal(0, NOISE_SIGMA, len(pts))

    def to_bands(vals):
        order = np.argsort(vals)
        band = np.empty(len(vals), np.int32)
        edges = np.array_split(order, N_BANDS)
        for b, idx in enumerate(edges):
            band[idx] = b
        return band

    band = to_bands(noisy)
    clean = to_bands(raw)
    moved = (band != clean).mean()
    return band, direction, moved


def evenness(pts, group, n_groups):
    """Total-variation distance between each group's spatial histogram and the
    whole portrait's. Random interleaved groups -> small. Spatial regions -> ~1.
    """
    bins = 10
    gx = np.clip((pts[:, 0] / GRID_W * bins).astype(int), 0, bins - 1)
    gy = np.clip((pts[:, 1] / GRID_H * bins).astype(int), 0, bins - 1)
    cell = gy * bins + gx
    overall = np.bincount(cell, minlength=bins * bins).astype(float)
    overall /= overall.sum()
    tv = []
    for g in range(n_groups):
        m = group == g
        h = np.bincount(cell[m], minlength=bins * bins).astype(float)
        if h.sum() == 0:
            continue
        h /= h.sum()
        tv.append(0.5 * np.abs(h - overall).sum())
    return float(np.mean(tv))


def build(theme_name):
    t = THEMES[theme_name]
    dots = np.load("banner/portrait_dots.npy" if theme_name == "dark"
                   else "banner/portrait_dots_light.npy").astype(np.int32)
    logos = np.load("banner/logo_paths.npy")          # (4, 900, 2)
    rng = np.random.default_rng(7)

    # ---- portrait placement ------------------------------------------------
    pad = 16
    ix, iy = PORT[0] + pad, PORT[1] + pad + 26
    iw, ih = PORT[2] - 2 * pad, PORT[3] - 2 * pad - 26
    scale = min(iw / GRID_W, ih / GRID_H)
    ox = ix + (iw - GRID_W * scale) / 2
    oy = iy + (ih - GRID_H * scale) / 2

    kt, tsec = keytimes()

    # ---- layer 1: intro ----------------------------------------------------
    intro_group = rng.integers(0, N_INTRO_GROUPS, len(dots))
    even = evenness(dots, intro_group, N_INTRO_GROUPS)

    intro = []
    for g in range(N_INTRO_GROUPS):
        d = runs_to_path(dots[intro_group == g])
        begin = round(rng.uniform(0, INTRO_FADE), 3)
        intro.append(
            f'<path d="{d}" opacity="0">'
            f'<animate attributeName="opacity" values="0;1" dur="0.45s" '
            f'begin="{begin}s" fill="freeze"/></path>')

    # ---- layer 2: loop drift bands ----------------------------------------
    band, direction, moved = build_bands(dots.astype(float), logos[0].mean(0), rng)
    centre = dots.mean(0)
    loop = []
    for b in range(N_BANDS):
        sel = dots[band == b]
        if len(sel) == 0:
            continue
        bc = sel.mean(0)
        dx, dy = (logos[0].mean(0) - bc) * DRIFT_FRACTION
        # hold, drift out, stay out for the three logos, drift back
        vals = (f"0 0;0 0;{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};"
                f"{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};"
                f"{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};0 0")
        op = "1;1;0;0;0;0;0;0;1"
        loop.append(
            f'<path d="{runs_to_path(sel)}">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'values="{vals}" keyTimes="{";".join(map(str, kt))}" '
            f'dur="{LOOP}s" begin="{INTRO_HOLD}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="{op}" '
            f'keyTimes="{";".join(map(str, kt))}" dur="{LOOP}s" '
            f'begin="{INTRO_HOLD}s" repeatCount="indefinite"/></path>')

    # ---- layer 3: travellers ----------------------------------------------
    # Opacity 0 while the portrait is up, then each logo in turn.
    trav_kt = ";".join(map(str, kt))
    trav = []
    for i in range(logos.shape[1]):
        p = logos[:, i, :]
        v = (f"{p[0,0]:.1f} {p[0,1]:.1f};{p[0,0]:.1f} {p[0,1]:.1f};"
             f"{p[0,0]:.1f} {p[0,1]:.1f};{p[0,0]:.1f} {p[0,1]:.1f};"
             f"{p[1,0]:.1f} {p[1,1]:.1f};{p[1,0]:.1f} {p[1,1]:.1f};"
             f"{p[2,0]:.1f} {p[2,1]:.1f};{p[2,0]:.1f} {p[2,1]:.1f};"
             f"{p[3,0]:.1f} {p[3,1]:.1f}")
        trav.append(
            f'<path d="M0 0h1v1h-1z" opacity="0">'
            f'<animateTransform attributeName="transform" type="translate" '
            f'values="{v}" keyTimes="{trav_kt}" dur="{LOOP}s" '
            f'begin="{INTRO_HOLD}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0;0;1;1;1;1;1;1;0" '
            f'keyTimes="{trav_kt}" dur="{LOOP}s" begin="{INTRO_HOLD}s" '
            f'repeatCount="indefinite"/></path>')

    # ---- info panel --------------------------------------------------------
    rows, y = [], PANEL_Y + 66
    for label, value in ROWS:
        if label is None:
            y += 11
            continue
        lw = measure(label, FS_ROW)
        vw = measure(value, FS_ROW)
        rows.append(
            f'<text x="{LX}" y="{y}" textLength="{lw:.1f}" '
            f'lengthAdjust="spacingAndGlyphs" fill="{t["muted"]}">{esc(label)}</text>'
            f'<line x1="{LX + lw + 8:.1f}" y1="{y - 4}" x2="{RX - vw - 8:.1f}" '
            f'y2="{y - 4}" stroke="{t["leader"]}" stroke-width="1" '
            f'stroke-dasharray="1 4"/>'
            f'<text x="{RX - vw:.1f}" y="{y}" textLength="{vw:.1f}" '
            f'lengthAdjust="spacingAndGlyphs" fill="{t["text"]}">{esc(value)}</text>')
        y += ROW_STEP

    handle = "@Ujjwalkumar-pm"
    hw = measure(handle, FS_PILL)
    pill_y = PANEL_Y + PANEL_H - 46

    live_w = measure("LIVE", FS_LIVE)
    title = "profile.sh --live"
    tw = measure(title, FS_HEAD)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" \
viewBox="0 0 {W} {H}" role="img" aria-label="Ujjwal Kumar — Product Manager, Bengaluru">
<title>Ujjwal Kumar — Product Manager</title>
<style>
  text {{ font-family: {MONO_STACK}; font-size: {FS_ROW}px; }}
  .hd {{ font-size: {FS_HEAD}px; letter-spacing: .14em; }}
</style>
<rect width="{W}" height="{H}" fill="{t['bg']}"/>
<rect x="{WIN[0]}" y="{WIN[1]}" width="{WIN[2]}" height="{WIN[3]}" rx="12"
      fill="{t['panel']}" stroke="{t['stroke']}"/>
<line x1="{WIN[0]}" y1="{WIN[1]+BAR_H}" x2="{WIN[0]+WIN[2]}" y2="{WIN[1]+BAR_H}"
      stroke="{t['stroke']}"/>
<circle cx="{WIN[0]+26}" cy="{WIN[1]+BAR_H/2}" r="5.5" fill="#FF5F57"/>
<circle cx="{WIN[0]+46}" cy="{WIN[1]+BAR_H/2}" r="5.5" fill="#FEBC2E"/>
<circle cx="{WIN[0]+66}" cy="{WIN[1]+BAR_H/2}" r="5.5" fill="#28C840"/>
<text class="hd" x="{(W-tw)/2:.1f}" y="{WIN[1]+BAR_H/2+4.5}" textLength="{tw:.1f}"
      lengthAdjust="spacingAndGlyphs" fill="{t['title']}">{esc(title)}</text>

<rect x="{PORT[0]}" y="{PORT[1]}" width="{PORT[2]}" height="{PORT[3]}" rx="8"
      fill="none" stroke="{t['stroke']}"/>
<text class="hd" x="{PORT[0]+18}" y="{PORT[1]+26}" fill="{t['chrome']}">VISUAL.MAP</text>
<g transform="translate({ox:.2f} {oy:.2f}) scale({scale:.4f})"
   fill="{t['portrait']}" shape-rendering="crispEdges">
<g>{''.join(intro)}
<animate attributeName="opacity" values="1;0" dur="0.01s" begin="{INTRO_HOLD}s" fill="freeze"/>
</g>
<g opacity="0">{''.join(loop)}
<animate attributeName="opacity" values="0;1" dur="0.01s" begin="{INTRO_HOLD}s" fill="freeze"/>
</g>
<g fill="{t['chrome']}">{''.join(trav)}</g>
</g>

<rect x="{INFO[0]}" y="{INFO[1]}" width="{INFO[2]}" height="{INFO[3]}" rx="8"
      fill="none" stroke="{t['stroke']}"/>
<text class="hd" x="{LX}" y="{PANEL_Y+26}" fill="{t['chrome']}">SYSTEM.INFO</text>
<g>
  <circle cx="{RX-live_w-14:.1f}" cy="{PANEL_Y+21}" r="4" fill="#FF5F57">
    <animate attributeName="opacity" values="1;.25;1" dur="1.8s" repeatCount="indefinite"/>
  </circle>
  <text x="{RX-live_w:.1f}" y="{PANEL_Y+26}" font-size="{FS_LIVE}"
        textLength="{live_w:.1f}" lengthAdjust="spacingAndGlyphs"
        letter-spacing=".16em" fill="#FF5F57">LIVE</text>
</g>
{''.join(rows)}
<rect x="{LX}" y="{pill_y}" width="{hw+34:.1f}" height="28" rx="14" fill="{t['pillbg']}"/>
<text x="{LX+17:.1f}" y="{pill_y+19}" font-size="{FS_PILL}" textLength="{hw:.1f}"
      lengthAdjust="spacingAndGlyphs" fill="{t['pilltx']}">{esc(handle)}</text>
</svg>
'''

    path = f"{theme_name}.svg"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)

    kb = len(svg.encode()) / 1024
    print(f"{path:10s} {kb:7.0f} KB | dots {len(dots):,} | bands {N_BANDS} "
          f"| intro groups {N_INTRO_GROUPS} | travellers {logos.shape[1]}")
    print(f"           intro evenness {even:.3f} "
          f"(~0.05 good, ~0.7 patchy) | noise moved {moved*100:.1f}% of dots "
          f"between bands")
    return path


if __name__ == "__main__":
    for name in ("dark", "light"):
        build(name)
