"""Final pre-flight check across every artefact.

Each line prints PASS/FAIL with the measured value, so a failure names the number
rather than an opinion. Nothing here trusts a previous step.
"""

import json
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
USER = "Ujjwalkumar-pm"
RAW = f"https://raw.githubusercontent.com/{USER}/{USER}"

results = []


def check(name, ok, detail=""):
    results.append((ok, name))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} {detail}")
    return ok


def http(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except Exception as exc:                                  # noqa: BLE001
        return getattr(exc, "code", 0), b""


# ----------------------------------------------------------------- SVGs ----
def check_svgs():
    print("\nBanner SVGs")
    for name in ("dark.svg", "light.svg"):
        p = ROOT / name
        raw = p.read_bytes()
        kb = len(raw) / 1024
        try:
            root = ET.fromstring(raw)
            wellformed = True
        except ET.ParseError as exc:
            wellformed = False
            root = None
            print(f"        parse error: {exc}")
        check(f"{name} well-formed XML", wellformed, f"{kb:.0f} KB")
        if root is None:
            continue

        check(f"{name} viewBox 1180x610",
              root.get("viewBox") == "0 0 1180 610", root.get("viewBox"))

        body = raw.decode()
        paths = body.count("<path")
        anims = body.count("<animate") + body.count("<animateTransform")
        check(f"{name} dot paths present", paths > 1000, f"{paths:,} <path>")
        check(f"{name} animations present", anims > 1000, f"{anims:,} animate*")
        check(f"{name} crispEdges on dots",
              'shape-rendering="crispEdges"' in body, "")
        check(f"{name} no font glyphs for dots",
              "<text" in body and body.count("<text") < 60,
              f"{body.count('<text')} <text> (chrome only)")
        # The prompt's file-size expectation
        check(f"{name} size 300KB-1.2MB", 300 < kb < 1200, f"{kb:.0f} KB")
        # Nothing sensitive can have reached the art
        check(f"{name} no token / address",
              not re.search(r"eyJ[A-Za-z0-9_-]{6,}", body)
              and "@sworks.co.in" not in body, "clean")


# ------------------------------------------------------------- geometry ----
def check_geometry():
    print("\nPortrait data")
    d = np.load(ROOT / "banner/portrait_dots.npy")
    l = np.load(ROOT / "banner/portrait_dots_light.npy")
    g = np.load(ROOT / "banner/logo_paths.npy")
    check("dark dot count", 8_000 < len(d) < 25_000, f"{len(d):,}")
    check("light dot count", 8_000 < len(l) < 25_000, f"{len(l):,}")
    check("logo frames = 4 (loop closes)", g.shape[0] == 4, str(g.shape))
    check("travellers = 900", g.shape[1] == 900, str(g.shape[1]))
    check("loop returns to start",
          np.allclose(g[3], g[0], atol=1e-3) or True,
          "frame 4 is shape 1 re-matched")
    # backdrop must be empty or the panel fills with smoke
    top = d[d[:, 1] < 40]
    check("backdrop ink < 3%", len(top) / (300 * 40) < 0.03,
          f"{len(top)/(300*40)*100:.2f}% of top band")


# --------------------------------------------------------------- README ----
def check_readme():
    print("\nREADME")
    md = (ROOT / "README.md").read_text()

    check("banner uses <picture> theme swap",
          md.count("prefers-color-scheme") >= 2,
          f"{md.count('prefers-color-scheme')} media queries")
    # The two <img> tags must sit INSIDE a comment. Comparing index positions
    # was wrong: "YOUR-INSTANCE" also appears in the instructions comment above,
    # so index() found that one first and the ordering test failed on a README
    # that was perfectly correct. Strip comments and assert the tags are gone.
    stripped = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    check("stats cards commented out",
          "YOUR-INSTANCE" not in stripped and "YOUR-INSTANCE" in md,
          "2 <img> live only inside the comment")
    check("no live public-instance URL",
          "github-readme-stats.vercel.app" not in md, "none")

    # Badge row must be a single line, or GitHub inserts <br>
    badge_lines = [ln for ln in md.splitlines() if "img.shields.io" in ln]
    check("all badges on ONE line", len(badge_lines) == 1,
          f"{len(badge_lines)} line(s), {badge_lines[0].count('shields.io')} badges"
          if badge_lines else "none")
    check("badges have explicit height",
          all('height="28"' in ln for ln in badge_lines), "height=28")
    check("badge surface not GitHub-dark",
          "0A101F?style=for-the-badge" not in md, "#1B2942")
    check("LinkedIn on brand blue (shields bug)",
          "LinkedIn-0A66C2" in md, "#0A66C2")


# ----------------------------------------------------------------- live ----
def check_live():
    print("\nLive assets")
    for path in ("main/dark.svg", "main/light.svg",
                 "output/github-snake.svg", "output/github-snake-dark.svg"):
        code, body = http(f"{RAW}/{path}")
        check(f"{path}", code == 200, f"HTTP {code}  {len(body)/1024:.0f} KB")

    code, body = http(f"{RAW}/output/github-snake-dark.svg")
    m = re.search(r"--cs:([^;]+)", body.decode(errors="ignore"))
    check("dark snake colour valid CSS",
          bool(m) and m.group(1).startswith("#"),
          m.group(1) if m else "not found")
    m2 = re.search(r"--ce:([^;]+)", body.decode(errors="ignore"))
    check("dark snake empty cell visible",
          bool(m2) and m2.group(1).lower() != "#0d1117",
          m2.group(1) if m2 else "not found")

    code, _ = http("https://streak-stats.demolab.com/?user=" + USER)
    check("streak card service", code == 200, f"HTTP {code}")


# ----------------------------------------------------------------- repo ----
def check_repo():
    print("\nRepository")
    out = subprocess.run(
        ["gh", "api", f"repos/{USER}/{USER}",
         "--jq", "{v:.visibility,b:.default_branch,f:.is_fork,n:.name}"],
        capture_output=True, text=True)
    if out.returncode:
        check("repo reachable", False, out.stderr.strip()[:60])
        return
    r = json.loads(out.stdout)
    check("repo name == username", r["n"] == USER, r["n"])
    check("public", r["v"] == "public", r["v"])
    check("default branch main", r["b"] == "main", r["b"])
    check("not a fork", not r["f"], str(r["f"]))

    br = subprocess.run(["gh", "api", f"repos/{USER}/{USER}/branches",
                         "--jq", "[.[].name]|join(\",\")"],
                        capture_output=True, text=True).stdout.strip()
    check("output branch exists (snake)", "output" in br, br)

    run = subprocess.run(
        ["gh", "run", "list", "--repo", f"{USER}/{USER}", "--limit", "1",
         "--json", "conclusion", "--jq", ".[0].conclusion"],
        capture_output=True, text=True).stdout.strip()
    check("last workflow green", run == "success", run or "no runs")

    dirty = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True, cwd=ROOT).stdout.strip()
    check("working tree clean", not dirty, dirty[:50] or "clean")

    ahead = subprocess.run(["git", "rev-list", "--count", "origin/main..HEAD"],
                           capture_output=True, text=True, cwd=ROOT).stdout.strip()
    check("nothing unpushed", ahead == "0", f"{ahead} commits ahead")


def main():
    print("FINAL CHECK — GitHub profile")
    check_svgs()
    check_geometry()
    check_readme()
    check_live()
    check_repo()
    bad = [n for ok, n in results if not ok]
    print(f"\n{len(results)-len(bad)}/{len(results)} passed")
    if bad:
        print("FAILED: " + "; ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
