"""Render the SVG at a chosen animation time via headless Chrome.

cairosvg is not usable here: it renders only the first SMIL frame and mishandles
textLength, so it would report every later phase as "identical" and every locked
row as misaligned. Chrome is the engine GitHub viewers actually use, and
`setCurrentTime` seeks SMIL exactly.
"""
import subprocess, sys, pathlib
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def shoot(svg, t, out, w=1180, h=610):
    src = pathlib.Path(svg).read_text()
    src = src.replace('<svg ', '<svg id="s" ', 1)
    html = f"""<!doctype html><meta charset="utf-8">
<style>html,body{{margin:0;background:#222}}</style>{src}
<script>
  const s = document.getElementById('s');
  s.pauseAnimations(); s.setCurrentTime({t});
</script>"""
    tmp = pathlib.Path(f"/tmp/shoot_{pathlib.Path(out).stem}.html")
    tmp.write_text(html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--hide-scrollbars", f"--window-size={w},{h}",
                    "--virtual-time-budget=4000",
                    f"--screenshot={out}", tmp.as_uri()],
                   check=True, capture_output=True)
    print(f"  {out}  t={t}s")

if __name__ == "__main__":
    for t, name in [(1.0, "a_intro"), (5.0, "b_portrait"), (8.2, "c_code"),
                    (11.5, "d_triangle"), (14.6, "e_term")]:
        shoot(sys.argv[1] if len(sys.argv) > 1 else "dark.svg", t,
              f"banner/shot_{name}.png")
