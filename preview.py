"""Local preview of the profile README, as GitHub renders it.

Two things make this faithful rather than approximate:

  * The HTML comes from GitHub's own /markdown endpoint, so the markup is
    byte-for-byte what github.com produces -- not a local markdown library that
    might handle the raw HTML blocks differently.
  * The styling is GitHub's own primer sheet (github-markdown-css), in both
    themes, with the same page backgrounds GitHub uses (#ffffff / #0d1117).

Image URLs are rewritten to local files. That matters for two reasons: the SVG
animations restart on load instead of being served from a warmed CDN cache, and
GitHub's camo proxy is taken out of the picture, so what you see is the asset
itself rather than a proxied copy.

    .venv/bin/python preview.py          # build + serve on :8420
"""

import http.server
import functools
import re
import shutil
import socketserver
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "preview"
USER = "Ujjwalkumar-pm"
RAW = f"https://raw.githubusercontent.com/{USER}/{USER}"
PORT = 8420

CSS_URL = ("https://cdn.jsdelivr.net/npm/github-markdown-css@5.5.1/"
           "github-markdown.min.css")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def render_markdown(md: str) -> str:
    """GitHub's own renderer, so the markup matches github.com exactly."""
    out = subprocess.run(
        ["gh", "api", "-X", "POST", "/markdown", "-f", "mode=gfm",
         "-f", f"context={USER}/{USER}", "-f", f"text={md}"],
        capture_output=True, text=True, check=True)
    return out.stdout


def build():
    OUT.mkdir(exist_ok=True)

    # Local copies so animations run fresh and camo is out of the loop.
    for name in ("dark.svg", "light.svg"):
        shutil.copy(ROOT / name, OUT / name)
    for name in ("github-snake.svg", "github-snake-dark.svg"):
        (OUT / name).write_bytes(fetch(f"{RAW}/output/{name}"))

    html = render_markdown((ROOT / "README.md").read_text())
    html = html.replace(f"{RAW}/main/", "").replace(f"{RAW}/output/", "")
    # GitHub rewrites <img src> through camo in its own render; strip that so we
    # load the local file rather than a proxied copy.
    html = re.sub(r'src="https://camo\.githubusercontent\.com/[^"]+"',
                  lambda m: m.group(0), html)
    html = re.sub(r'<img([^>]*?)src="https://camo[^"]*"([^>]*?)'
                  r'data-canonical-src="([^"]+)"',
                  r'<img\1src="\3"\2', html)

    css = fetch(CSS_URL).decode()

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Profile preview — {USER}</title>
<style>
{css}
:root {{ color-scheme: light dark; }}
body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
        Helvetica, Arial, sans-serif; }}
.page {{ background:#ffffff; }}
.wrap {{ max-width: 1012px; margin: 0 auto; padding: 32px 16px 80px; }}
.bar {{ position: sticky; top:0; z-index:10; display:flex; gap:12px;
        align-items:center; padding:10px 16px; font:13px/1.4 ui-monospace,
        SFMono-Regular, Menlo, monospace; border-bottom:1px solid #d1d9e0;
        background:#f6f8fa; color:#1f2328; }}
.bar b {{ font-weight:600 }}
.bar button {{ font:inherit; padding:4px 12px; border-radius:6px; cursor:pointer;
        border:1px solid #d1d9e0; background:#fff; color:#1f2328; }}
.note {{ color:#59636e }}
html[data-t="dark"] .page {{ background:#0d1117; }}
html[data-t="dark"] .bar {{ background:#151b23; border-color:#3d444d;
        color:#f0f6fc; }}
html[data-t="dark"] .bar button {{ background:#212830; border-color:#3d444d;
        color:#f0f6fc; }}
html[data-t="dark"] .note {{ color:#9198a1 }}
html[data-t="dark"] .markdown-body {{ color:#f0f6fc; }}
/* GitHub sizes the README column at 1012px on a profile. Keep that, or the
   badge row and the 1180px banner will not wrap the way they really do. */
.markdown-body {{ background: transparent; }}
</style>
<html lang="en">
<div class="bar">
  <b>Local preview</b>
  <button onclick="t('light')">Light</button>
  <button onclick="t('dark')">Dark</button>
  <button onclick="location.reload()">Replay animation</button>
  <span class="note">Rendered by GitHub's own /markdown API · assets served
  locally, so the SVG animation restarts on reload and camo is bypassed.</span>
</div>
<div class="page"><div class="wrap">
<article class="markdown-body">
{html}
</article>
</div></div>
<script>
  const H = document.documentElement;
  function t(v) {{ H.dataset.t = v; localStorage.tt = v;
    for (const s of document.querySelectorAll('source')) {{
      // <picture> picks by media query; force the theme we asked for.
      s.media = s.getAttribute('data-m') || s.media;
    }}
  }}
  for (const s of document.querySelectorAll('source'))
    s.setAttribute('data-m', s.media);
  // ?t=light|dark wins, so a screenshot can pin the theme; otherwise follow
  // the last choice, then the OS.
  const q = new URLSearchParams(location.search).get('t');
  t(q || localStorage.tt || (matchMedia('(prefers-color-scheme: dark)').matches
     ? 'dark' : 'light'));
</script>
"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    kb = sum(f.stat().st_size for f in OUT.iterdir()) / 1024
    print(f"built preview/  ({kb:.0f} KB, {len(list(OUT.iterdir()))} files)")


def serve():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(OUT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        print(f"\n  http://127.0.0.1:{PORT}\n\n  Ctrl-C to stop.")
        httpd.serve_forever()


if __name__ == "__main__":
    build()
    serve()
