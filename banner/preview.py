import numpy as np
from PIL import Image
W, H, S = 300, 340, 2
for f, bg, fg, name in (("banner/portrait_dots.npy", (10,16,31), (167,139,250), "dark"),
                        ("banner/portrait_dots_light.npy", (247,247,250), (124,58,237), "light")):
    pts = np.load(f)
    a = np.zeros((H*S, W*S, 3), np.uint8); a[:] = bg
    for x, y in pts:
        a[y*S:(y+1)*S, x*S:(x+1)*S] = fg
    Image.fromarray(a).save(f"banner/preview_{name}.png")
# side by side
d = Image.open("banner/preview_dark.png"); l = Image.open("banner/preview_light.png")
s = Image.new("RGB", (d.width + l.width + 12, d.height), (30,30,30))
s.paste(d, (0,0)); s.paste(l, (d.width+12, 0)); s.thumbnail((1000, 600))
s.save("banner/preview_both.png"); print("banner/preview_both.png", s.size)
