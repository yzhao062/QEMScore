"""Make the README hero from the paper's Figure 1 (v5) by removing its three section pointers.

The figure prints "§3.1", "§3.4" and "§4.2" at the top right of its panels. Those numbers index the
paper, so on GitHub they point at nothing. Each label sits alone on its panel's flat background, so
the edit is a fill: find the non-background pixels in the top-right corner of each panel, take
their bounding box with a margin, and paint it the background color sampled beside the label.
Nothing else in the artwork is touched, and the output is written at the source resolution.

Usage: python strip_section_labels.py <in.png> <out.png>
"""
import sys

from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
W, H = im.size
px = im.load()

# panel x-extents in source pixels, read off the rounded panel cards (2400 px wide source)
PANELS = [(8, 760), (796, 1795), (1830, 2392)]
BAND = (int(0.05 * H), int(0.16 * H))       # the title band, where the pointers live
TOL = 18


def close(a, b):
    return all(abs(x - y) <= TOL for x, y in zip(a, b))


boxes = []
for x0, x1 in PANELS:
    xs = range(int(x0 + 0.72 * (x1 - x0)), x1)  # right 28 percent of the panel
    bg = px[xs[0], BAND[0]]                     # panel background beside the label
    pts = [(x, y) for x in xs for y in range(*BAND) if not close(px[x, y], bg)]
    if not pts:
        print(f"panel {x0}-{x1}: no label found")
        continue
    bx0, bx1 = min(p[0] for p in pts), max(p[0] for p in pts)
    by0, by1 = min(p[1] for p in pts), max(p[1] for p in pts)
    m = 6
    box = (bx0 - m, by0 - m, bx1 + m, by1 + m)
    boxes.append((box, bg))
    print(f"panel {x0}-{x1}: label box {box}, {len(pts)} ink pixels, bg {bg}")

for (bx0, by0, bx1, by1), bg in boxes:
    for x in range(bx0, bx1 + 1):
        for y in range(by0, by1 + 1):
            px[x, y] = bg

im.save(dst, optimize=True)
print("wrote", dst, im.size)
