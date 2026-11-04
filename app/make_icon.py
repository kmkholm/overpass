"""Draws the Ayna icon (a location pin with a Doppler-curve wave) into assets/ayna.ico and assets/ayna.png."""
from pathlib import Path

from PIL import Image, ImageDraw

out = Path(__file__).with_name("assets"); out.mkdir(exist_ok=True)
S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
# rounded teal tile
d.rounded_rectangle([8, 8, S - 8, S - 8], radius=56, fill=(31, 111, 139, 255))
# pin
d.ellipse([78, 40, 178, 140], fill=(255, 255, 255, 255))
d.polygon([(84, 118), (172, 118), (128, 212)], fill=(255, 255, 255, 255))
d.ellipse([106, 68, 150, 112], fill=(31, 111, 139, 255))
# Doppler sweep (S-curve) across the bottom
pts = []
for i in range(0, 161):
    x = 48 + i; t = (i - 80) / 22.0
    y = 196 + 14 * (2 / (1 + 2.718281828 ** (-t)) - 1)
    pts.append((x, y))
d.line(pts, fill=(255, 211, 105, 255), width=9)
img.save(out / "ayna.png")
img.save(out / "ayna.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
print("icon written to", out)
