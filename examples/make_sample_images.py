"""Generates the two sample images so the example runs with no assets and no API key.

Synthetic on purpose: the repository should be clonable and runnable in one
command, and nobody's real site photos belong in a public repo.
"""

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent / "images"
SKY, GROUND, PANEL, FRAME, DIRT = (
    (176, 196, 214), (196, 178, 150), (38, 48, 74), (150, 155, 165), (150, 132, 96),
)


def _row(draw: ImageDraw.ImageDraw, skew: int, soiled: bool) -> None:
    for i in range(4):
        x = 60 + i * 220
        top, bottom = 250, 470
        lean = skew if i == 2 else 0  # only the third module is off
        pts = [(x, top + lean), (x + 190, top - 18 + lean), (x + 190, bottom - 18), (x, bottom)]
        draw.polygon(pts, fill=PANEL, outline=FRAME)
        if soiled and i == 2:
            draw.ellipse([x + 40, top + 60, x + 140, top + 130], fill=DIRT)


def build() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for name, skew, soiled in (("reference", 0, False), ("subject", 26, True)):
        img = Image.new("RGB", (960, 640), SKY)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 470, 960, 640], fill=GROUND)
        _row(d, skew, soiled)
        img.save(HERE / f"{name}.jpg", quality=88)
        print(f"wrote {HERE / f'{name}.jpg'}")


if __name__ == "__main__":
    build()
