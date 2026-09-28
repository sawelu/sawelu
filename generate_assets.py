from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE = np.array([7, 11, 20], dtype=np.float64)
BLOBS = [
    ((0.12, 0.18), 0.62, 0.95, np.array([124, 58, 237])),
    ((0.88, 0.82), 0.58, 0.85, np.array([6, 182, 212])),
    ((0.52, 0.06), 0.40, 0.55, np.array([236, 72, 153])),
    ((0.42, 0.62), 0.50, 0.70, np.array([79, 70, 229])),
    ((0.72, 0.30), 0.34, 0.45, np.array([14, 165, 233])),
]

FONT_BLACK = "/usr/share/fonts/truetype/lato/Lato-Black.ttf"
FONT_LIGHT = "/usr/share/fonts/truetype/lato/Lato-Light.ttf"
FONT_MEDIUM = "/usr/share/fonts/truetype/lato/Lato-Medium.ttf"


def gradient(w: int, h: int) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, w)[None, :]
    ys = np.linspace(0.0, 1.0, h)[:, None]

    acc = np.zeros((h, w, 3), dtype=np.float64)
    wsum = np.ones((h, w), dtype=np.float64)

    for (cx, cy), sigma, strength, color in BLOBS:
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        weight = strength * np.exp(-d2 / (2.0 * sigma**2))
        acc += weight[..., None] * color
        wsum += weight

    rgb = (acc + BASE) / wsum[..., None]

    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt(((xs - 0.5) * 2) ** 2 + ((ys - 0.5) * 2) ** 2) / 1.414
    rgb = rgb * (1.0 - 0.30 * np.clip(r - 0.35, 0, None) ** 2)[..., None]

    return np.clip(rgb, 0, 255)


def add_grain(rgb: np.ndarray, amount: float = 3.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, amount, size=rgb.shape[:2])[..., None]
    return np.clip(rgb + noise, 0, 255)


def draw_tracked(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill,
    tracking: float,
) -> float:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking
    return x - tracking


def text_tracked_width(draw: ImageDraw.ImageDraw, text: str, font, tracking: float) -> float:
    if not text:
        return 0.0
    return sum(draw.textlength(c, font=font) for c in text) + tracking * (len(text) - 1)


def make_banner(path: str, name: str, tagline: str, label: str) -> None:
    w, h = 1500, 500
    rgb = gradient(w, h)
    img = Image.fromarray(add_grain(rgb).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)

    grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    for x in range(0, w, 50):
        g.line([(x, 0), (x, h)], fill=(255, 255, 255, 8), width=1)
    for y in range(0, h, 50):
        g.line([(0, y), (w, y)], fill=(255, 255, 255, 8), width=1)
    img = Image.alpha_composite(img.convert("RGBA"), grid)

    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([-260, -420, 620, 260], fill=(139, 92, 246, 46))
    gd.ellipse([1060, 300, 1800, 780], fill=(6, 182, 212, 40))
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    img = Image.alpha_composite(img, glow)

    draw = ImageDraw.Draw(img)
    left = 96

    f_label = ImageFont.truetype(FONT_MEDIUM, 22)
    draw_tracked(draw, (left, 150), label.upper(), f_label, (255, 255, 255, 150), 6.0)

    f_name = ImageFont.truetype(FONT_BLACK, 118)
    draw_tracked(draw, (left - 4, 196), name, f_name, (255, 255, 255, 255), -2.0)

    bar_w, bar_h = 132, 5
    bar = Image.new("RGBA", (bar_w, bar_h))
    bd = ImageDraw.Draw(bar)
    for i in range(bar_w):
        t = i / (bar_w - 1)
        c = tuple(
            int(a + (b - a) * t) for a, b in zip((167, 139, 250), (34, 211, 238))
        )
        bd.line([(i, 0), (i, bar_h)], fill=(*c, 235))
    img.alpha_composite(bar, (left, 352))

    f_tag = ImageFont.truetype(FONT_LIGHT, 34)
    draw_tracked(draw, (left, 384), tagline, f_tag, (255, 255, 255, 205), 0.4)

    img.convert("RGB").save(path, quality=95)
    print(f"banner -> {path} ({w}x{h})")


def make_avatar(path: str, letter: str, size: int = 1024) -> None:
    rgb = gradient(size, size)
    img = Image.fromarray(add_grain(rgb, seed=11).astype(np.uint8), "RGB").convert("RGBA")

    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    pad = int(size * 0.145)
    rd.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=int((size - 2 * pad) * 0.30),
        outline=(255, 255, 255, 46),
        width=max(2, size // 220),
    )
    ring = ring.filter(ImageFilter.GaussianBlur(size / 900))
    img = Image.alpha_composite(img, ring)

    mark = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    md = ImageDraw.Draw(mark)
    f = ImageFont.truetype(FONT_BLACK, int(size * 0.50))
    box = md.textbbox((0, 0), letter, font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    pos = ((size - w) / 2 - box[0], (size - h) / 2 - box[1] - size * 0.022)

    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((pos[0], pos[1] + size * 0.012), letter, font=f, fill=(0, 0, 0, 90))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(size / 190)))

    md = ImageDraw.Draw(img)
    md.text(pos, letter, font=f, fill=(255, 255, 255, 246))

    img.convert("RGB").save(path, quality=95)
    print(f"avatar -> {path} ({size}x{size})")


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "sawelu"
    tagline = sys.argv[2] if len(sys.argv) > 2 else "React · TypeScript · Go · Python"
    label = sys.argv[3] if len(sys.argv) > 3 else "developer"
    out = "assets"

    make_banner(f"{out}/banner.png", name, tagline, label)
    make_avatar(f"{out}/avatar.png", name[0].upper())
    make_avatar(f"{out}/avatar-256.png", name[0].upper(), 256)
