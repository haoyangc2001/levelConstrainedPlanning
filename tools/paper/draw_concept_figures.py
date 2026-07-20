#!/usr/bin/env python3
"""Generate concept figures for the paper with Pillow.

The figures are intentionally simple, reproducible bitmap assets.  They avoid
external converters so the paper can carry committed PNGs even on a headless
server without LaTeX or SVG tooling.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("docs/paper/figures")
WIDE = (2200, 1050)
COL = {
    "ink": (34, 40, 49),
    "muted": (91, 103, 112),
    "grid": (225, 229, 232),
    "blue": (54, 112, 184),
    "green": (62, 145, 98),
    "red": (194, 85, 77),
    "gold": (210, 150, 55),
    "teal": (40, 150, 155),
    "purple": (122, 96, 168),
    "paper": (252, 252, 250),
    "panel": (246, 248, 249),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(44, True)
F_H = font(34, True)
F_BODY = font(28)
F_SMALL = font(23)


def canvas(size: tuple[int, int] = WIDE) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, COL["paper"])
    draw = ImageDraw.Draw(image)
    return image, draw


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, *, fill=COL["ink"], fnt=F_BODY, anchor=None) -> None:
    draw.multiline_text(xy, value, fill=fill, font=fnt, spacing=8, anchor=anchor)


def box(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], label: str, *, fill, outline=None, fnt=F_BODY) -> None:
    draw.rounded_rectangle(rect, radius=18, fill=fill, outline=outline or COL["ink"], width=3)
    x0, y0, x1, y1 = rect
    text(draw, ((x0 + x1) // 2, (y0 + y1) // 2), label, fill=COL["ink"], fnt=fnt, anchor="mm")


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, fill=COL["ink"], width=6) -> None:
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    size = 22
    pts = [
        (ex, ey),
        (ex - size * math.cos(angle - 0.45), ey - size * math.sin(angle - 0.45)),
        (ex - size * math.cos(angle + 0.45), ey - size * math.sin(angle + 0.45)),
    ]
    draw.polygon(pts, fill=fill)


def polyline(draw: ImageDraw.ImageDraw, points: Iterable[tuple[int, int]], *, fill, width=5) -> None:
    draw.line(list(points), fill=fill, width=width, joint="curve")


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def fig1() -> None:
    image, draw = canvas()
    text(draw, (80, 55), "Constraint manifold and verified fallback", fnt=F_TITLE)
    draw.line([(1100, 130), (1100, 980)], fill=COL["grid"], width=5)

    text(draw, (300, 150), "Soft penalty only", fnt=F_H, fill=COL["red"], anchor="mm")
    text(draw, (1650, 150), "Manifold-aware seeds + hard gate", fnt=F_H, fill=COL["green"], anchor="mm")

    # Left manifold and bad descent.
    band = [(180, 560), (310, 470), (470, 430), (650, 465), (840, 575), (980, 700)]
    polyline(draw, band, fill=COL["teal"], width=28)
    polyline(draw, band, fill=(230, 248, 247), width=18)
    text(draw, (240, 670), "level manifold", fnt=F_SMALL, fill=COL["teal"])
    draw.ellipse((280, 280, 330, 330), fill=COL["red"], outline=COL["ink"], width=3)
    text(draw, (350, 280), "generic seed", fnt=F_SMALL, fill=COL["red"])
    arrow(draw, (325, 325), (620, 610), fill=COL["red"])
    draw.ellipse((600, 590, 650, 640), fill=(255, 225, 220), outline=COL["red"], width=5)
    text(draw, (575, 685), "invalid local\nminimum", fnt=F_SMALL, fill=COL["red"], anchor="mm")
    box(draw, (220, 805, 900, 925), "optimizer convergence\nis not task success", fill=(255, 240, 236), outline=COL["red"], fnt=F_BODY)

    # Right side.
    band2 = [(1230, 620), (1370, 500), (1530, 455), (1700, 500), (1860, 620), (2010, 760)]
    polyline(draw, band2, fill=COL["teal"], width=30)
    polyline(draw, band2, fill=(230, 248, 247), width=18)
    for x, y, c in [(1340, 500, COL["blue"]), (1490, 465, COL["green"]), (1640, 485, COL["gold"]), (1800, 610, COL["purple"])]:
        draw.ellipse((x - 20, y - 20, x + 20, y + 20), fill=c, outline=COL["ink"], width=2)
    arrow(draw, (1850, 620), (1950, 620), fill=COL["green"])
    box(draw, (1930, 540, 2110, 700), "hard\ngate", fill=(234, 247, 239), outline=COL["green"], fnt=F_BODY)
    arrow(draw, (2020, 705), (2020, 830), fill=COL["gold"])
    box(draw, (1710, 810, 2110, 930), "fallback to\nverified rule branch", fill=(255, 246, 226), outline=COL["gold"], fnt=F_SMALL)
    text(draw, (1225, 795), "learned candidates are proposals,\nnot executions", fnt=F_BODY, fill=COL["ink"])
    save(image, OUT_DIR / "fig1_teaser.png")


def fig2() -> None:
    image, draw = canvas((2400, 1050))
    text(draw, (80, 55), "Verified planning pipeline and learning loop", fnt=F_TITLE)
    stages = [
        ((120, 260, 470, 430), "request\nstart, goal, world,\nconstraint", COL["panel"]),
        ((610, 260, 960, 430), "manifold-aware\nseed family", (232, 244, 255)),
        ((1100, 260, 1450, 430), "batch optimizer\nrepair/refine", (242, 238, 252)),
        ((1590, 260, 1940, 430), "hard validator\nand selector", (232, 247, 239)),
        ((2080, 260, 2310, 430), "selected\ntrajectory", (255, 246, 226)),
    ]
    for rect, label, fill in stages:
        box(draw, rect, label, fill=fill, outline=COL["ink"], fnt=F_BODY)
    for a, b in [((470, 345), (610, 345)), ((960, 345), (1100, 345)), ((1450, 345), (1590, 345)), ((1940, 345), (2080, 345))]:
        arrow(draw, a, b, fill=COL["blue"])

    # Learning loop.
    loop = [
        ((320, 650, 650, 790), "labeled run\nartifacts", COL["panel"]),
        ((790, 650, 1120, 790), "diffusion seed\nmodel", (235, 242, 255)),
        ((1260, 650, 1590, 790), "success critic\nfrom failures", (252, 236, 239)),
        ((1730, 650, 2060, 790), "candidate order\nand fallback", (255, 246, 226)),
    ]
    for rect, label, fill in loop:
        box(draw, rect, label, fill=fill, outline=COL["ink"], fnt=F_BODY)
    for a, b in [((650, 720), (790, 720)), ((1120, 720), (1260, 720)), ((1590, 720), (1730, 720))]:
        arrow(draw, a, b, fill=COL["teal"])
    arrow(draw, (1895, 650), (780, 435), fill=COL["teal"])
    arrow(draw, (1770, 430), (485, 650), fill=COL["gold"])
    text(draw, (1180, 910), "pipeline = executor + data generator + verifier with fallback", fnt=F_H, fill=COL["muted"], anchor="mm")
    save(image, OUT_DIR / "fig2_system.png")


def fig3() -> None:
    image, draw = canvas()
    text(draw, (80, 55), "Goal-relative seed construction", fnt=F_TITLE)
    # Cartesian path.
    poly = [(210, 620), (410, 520), (620, 470), (850, 500), (1060, 610), (1280, 735)]
    polyline(draw, poly, fill=COL["blue"], width=8)
    for i, (x, y) in enumerate(poly):
        draw.ellipse((x - 16, y - 16, x + 16, y + 16), fill=COL["blue"], outline=COL["ink"], width=2)
        draw.line([(x, y), (x + 65, y - 70)], fill=COL["green"], width=5)
        arrow(draw, (x + 50, y - 55), (x + 65, y - 70), fill=COL["green"], width=5)
        text(draw, (x - 15, y + 35), f"Q{i}", fnt=F_SMALL, fill=COL["muted"])
    text(draw, (270, 780), "Cartesian interpolation with level-preserving tool axes", fnt=F_BODY, fill=COL["ink"])

    # IK branch graph.
    draw.line([(1500, 250), (1500, 850)], fill=COL["grid"], width=4)
    text(draw, (1710, 210), "branch-consistent IK", fnt=F_H, fill=COL["ink"], anchor="mm")
    ys = [330, 470, 610, 750]
    colors = [COL["red"], COL["gold"], COL["teal"], COL["purple"]]
    for idx, y in enumerate(ys):
        polyline(draw, [(1540, y), (1670, y - 30), (1800, y + 25), (1950, y - 15), (2080, y + 10)], fill=colors[idx], width=5)
        text(draw, (1970, y + 34), f"branch {idx+1}", fnt=F_SMALL, fill=colors[idx])
    selected = [(1540, 470), (1670, 440), (1800, 495), (1950, 595), (2080, 620)]
    polyline(draw, selected, fill=COL["ink"], width=11)
    text(draw, (1780, 865), "selected sequence minimizes jumps,\ntrend reversals, goal-anchor distance,\nand limit proximity", fnt=F_BODY, fill=COL["ink"], anchor="mm")
    box(draw, (680, 195, 1240, 335), "Q_i = Q_g T_y(theta_i)", fill=(255, 246, 226), outline=COL["gold"], fnt=F_H)
    save(image, OUT_DIR / "fig3_seed_construction.png")


def main() -> int:
    fig1()
    fig2()
    fig3()
    for path in sorted(OUT_DIR.glob("fig*_*.png")):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
