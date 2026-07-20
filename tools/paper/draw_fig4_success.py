#!/usr/bin/env python3
"""Render Fig.4 Success@K from A4.3 JSON using Pillow."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IN_JSON = Path("reports/d7_paper_assets/fig4_success_at_k.json")
OUT_PNG = Path("docs/paper/figures/fig4_success_at_k.png")
COLORS = {
    "rule_only": (54, 112, 184),
    "diffusion_only": (194, 85, 77),
    "diffusion_critic": (122, 96, 168),
    "mixed_fallback": (62, 145, 98),
}
LABELS = {
    "rule_only": "rule only",
    "diffusion_only": "diffusion only",
    "diffusion_critic": "diffusion + critic",
    "mixed_fallback": "mixed fallback",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def main() -> int:
    payload = json.loads(IN_JSON.read_text(encoding="utf-8"))
    points = []
    for method, series in payload["series"].items():
        if not series:
            continue
        point = dict(series[0])
        point["method"] = method
        points.append(point)

    width, height = 1600, 1100
    margin_l, margin_r, margin_t, margin_b = 170, 80, 185, 250
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), (252, 252, 250))
    draw = ImageDraw.Draw(image)
    f_title = font(48, True)
    f_axis = font(30)
    f_tick = font(26)
    f_label = font(25)

    draw.text((70, 45), "Success@K under matched compute budget", fill=(34, 40, 49), font=f_title)
    draw.text((70, 112), "K = 6 solve-call budget, Wilson 95% intervals", fill=(91, 103, 112), font=f_axis)

    x0, y0 = margin_l, height - margin_b
    x1, y1 = width - margin_r, margin_t
    draw.line([(x0, y0), (x1, y0)], fill=(34, 40, 49), width=4)
    draw.line([(x0, y0), (x0, y1)], fill=(34, 40, 49), width=4)
    for i in range(0, 7):
        value = i * 0.05
        y = y0 - int(value / 0.30 * plot_h)
        draw.line([(x0 - 10, y), (x1, y)], fill=(225, 229, 232), width=2)
        draw.text((x0 - 82, y - 15), f"{value:.2f}", fill=(91, 103, 112), font=f_tick)
    draw.text((x0 + plot_w // 2 - 115, y0 + 150), "Method (K=6)", fill=(34, 40, 49), font=f_axis)

    n = len(points)
    step = plot_w / max(1, n)
    bar_w = 120
    for idx, point in enumerate(points):
        method = point["method"]
        color = COLORS.get(method, (80, 80, 80))
        cx = int(x0 + step * (idx + 0.5))
        rate = float(point["success_at_k"] or 0.0)
        lo = float(point["wilson_lo"] or 0.0)
        hi = float(point["wilson_hi"] or 0.0)
        y_rate = y0 - int(rate / 0.30 * plot_h)
        y_lo = y0 - int(lo / 0.30 * plot_h)
        y_hi = y0 - int(hi / 0.30 * plot_h)
        draw.rectangle((cx - bar_w // 2, y_rate, cx + bar_w // 2, y0), fill=color)
        draw.line([(cx, y_lo), (cx, y_hi)], fill=(34, 40, 49), width=5)
        draw.line([(cx - 35, y_lo), (cx + 35, y_lo)], fill=(34, 40, 49), width=5)
        draw.line([(cx - 35, y_hi), (cx + 35, y_hi)], fill=(34, 40, 49), width=5)
        draw.text((cx - 55, y_rate - 42), f"{rate:.3f}", fill=(34, 40, 49), font=f_label)
        label = LABELS.get(method, method)
        draw.multiline_text((cx - 95, y0 + 22), label.replace(" + ", "\n+ "), fill=(34, 40, 49), font=f_label, spacing=5)

    draw.text(
        (x0 + 35, height - 70),
        "Pre-registered C4 verdict: no learned method shown superior to rule_only.",
        fill=(194, 85, 77),
        font=f_axis,
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT_PNG, optimize=True)
    print(OUT_PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
