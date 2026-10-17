#!/usr/bin/env python3
"""Generate Figure 1: nine elementary motility response rules.

Foraging sign convention (allowed by the manuscript: no sign is prescribed):
  low nutrient or a down-gradient heading  -> search (more noise, more looking)
  high nutrient or an up-gradient heading  -> exploit (quieter, more persistent)
  Case 9 keeps chi > 0: steer onto the gradient.

Colour is Okabe-Ito (Wong, Nature Methods 2011): orange = search, blue = exploit,
bluish green = steering. Nutrient is drawn as food (warm gold), not grey dust.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIRECTORY = PROJECT_ROOT / "manuscript" / "figures"
OUT = FIGURE_DIRECTORY / "figure-1-response-rules.svg"
PDF_OUT = FIGURE_DIRECTORY / "figure-1-response-rules.pdf"

# Canvas. Sized for a preprint / two-column PDF (~7 in wide).
# APS: smallest letters >= 2 mm (~8 pt) at final width. Screen PDF: match ~10 pt body.
W, H = 1320, 1188
LEFT, TOP = 160, 124
GAP = 22
PW, PH = 266, 278
RIGHT = LEFT + 4 * PW + 3 * GAP
ROW1 = TOP
ROW2 = TOP + PH + GAP
ROW3 = TOP + 2 * (PH + GAP)
PH9 = H - ROW3 - 114
PW9 = RIGHT - LEFT

# Type in SVG user units. At width 7.0 in, 1 unit ~ 0.39 pt, so 26 ~ 10 pt.
FS_PANEL = 30
FS_HEAD = 32
FS_SUB = 20
FS_LABEL = 26
FS_ENGLISH = 26
FS_LEGEND = 24
FS_MATH = 28

# Okabe-Ito + warm nutrient (Wong 2011; Nature illustration: saturate the action)
INK = "#222222"
INK_SOFT = "#3F3F3F"
SEARCH = "#D55E00"  # vermillion
SEARCH_SOFT = "#E69F00"  # orange
EXPLOIT = "#0072B2"  # blue
EXPLOIT_SOFT = "#56B4E9"  # sky blue
STEER = "#009E73"  # bluish green
FOOD_PALE = "#F7E7B0"
FOOD_RICH = "#E69F00"
WASH_LOW = "#FFF9EE"
WASH_HIGH = "#F3E09A"
FRAME = "#B39A6C"
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
MATH = "DejaVu Serif, Times New Roman, Times, serif"

MODE_COLOR = {
    "search": SEARCH,
    "exploit": EXPLOIT,
    "steer": STEER,
}
MODE_SOFT = {
    "search": SEARCH_SOFT,
    "exploit": EXPLOIT_SOFT,
    "steer": STEER,
}


def lerp(a: str, b: str, t: float) -> str:
    def hex_rgb(c: str) -> tuple[int, int, int]:
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    ar, ag, ab = hex_rgb(a)
    br, bg, bb = hex_rgb(b)
    r = int(ar + (br - ar) * t)
    g = int(ag + (bg - ag) * t)
    b_ = int(ab + (bb - ab) * t)
    return f"#{r:02x}{g:02x}{b_:02x}"


def gauss2(
    rng: random.Random, mux: float, muy: float, sig: float
) -> tuple[float, float]:
    u = max(rng.random(), 1e-12)
    v = rng.random()
    r = sig * math.sqrt(-2.0 * math.log(u))
    ang = 2.0 * math.pi * v
    return mux + r * math.cos(ang), muy + r * math.sin(ang)


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def heading_svg(deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    return math.cos(rad), math.sin(rad)


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def defs(svg: Svg) -> None:
    svg.add("<defs>")
    styles = {
        "search": (SEARCH, "#FFF1E4", SEARCH_SOFT),
        "exploit": (EXPLOIT, "#E7F3FB", EXPLOIT_SOFT),
        "steer": (STEER, "#E5F7F1", STEER),
    }
    for name, (stroke, fill, ghost) in styles.items():
        svg.add(
            f'<g id="bact-{name}">'
            '<path d="M -21,-12.6 L 21,-12.6 A 12.6,12.6 0 0 1 21,12.6 '
            'L -21,12.6 A 12.6,12.6 0 0 1 -21,-12.6 Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2.7"/>'
            f'<polygon points="21,-7.0 36.5,0 21,7.0" fill="{stroke}"/>'
            "</g>"
        )
        svg.add(
            f'<g id="ghost-{name}">'
            '<path d="M -21,-12.6 L 21,-12.6 A 12.6,12.6 0 0 1 21,12.6 '
            'L -21,12.6 A 12.6,12.6 0 0 1 -21,-12.6 Z" '
            f'fill="none" stroke="{ghost}" stroke-width="2.2"/>'
            f'<polygon points="21,-5.8 33.0,0 21,5.8" fill="none" stroke="{ghost}" stroke-width="1.9"/>'
            "</g>"
        )
        svg.add(
            f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 Z" fill="{stroke}"/>'
            "</marker>"
        )
    svg.add(
        '<g id="ghost-neutral">'
        '<path d="M -21,-12.6 L 21,-12.6 A 12.6,12.6 0 0 1 21,12.6 '
        'L -21,12.6 A 12.6,12.6 0 0 1 -21,-12.6 Z" '
        'fill="none" stroke="#6E6E6E" stroke-width="2.2"/>'
        '<polygon points="21,-5.8 33.0,0 21,5.8" fill="none" stroke="#6E6E6E" stroke-width="1.9"/>'
        "</g>"
    )
    svg.add(
        '<linearGradient id="etaGrad" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{WASH_LOW}"/>'
        f'<stop offset="1" stop-color="{WASH_HIGH}"/>'
        "</linearGradient>"
    )
    svg.add(
        '<linearGradient id="etaBar" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{FOOD_PALE}"/>'
        f'<stop offset="1" stop-color="{FOOD_RICH}"/>'
        "</linearGradient>"
    )
    for i in range(8):
        col, row = i % 4, i // 4
        x = LEFT + col * (PW + GAP)
        y = ROW1 if row == 0 else ROW2
        svg.add(
            f'<clipPath id="clip{i}">'
            f'<rect x="{x}" y="{y}" width="{PW}" height="{PH}" rx="10"/>'
            f"</clipPath>"
        )
    svg.add(
        f'<clipPath id="clip8">'
        f'<rect x="{LEFT}" y="{ROW3}" width="{PW9}" height="{PH9}" rx="10"/>'
        f"</clipPath>"
    )
    svg.add("</defs>")


def bact(
    x: float,
    y: float,
    ang: float,
    mode: str,
    *,
    ghost: bool = False,
    opacity: float | None = None,
    scale: float = 1.0,
) -> str:
    href = f"#ghost-{mode}" if ghost else f"#bact-{mode}"
    tr = f"translate({x:.2f},{y:.2f}) rotate({ang:.1f})"
    if abs(scale - 1.0) > 1e-6:
        tr += f" scale({scale:.3f})"
    op = f' opacity="{opacity:.2f}"' if opacity is not None else ""
    return f'<use href="{href}" transform="{tr}"{op}/>'


def panel_frame(x: float, y: float, w: float, h: float) -> str:
    return (
        f'<rect x="{x + 0.6:.1f}" y="{y + 0.6:.1f}" width="{w - 1.2:.1f}" '
        f'height="{h - 1.2:.1f}" rx="10" fill="none" stroke="{FRAME}" stroke-width="1.6"/>'
    )


def halo_text(
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    fill: str,
    anchor: str = "middle",
    weight: str = "400",
    family: str = FONT,
    italic: bool = False,
    extra: str = "",
    halo: bool = True,
) -> str:
    """Optional thin white halo so type stays readable on gold."""
    style = ' font-style="italic"' if italic else ""
    common = (
        f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{family}" '
        f'font-size="{size:.0f}" font-weight="{weight}"{style}{extra}'
    )
    if not halo:
        return f'<text {common} fill="{fill}">{text}</text>'
    sw = 2.2 if italic else max(3.2, 0.13 * size)
    # Emit the halo as its own stroked copy underneath the glyphs.  SVG
    # ``paint-order`` would express this in one element, but CairoSVG ignores
    # it and paints the stroke last, which hollows out every haloed label.
    return (
        f'<text {common} fill="none" stroke="#ffffff" stroke-width="{sw:.1f}" '
        f'stroke-linejoin="round">{text}</text>'
        f'<text {common} fill="{fill}">{text}</text>'
    )


def letter(x: float, y: float, lab: str) -> str:
    return halo_text(
        x, y, f"({lab})", size=FS_PANEL, fill=INK, anchor="start", weight="700"
    )


def header_math(x: float, y: float, xml: str, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{MATH}" '
        f'font-size="{FS_HEAD}" font-style="italic" fill="{INK}">{xml}</text>'
    )


def small_label(
    x: float,
    y: float,
    text: str,
    fill: str = INK_SOFT,
    anchor: str = "middle",
    weight: str = "400",
) -> str:
    return halo_text(x, y, text, size=FS_LABEL, fill=fill, anchor=anchor, weight=weight)


def eta_bar(x: float, y: float, w: float = 64, *, compact: bool = False) -> str:
    h = 10.0
    tip = 12.0
    bar = (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="url(#etaBar)"/>'
        f'<polygon points="{x + w:.1f},{y - 2.2:.1f} {x + w + tip:.1f},{y + h / 2:.1f} {x + w:.1f},{y + h + 2.2:.1f}" fill="{FOOD_RICH}"/>'
    )
    if compact:
        return bar + halo_text(
            x + w + tip + 8,
            y + 9.0,
            "η",
            size=FS_MATH,
            fill=INK,
            anchor="start",
            family=MATH,
            italic=True,
            halo=False,
        )
    return (
        bar
        + halo_text(
            x + w + tip + 10, y + 9.0, "food", size=FS_LABEL, fill=INK, anchor="start"
        )
        + halo_text(
            x + w + tip + 78,
            y + 9.0,
            "η",
            size=FS_MATH,
            fill=INK_SOFT,
            anchor="start",
            family=MATH,
            italic=True,
            halo=False,
        )
    )


def field_dots(
    x0: float,
    y0: float,
    w: float,
    h: float,
    eta_fn,
    rng: random.Random,
    n: int,
    forbid: list[tuple[float, float, float, float]] | None = None,
) -> str:
    chunks = []
    tries = 0
    kept = 0
    while kept < n and tries < n * 16:
        tries += 1
        x = rng.uniform(8, w - 8)
        y = rng.uniform(8, h - 8)
        if forbid:
            skip = False
            pad = 8.0
            for xa, ya, xb, yb in forbid:
                if (xa - pad) <= x <= (xb + pad) and (ya - pad) <= y <= (yb + pad):
                    skip = True
                    break
            if skip:
                continue
        eta = max(0.0, min(1.0, eta_fn(x, y)))
        if rng.random() > 0.10 + 0.90 * eta:
            continue
        r = 2.4 + 2.0 * eta
        op = 0.38 + 0.50 * eta
        fill = lerp(FOOD_PALE, FOOD_RICH, eta)
        chunks.append(
            f'<circle cx="{x0 + x:.1f}" cy="{y0 + y:.1f}" r="{r:.2f}" '
            f'fill="{fill}" opacity="{op:.2f}"/>'
        )
        kept += 1
    return "\n".join(chunks)


def split_washes(x: float, y: float, w: float, h: float) -> str:
    m = w / 2
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{m:.1f}" height="{h:.1f}" fill="{WASH_LOW}"/>'
        f'<rect x="{x + m:.1f}" y="{y:.1f}" width="{m:.1f}" height="{h:.1f}" fill="{WASH_HIGH}"/>'
        f'<line x1="{x + m:.1f}" y1="{y + 48:.1f}" x2="{x + m:.1f}" y2="{y + h - 20:.1f}" '
        f'stroke="{FRAME}" stroke-width="1.6" stroke-dasharray="5 4"/>'
    )


def grad_wash(x: float, y: float, w: float, h: float) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="10" fill="url(#etaGrad)"/>'
    )


def inside_pad(
    x: float, y: float, pad: tuple[float, float, float, float] | None
) -> bool:
    if pad is None:
        return True
    xmin, ymin, xmax, ymax = pad
    return xmin <= x <= xmax and ymin <= y <= ymax


def cloud(
    cx: float,
    cy: float,
    heading: float,
    sig: float,
    n_ghost: int,
    rng: random.Random,
    mode: str,
    scale: float = 1.0,
    pad: tuple[float, float, float, float] | None = None,
) -> list[str]:
    if n_ghost <= 0:
        return [bact(cx, cy, heading, mode, scale=scale)]
    out: list[str] = []
    pts: list[tuple[float, float]] = []
    min_sep2 = 15.0**2
    for _ in range(max(n_ghost * 40, 80)):
        x, y = gauss2(rng, cx, cy, sig)
        if not inside_pad(x, y, pad):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < min_sep2 for px, py in pts):
            continue
        pts.append((x, y))
        if len(pts) >= n_ghost:
            break
    for x, y in pts:
        out.append(
            bact(x, y, heading, mode, ghost=True, opacity=0.48, scale=scale * 0.94)
        )
    out.append(bact(cx, cy, heading, mode, scale=scale))
    return out


def pull_into_pad(
    cx: float,
    cy: float,
    gx: float,
    gy: float,
    pad: tuple[float, float, float, float] | None,
) -> tuple[float, float] | None:
    """Keep a ghost on the heading ray rather than dropping it at the clip."""
    if inside_pad(gx, gy, pad):
        return gx, gy
    for t in (0.88, 0.74, 0.60, 0.46, 0.32):
        x = cx + t * (gx - cx)
        y = cy + t * (gy - cy)
        if inside_pad(x, y, pad):
            return x, y
    return None


def fan(
    cx: float,
    cy: float,
    heading: float,
    half_span: float,
    n_side: int,
    mode: str,
    scale: float = 1.0,
    pad: tuple[float, float, float, float] | None = None,
) -> list[str]:
    if n_side <= 0 or half_span <= 1.0:
        return [bact(cx, cy, heading, mode, scale=1.0)]
    angs = []
    for i in range(1, n_side + 1):
        d = half_span * i / n_side
        angs.append(heading - d)
        angs.append(heading + d)
    angs.sort(key=lambda a: abs(wrap_deg(a - heading)), reverse=True)
    out = []
    for a in angs:
        d = abs(wrap_deg(a - heading))
        frac = d / (half_span + 1e-6)
        op = 0.22 + 0.28 * (1.0 - frac)
        ux, uy = heading_svg(a)
        offset = 18.0 + 20.0 * frac
        placed = pull_into_pad(cx, cy, cx + offset * ux, cy + offset * uy, pad)
        if placed is None:
            continue
        gx, gy = placed
        out.append(
            bact(
                gx,
                gy,
                a,
                mode,
                ghost=True,
                opacity=op,
                scale=scale * (0.90 + 0.08 * (1.0 - frac)),
            )
        )
    out.append(bact(cx, cy, heading, mode, scale=1.0))
    return out


def run_line(p0: tuple[float, float], p1: tuple[float, float], mode: str) -> str:
    """Stroke only: the terminal cell already carries the heading arrow."""
    col = MODE_COLOR[mode]
    return (
        f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" '
        f'stroke="{col}" stroke-width="3.0" opacity="0.92"/>'
    )


def run_cells(
    x0: float,
    y0: float,
    heading: float,
    length: float,
    n: int,
    mode: str,
) -> tuple[list[str], tuple[float, float], tuple[float, float]]:
    ux, uy = heading_svg(heading)
    x1, y1 = x0 + ux * length, y0 + uy * length
    out = []
    for i in range(n):
        t = i / (n - 1)
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if i == n - 1:
            out.append(bact(x, y, heading, mode))
        else:
            out.append(bact(x, y, heading, mode, ghost=True, opacity=0.58))
    return out, (x0, y0), (x1, y1)


def polyline(pts: list[tuple[float, float]], mode: str, extra: str = "") -> str:
    s = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    col = MODE_COLOR[mode]
    return (
        f'<polyline points="{s}" fill="none" stroke="{col}" stroke-width="3.1" '
        f'stroke-linejoin="round" stroke-linecap="round" opacity="0.9"{extra}/>'
    )


def cream_chip(x: float, y: float, w: float, h: float) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="6" fill="#FFFBF2" opacity="0.97" stroke="{FRAME}" stroke-width="0.7"/>'
    )


def tumble_dot(x: float, y: float, mode: str) -> str:
    """Parkinson-style tumble: a small star at the reorientation, not a path vertex."""
    col = MODE_COLOR[mode]
    pts = []
    for i in range(8):
        r = 8.4 if i % 2 == 0 else 3.2
        a = math.radians(-90 + i * 45)
        pts.append(f"{x + r * math.cos(a):.1f},{y + r * math.sin(a):.1f}")
    return f'<polygon points="{" ".join(pts)}" fill="{col}" opacity="0.92"/>'


def path_with_cells(
    waypoints: list[tuple[float, float, float, bool]],
    mode: str,
) -> list[str]:
    cleaned: list[tuple[float, float, float, bool]] = []
    for x, y, ang, tum in waypoints:
        if cleaned:
            px, py, _pa, pt = cleaned[-1]
            if (x - px) ** 2 + (y - py) ** 2 < 4.0:
                cleaned[-1] = (x, y, ang, pt or tum)
                continue
        cleaned.append((x, y, ang, tum))
    out: list[str] = []
    pts = [(x, y) for x, y, _, _ in cleaned]
    out.append(polyline(pts, mode, extra=""))
    n = len(cleaned)
    for i, (x, y, ang, is_tumble) in enumerate(cleaned):
        # Stars only at interior reorientations — not under the start/end cells.
        if is_tumble and 0 < i < n - 1:
            out.append(tumble_dot(x, y, mode))
        if i == 0:
            out.append(bact(x, y, ang, mode, ghost=True, opacity=0.58))
        elif i == n - 1:
            out.append(bact(x, y, ang, mode))
    return out


def steer_traj(
    x0: float,
    y0: float,
    phi0_deg: float,
    v: float,
    alpha: float,
    x_stop: float,
    y_min: float,
    y_max: float,
    dt: float = 0.18,
) -> list[tuple[float, float, float]]:
    phi = math.radians(phi0_deg)
    x, y = x0, y0
    pts = [(x, y, phi0_deg)]
    for _ in range(2500):
        phi += -alpha * math.sin(phi) * dt
        x += v * math.cos(phi) * dt
        y += v * math.sin(phi) * dt
        pts.append((x, y, math.degrees(phi)))
        if x >= x_stop or y <= y_min or y >= y_max:
            break
    return pts


def panel_origin(col: int, row: int) -> tuple[float, float]:
    x = LEFT + col * (PW + GAP)
    y = ROW1 if row == 0 else ROW2
    return x, y


def add_all(svg: Svg, items: list[str]) -> None:
    for item in items:
        svg.add(item)


def close_panel(svg: Svg, x: float, y: float, w: float, h: float) -> None:
    svg.add("</g>")
    svg.add(panel_frame(x, y, w, h))


def icon_jitter(cx: float, cy: float) -> str:
    dots = [(-13, -4), (2, 9), (11, -6), (-4, 7), (7, 5)]
    bits = [
        f'<circle cx="{cx + dx:.1f}" cy="{cy + dy:.1f}" r="2.8" fill="{INK_SOFT}" opacity="0.55"/>'
        for dx, dy in dots
    ]
    bits.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.0" fill="{INK}"/>')
    return "".join(bits)


def icon_heading(cx: float, cy: float) -> str:
    bits = []
    for a, op in ((-38, 0.40), (-18, 0.55), (0, 1.0), (18, 0.55), (38, 0.40)):
        rad = math.radians(a - 90)
        x2 = cx + 18.0 * math.cos(rad)
        y2 = cy + 18.0 * math.sin(rad)
        w = 2.8 if a == 0 else 2.0
        bits.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{INK}" stroke-width="{w}" opacity="{op}" stroke-linecap="round"/>'
        )
    return "".join(bits)


def icon_speed(cx: float, cy: float) -> str:
    return (
        f'<line x1="{cx - 18:.1f}" y1="{cy:.1f}" x2="{cx + 10:.1f}" y2="{cy:.1f}" '
        f'stroke="{INK}" stroke-width="2.6" stroke-linecap="round"/>'
        f'<polygon points="{cx + 9:.1f},{cy - 6.0:.1f} {cx + 22:.1f},{cy:.1f} '
        f'{cx + 9:.1f},{cy + 6.0:.1f}" fill="{INK}"/>'
    )


def icon_tumble(cx: float, cy: float) -> str:
    star = []
    for i in range(8):
        r = 6.6 if i % 2 == 0 else 2.6
        a = math.radians(-90 + i * 45)
        star.append(f"{cx + r * math.cos(a):.1f},{cy + 2.0 + r * math.sin(a):.1f}")
    return (
        f'<polyline points="{cx - 18:.1f},{cy - 7:.1f} {cx - 4:.1f},{cy - 7:.1f} '
        f'{cx + 2:.1f},{cy + 2.0:.1f}" fill="none" stroke="{INK}" stroke-width="2.3" '
        f'stroke-linejoin="round"/>'
        f'<polygon points="{" ".join(star)}" fill="{INK}"/>'
        f'<polyline points="{cx + 2:.1f},{cy + 2.0:.1f} {cx + 15:.1f},{cy - 6:.1f}" '
        f'fill="none" stroke="{INK}" stroke-width="2.3"/>'
    )


def column_heads(svg: Svg) -> None:
    specs = [
        (
            icon_jitter,
            "jitter",
            f'D<tspan baseline-shift="sub" font-size="{FS_SUB}">T</tspan>',
        ),
        (
            icon_heading,
            "heading",
            f'D<tspan baseline-shift="sub" font-size="{FS_SUB}">R</tspan>',
        ),
        (icon_speed, "speed", "v"),
        (icon_tumble, "tumbles", "λ"),
    ]
    for col, (icon, english, math_xml) in enumerate(specs):
        cx = LEFT + col * (PW + GAP) + PW / 2
        svg.add(icon(cx - 30, 44))
        svg.add(header_math(cx + 12, 52, math_xml, anchor="start"))
        svg.add(
            f'<text x="{cx:.1f}" y="94" text-anchor="middle" font-family="{FONT}" '
            f'font-size="{FS_ENGLISH}" fill="{INK_SOFT}">{english}</text>'
        )


def row_heads(svg: Svg) -> None:
    rows = [
        (ROW1 + PH / 2, "food", "level", "δη"),
        (
            ROW2 + PH / 2,
            "heading",
            "vs food",
            f"ζ<tspan baseline-shift='sub' font-size='{FS_SUB}'>∥</tspan>",
        ),
        (
            ROW3 + PH9 / 2,
            "turn to",
            "food",
            f"ζ<tspan baseline-shift='sub' font-size='{FS_SUB}'>⊥</tspan>",
        ),
    ]
    gx = LEFT * 0.50
    for y, line1, line2, math_xml in rows:
        svg.add(
            f'<text x="{gx:.1f}" y="{y - 34:.1f}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="{FS_ENGLISH}" fill="{INK_SOFT}">{line1}</text>'
        )
        svg.add(
            f'<text x="{gx:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="{FS_ENGLISH}" fill="{INK_SOFT}">{line2}</text>'
        )
        svg.add(header_math(gx, y + 34, math_xml, anchor="middle"))


def same_dt_label(x: float, y: float) -> str:
    """Roman 'same' + italic Δt, on a chip so it does not sit in the gold wash."""
    w, h = 108.0, 28.0
    return (
        cream_chip(x - w / 2, y - 20, w, h)
        + halo_text(
            x - 4,
            y,
            "same",
            size=FS_LABEL,
            fill=INK,
            anchor="end",
            weight="700",
            halo=False,
        )
        + halo_text(
            x,
            y,
            "Δt",
            size=FS_MATH,
            fill=INK,
            anchor="start",
            family=MATH,
            italic=True,
            halo=False,
        )
    )


def stacked_label(
    x: float,
    y: float,
    lines: list[str],
    fill: str,
    *,
    anchor: str = "start",
    weight: str = "700",
    leading: float | None = None,
) -> str:
    dy = FS_LABEL + 4 if leading is None else leading
    return "".join(
        small_label(x, y + i * dy, line, fill=fill, anchor=anchor, weight=weight)
        for i, line in enumerate(lines)
    )


def same_eta_guide(x: float, y1: float, y2: float) -> str:
    """Broken vertical rule with roman 'same' + italic η in the gap."""
    ym = 0.5 * (y1 + y2)
    tw, th = 108.0, 30.0
    return (
        f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{ym - 20:.1f}" '
        f'stroke="{INK_SOFT}" stroke-width="1.6" stroke-dasharray="4 4" opacity="0.55"/>'
        + cream_chip(x - tw / 2, ym - 12, tw, th)
        + halo_text(
            x - 2,
            ym + 10,
            "same",
            size=FS_LABEL,
            fill=INK,
            anchor="end",
            weight="700",
            halo=False,
        )
        + halo_text(
            x + 2,
            ym + 10,
            "η",
            size=FS_MATH,
            fill=INK,
            anchor="start",
            family=MATH,
            italic=True,
            halo=False,
        )
        + (
            f'<line x1="{x:.1f}" y1="{ym + 20:.1f}" x2="{x:.1f}" y2="{y2:.1f}" '
            f'stroke="{INK_SOFT}" stroke-width="1.6" stroke-dasharray="4 4" opacity="0.55"/>'
        )
    )


def legend(svg: Svg) -> None:
    """Legend sits on the same four-column grid as the panels."""
    svg.add(
        f'<line x1="{LEFT:.1f}" y1="{H - 98:.1f}" x2="{RIGHT:.1f}" y2="{H - 98:.1f}" '
        f'stroke="{FRAME}" stroke-width="1.0" opacity="0.45"/>'
    )
    y1 = H - 70
    y2 = H - 28
    xs = [LEFT + col * (PW + GAP) + 18 for col in range(4)]
    items = [
        ("search", "search", 0),
        ("exploit", "exploit", 1),
        ("steer", "steer", 2),
    ]
    for mode, label, col in items:
        svg.add(bact(xs[col], y1 - 6, 0, mode, scale=0.84))
        svg.add(
            small_label(
                xs[col] + 38,
                y1,
                label,
                fill=MODE_COLOR[mode],
                anchor="start",
                weight="700",
            )
        )
    svg.add(eta_bar(xs[3], y1 - 12, w=48))

    svg.add(
        f'<g transform="translate({xs[0]:.1f},{y2 - 6:.1f})">'
        '<use href="#ghost-neutral" transform="scale(0.78)" opacity="0.9"/>'
        "</g>"
    )
    svg.add(small_label(xs[0] + 38, y2, "other cells", fill=INK_SOFT, anchor="start"))
    svg.add(
        f'<line x1="{xs[1]:.1f}" y1="{y2 - 6:.1f}" x2="{xs[1] + 32:.1f}" y2="{y2 - 6:.1f}" '
        f'stroke="{INK}" stroke-width="2.6" stroke-linecap="round"/>'
        f'<polygon points="{xs[1] + 30:.1f},{y2 - 13:.1f} {xs[1] + 44:.1f},{y2 - 6:.1f} '
        f'{xs[1] + 30:.1f},{y2 + 1:.1f}" fill="{INK}"/>'
    )
    svg.add(small_label(xs[1] + 52, y2, "one path", fill=INK_SOFT, anchor="start"))
    svg.add(tumble_dot(xs[2] + 10, y2 - 6, "search"))
    svg.add(small_label(xs[2] + 30, y2, "tumble", fill=INK_SOFT, anchor="start"))


def build() -> None:
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    svg = Svg()
    svg.add(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="7.0in" height="{7.0 * H / W:.3f}in">'
    )
    svg.add("<title>Nine elementary motility response rules</title>")
    svg.add(
        "<desc>Nine isolated linear-response rules with a foraging sign convention. "
        "Orange: search when nutrient is scarce or the cell heads away from food. "
        "Blue: exploit when nutrient is rich or the cell heads toward food. "
        "Green: continuous steering onto the gradient. Gold is nutrient. "
        "Unlinked outlines are other realizations; a line is one trajectory.</desc>"
    )
    svg.add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    defs(svg)
    column_heads(svg)
    row_heads(svg)

    letters = "abcdefgh"

    # ---------- Top row: low η = search, high η = exploit ----------
    for col in range(4):
        x, y = panel_origin(col, 0)
        svg.add(f'<g clip-path="url(#clip{col})">')
        svg.add(split_washes(x, y, PW, PH))
        mid = PW / 2

        def eta_left(_xx: float, _yy: float) -> float:
            return 0.16

        def eta_right(_xx: float, _yy: float) -> float:
            return 0.95

        top_band = [
            (0.0, 0.0, mid, 48.0),
            (mid * 0.50 - 38.0, PH * 0.38, mid * 0.50 + 38.0, PH * 0.62),
        ]
        bot_band = [(0.0, PH - 72.0, mid, PH)] if col == 0 else []
        if col == 2:
            top_band.append((mid - 70.0, 0.0, mid, 50.0))
        svg.add(
            field_dots(
                x,
                y,
                mid,
                PH,
                eta_left,
                random.Random(10 + col),
                n=14,
                forbid=top_band + bot_band,
            )
        )
        right_forbid = list(top_band)
        if col == 0:
            right_forbid += bot_band
        if col == 2:
            right_forbid.append((0.0, 0.0, 70.0, 50.0))
        svg.add(
            field_dots(
                x + mid,
                y,
                mid,
                PH,
                eta_right,
                random.Random(20 + col),
                n=28,
                forbid=right_forbid,
            )
        )

        foot = 72 if col == 0 else 24
        halfL = (x + 20, y + 56, x + mid - 14, y + PH - foot)
        halfR = (x + mid + 16, y + 56, x + PW - 22, y + PH - foot)

        if col == 0:
            rng = random.Random(101)
            add_all(
                svg,
                cloud(
                    x + PW * 0.25,
                    y + PH * 0.50,
                    -90,
                    20.0,
                    12,
                    rng,
                    "search",
                    pad=halfL,
                ),
            )
            add_all(
                svg,
                cloud(
                    x + PW * 0.75, y + PH * 0.50, -90, 2.2, 0, rng, "exploit", pad=halfR
                ),
            )
            svg.add(small_label(x + mid / 2, y + PH - 58, "scarce", fill=INK_SOFT))
            svg.add(
                small_label(
                    x + mid / 2, y + PH - 32, "search", fill=SEARCH, weight="700"
                )
            )
            svg.add(small_label(x + mid + mid / 2, y + PH - 58, "rich", fill=INK_SOFT))
            svg.add(
                small_label(
                    x + mid + mid / 2,
                    y + PH - 32,
                    "exploit",
                    fill=EXPLOIT,
                    weight="700",
                )
            )
        elif col == 1:
            add_all(
                svg,
                fan(x + PW * 0.25, y + PH * 0.58, -90, 70.0, 4, "search", pad=halfL),
            )
            add_all(
                svg,
                fan(x + PW * 0.75, y + PH * 0.58, -90, 6.0, 0, "exploit", pad=halfR),
            )
        elif col == 2:
            # Start + end only: overlapping mid-cells read as jitter, not speed.
            cells, p0, p1 = run_cells(x + PW * 0.25, y + PH - 50, -90, 124, 2, "search")
            svg.add(run_line(p0, p1, "search"))
            add_all(svg, cells)
            cells, p0, p1 = run_cells(x + PW * 0.75, y + PH - 50, -90, 48, 2, "exploit")
            svg.add(run_line(p0, p1, "exploit"))
            add_all(svg, cells)
            svg.add(same_dt_label(x + PW * 0.64, y + 38))
        else:
            search_path = [
                (x + PW * 0.27, y + PH * 0.86, -90, False),
                (x + PW * 0.27, y + PH * 0.68, -90, True),
                (x + PW * 0.13, y + PH * 0.54, -40, True),
                (x + PW * 0.36, y + PH * 0.40, -140, True),
                (x + PW * 0.22, y + PH * 0.26, -80, False),
            ]
            exploit_path = [
                (x + PW * 0.75, y + PH * 0.86, -90, False),
                (x + PW * 0.75, y + PH * 0.26, -90, False),
            ]
            add_all(svg, path_with_cells(search_path, "search"))
            add_all(svg, path_with_cells(exploit_path, "exploit"))

        svg.add(letter(x + 16, y + 38, letters[col]))
        close_panel(svg, x, y, PW, PH)

    # ---------- Middle row: down-gradient = search, up-gradient = exploit ----------
    for col in range(4):
        x, y = panel_origin(col, 1)
        idx = 4 + col
        svg.add(f'<g clip-path="url(#clip{idx})">')
        svg.add(grad_wash(x, y, PW, PH))

        def eta_x(xx: float, _yy: float, w=PW) -> float:
            return xx / w

        forbid = [(0.0, 0.0, PW, 56.0), (0.0, 0.0, 58.0, 56.0)]
        if col == 0:
            forbid += [
                (0.0, PH * 0.16, 148.0, PH * 0.46),
                (0.0, PH * 0.68, 148.0, PH * 0.94),
                (PW * 0.76 - 58.0, PH * 0.48, PW * 0.76 + 58.0, PH * 0.66),
                (PW - 110.0, 0.0, PW, 50.0),
            ]
        if col == 2:
            forbid.append((PW * 0.30, 0.0, PW * 0.78, 56.0))
        svg.add(
            field_dots(
                x, y, PW, PH, eta_x, random.Random(30 + col), n=30, forbid=forbid
            )
        )

        box = (x + 24, y + 52, x + PW - 22, y + PH - 24)
        if col == 0:
            rng = random.Random(505)
            padU = (x + 150, y + 58, x + PW - 22, y + PH * 0.44)
            padD = (x + 156, y + PH * 0.68, x + PW - 22, y + PH - 24)
            add_all(
                svg,
                cloud(
                    x + PW * 0.76, y + PH * 0.32, 0, 2.0, 0, rng, "exploit", pad=padU
                ),
            )
            add_all(
                svg,
                cloud(
                    x + PW * 0.76, y + PH * 0.82, 180, 16.0, 10, rng, "search", pad=padD
                ),
            )
            svg.add(
                stacked_label(x + 16, y + PH * 0.32 - 10, ["toward", "food"], EXPLOIT)
            )
            svg.add(
                stacked_label(x + 16, y + PH * 0.82 - 20, ["away from", "food"], SEARCH)
            )
            svg.add(
                same_eta_guide(x + PW * 0.76, y + PH * 0.32 + 28, y + PH * 0.82 - 28)
            )
            svg.add(eta_bar(x + PW - 92, y + 22, w=32, compact=True))
        elif col == 1:
            add_all(
                svg, fan(x + PW * 0.74, y + PH * 0.32, 0, 6.0, 0, "exploit", pad=box)
            )
            add_all(
                svg, fan(x + PW * 0.74, y + PH * 0.78, 180, 68.0, 4, "search", pad=box)
            )
        elif col == 2:
            x0s = x + PW * 0.38
            cells, p0, p1 = run_cells(x0s, y + PH * 0.32, 0, 108, 2, "exploit")
            svg.add(run_line(p0, p1, "exploit"))
            add_all(svg, cells)
            cells, p0, p1 = run_cells(x0s, y + PH * 0.76, 180, 46, 2, "search")
            svg.add(run_line(p0, p1, "search"))
            add_all(svg, cells)
            svg.add(same_dt_label(x + PW * 0.58, y + 38))
        else:
            x0s = x + PW * 0.58
            up = [
                (x0s, y + PH * 0.32, 0, False),
                (x + PW * 0.84, y + PH * 0.32, 0, False),
            ]
            down = [
                (x0s, y + PH * 0.74, 180, False),
                (x + PW * 0.44, y + PH * 0.74, 180, True),
                (x + PW * 0.32, y + PH * 0.86, 125, True),
                (x + PW * 0.20, y + PH * 0.62, -150, False),
            ]
            add_all(svg, path_with_cells(up, "exploit"))
            add_all(svg, path_with_cells(down, "search"))

        svg.add(letter(x + 16, y + 38, letters[4 + col]))
        close_panel(svg, x, y, PW, PH)

    # ---------- Case 9: χ > 0, search → align onto food ----------
    x, y = LEFT, ROW3
    svg.add('<g clip-path="url(#clip8)">')
    svg.add(grad_wash(x, y, PW9, PH9))

    def eta_wide(xx: float, _yy: float, w=PW9) -> float:
        return xx / w

    svg.add(
        field_dots(
            x,
            y,
            PW9,
            PH9,
            eta_wide,
            random.Random(909),
            n=64,
            forbid=[
                (0.0, 0.0, PW9, 54.0),
                (0.0, PH9 * 0.14, 240.0, PH9 * 0.32),
                (300.0, PH9 * 0.36, 620.0, PH9 * 0.52),
                (0.0, PH9 * 0.72, 240.0, PH9),
                (PW9 - 220.0, 0.0, PW9, 54.0),
                (PW9 - 160.0, PH9 * 0.42, PW9, PH9 * 0.62),
            ],
        )
    )

    v, alpha = 3.15, 0.032
    x0 = x + 96
    x_stop = x + PW9 - 168
    y_min, y_max = y + 36, y + PH9 - 36
    y_top, y_mid, y_bot = y + PH9 * 0.26, y + PH9 * 0.50, y + PH9 * 0.74
    svg.add(letter(x + 16, y + 38, "i"))
    svg.add(eta_bar(x + PW9 - 210, y + 22, w=52))
    svg.add(
        small_label(
            x0 + 10, y_top - 30, "lost", fill=SEARCH, anchor="start", weight="700"
        )
    )
    note_x, note_y = x0 + 130, y_mid - 22
    svg.add(cream_chip(note_x - 10, note_y - 20, 228, 28))
    svg.add(
        small_label(
            note_x,
            note_y,
            "already aligned",
            fill=EXPLOIT,
            anchor="start",
            weight="700",
        )
    )
    svg.add(
        small_label(
            x0 + 10, y_bot + 34, "lost", fill=SEARCH, anchor="start", weight="700"
        )
    )

    trajs = [
        (steer_traj(x0, y_top, 22.0, v, alpha, x_stop, y_min, y_max), True),
        (steer_traj(x0, y_mid, 0.0, v, alpha, x_stop, y_min, y_max), False),
        (steer_traj(x0, y_bot, -22.0, v, alpha, x_stop, y_min, y_max), True),
    ]
    for pts, turning in trajs:
        path_pts = pts[::5]
        if path_pts[-1] != pts[-1]:
            path_pts.append(pts[-1])
        xy = [(px, py) for px, py, _ in path_pts]
        if turning:
            split = next(
                (
                    k
                    for k, (_px, _py, ang) in enumerate(path_pts)
                    if abs(wrap_deg(ang)) < 6
                ),
                len(path_pts) - 1,
            )
            svg.add(polyline(xy[: split + 1], "steer"))
            if split < len(xy) - 1:
                svg.add(polyline(xy[split:], "exploit"))
        else:
            svg.add(polyline(xy, "exploit"))
        n = len(pts)
        if turning:
            turn_end = n - 1
            for k, (_px, _py, ang) in enumerate(pts):
                if abs(wrap_deg(ang)) < 6:
                    turn_end = k
                    break
            early = [round(k * turn_end / 2) for k in range(2)]
            late = [round(turn_end + k * (n - 1 - turn_end) / 2) for k in range(1, 3)]
            idxs = early + late
        else:
            idxs = [0, (n - 1) // 2, n - 1]
        seen: set[int] = set()
        for j, i in enumerate(idxs):
            if i in seen:
                continue
            seen.add(i)
            px, py, ang = pts[i]
            ap = abs(wrap_deg(ang))
            if ap > 18:
                mode = "search"
            elif ap > 5:
                mode = "steer"
            else:
                mode = "exploit"
            t = i / max(n - 1, 1)
            if j == len(idxs) - 1:
                svg.add(bact(px, py, ang, "exploit"))
            else:
                svg.add(
                    bact(
                        px,
                        py,
                        ang,
                        mode,
                        ghost=(t < 0.88),
                        opacity=0.52 if mode == "search" else 0.42 + 0.45 * t,
                        scale=0.96,
                    )
                )
    svg.add(
        small_label(
            x_stop + 48,
            y_mid + 8,
            "aligned",
            fill=EXPLOIT,
            anchor="start",
            weight="700",
        )
    )
    close_panel(svg, x, y, PW9, PH9)

    legend(svg)
    svg.add("</svg>")
    svg.write(OUT)
    print(f"wrote {OUT}")
    try:
        import cairosvg
    except ImportError:
        print(
            "SVG written; install CairoSVG to also create the manuscript PDF: "
            "python -m pip install cairosvg"
        )
    else:
        cairosvg.svg2pdf(url=str(OUT), write_to=str(PDF_OUT))
        print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    build()
