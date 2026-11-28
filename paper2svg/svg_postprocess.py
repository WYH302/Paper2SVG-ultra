"""Deterministic SVG Post-Processor & XML Repair Utilities (0.05s, ZERO raster patching!).

Solves the 5 real deterministic geometry/syntax bugs observed when LLMs reconstruct complex academic SVG templates:
  1. `sanitize_svg_numeric_typos`: Repairs double-decimal typos like `width="95..0"` that zero out clipPaths in resvg-py.
  2. `fix_collapsed_arrow_paths`: Rebuilds malformed closed wedge arrow paths (`M x y h dx l hx hy v hv z`) into clean
     symmetric 7-point block arrows.
  3. `autofit_text_in_rects`: Measures true Pillow glyph pixel widths (bold/regular, serif/sans, CSS-safe inline style)
     for every `<text>` inside an enclosing `<rect>` AND on the canvas so ZERO text overflows its box or clips edges.
  4. `resolve_sibling_text_collisions`: Detects same-row `<text>` pairs whose rendered bounding boxes collide horizontally
     (e.g. `Observed Free` touching `1` inside a narrow legend) and cleanly shrinks/nudges them apart.
  5. `nudge_text_away_from_images`: Prevents `<text>` labels from grazing or overlapping embedded `<image>` icons.
"""

from __future__ import annotations

import re
from typing import Any

from lxml import etree
from PIL import ImageFont

from .svg_linter import (
    _cumulative_translate,
    _is_bold_weight,
    _is_inside_non_rendered_def,
    _parse_css_rules,
    _parse_float,
    _resolve_style_prop,
    validate_svg_xml,
)

_FONT_MAP = {
    (False, False): (
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ),
    (True, False): (
        r"C:\Windows\Fonts\arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    (False, True): (
        r"C:\Windows\Fonts\georgia.ttf",
        r"C:\Windows\Fonts\times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ),
    (True, True): (
        r"C:\Windows\Fonts\georgiab.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    ),
}


def extract_svg_block(text: str) -> str:
    """Extract clean root `<svg ...>...</svg>` code from model response text (supports nested `<svg>`)."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:svg|xml|html)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    m = re.search(r"(<svg\b[^>]*>.*</svg>)", cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def normalize_svg_root(svg_code: str, width: int, height: int) -> str:
    """Ensure `<svg>` has standard xmlns and exact width/height/viewBox attributes."""
    if not svg_code:
        return svg_code
    if 'xmlns="http://www.w3.org/2000/svg"' not in svg_code and "xmlns='http://www.w3.org/2000/svg'" not in svg_code:
        svg_code = re.sub(r"<svg\b", '<svg xmlns="http://www.w3.org/2000/svg"', svg_code, count=1)
    if "viewBox=" not in svg_code:
        svg_code = re.sub(
            r"<svg\b",
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}"',
            svg_code,
            count=1,
        )
    return svg_code


def fix_svg_xml_with_llm(
    client: Any,
    svg_code: str,
    errors: list[str],
    *,
    width: int,
    height: int,
    max_retries: int = 3,
) -> str:
    """Iteratively repair broken XML syntax using the main LLM until lxml validates cleanly."""
    current = svg_code
    current_errors = errors
    for _ in range(max_retries):
        prompt = f"""The following SVG code has XML syntax errors that break SVG parsers/editors.
Fix ALL XML syntax errors (such as unescaped `&` -> `&amp;`, unescaped `<` inside text -> `&lt;`, unclosed tags, duplicate attributes, or truncated ending) WITHOUT changing the visual layout, text labels, or `<g id="AF..">` / `__B64_IMG_...__` icon placeholders.

XML ERRORS:
{chr(10).join('- ' + e for e in current_errors)}

BROKEN SVG:
{current}

Output ONLY the repaired, valid SVG code starting with `<svg` and ending with `</svg>`. NO markdown fences, NO commentary."""
        repaired_raw = client.generate_text(prompt, max_output_tokens=64000, temperature=0.1, reasoning_effort="low")
        candidate = normalize_svg_root(extract_svg_block(repaired_raw), width, height)
        if not candidate:
            continue
        ok, new_errs = validate_svg_xml(candidate)
        if ok:
            return candidate
        current = candidate
        current_errors = new_errs
    return current


def _measure_true_px(text: str, font_size: float, *, bold: bool, serif: bool) -> float:
    for fp in _FONT_MAP.get((bold, serif), _FONT_MAP[(bold, False)]):
        try:
            font = ImageFont.truetype(fp, max(6, int(round(font_size))))
            bb = font.getbbox(text)
            return float(bb[2] - bb[0])
        except OSError:
            continue
    return len(text) * font_size * (0.62 if bold else 0.56)


def _is_serif_family(ff: str | None) -> bool:
    if not ff:
        return False
    low = ff.lower()
    return any(k in low for k in ("georgia", "times", "serif", "cambria", "garamond")) and "sans-serif" not in low


def sanitize_svg_numeric_typos(svg_code: str) -> str:
    """Fix occasional LLM double-decimal typos like width='95..0'."""
    return re.sub(
        r'(\b(?:x|y|x1|y1|x2|y2|cx|cy|r|rx|ry|width|height|font-size|stroke-width)=["\']-?\d+)\.\.+(\d+)',
        r"\1.\2",
        svg_code,
    )


def fix_collapsed_arrow_paths(svg_code: str) -> tuple[str, int]:
    """Convert malformed closed shaft+head wedge paths (`M x y h dx l hx hy v hv z`) into clean 7-point block arrows."""
    fixed = 0
    num = r"[-+]?\d+(?:\.\d+)?"

    pat_h = re.compile(
        rf'd=["\']M\s*({num})[,\s]*({num})\s*h\s*({num})\s*l\s*({num})[,\s]*({num})\s*v\s*({num})\s*z["\']',
        re.IGNORECASE,
    )

    def _repl_h(m: re.Match[str]) -> str:
        nonlocal fixed
        x0, y0, dx, hx, hy, hv = [float(g) for g in m.groups()]
        if abs(dx) < 16 or abs(hy) < 3 or abs(abs(hv) - 2.0 * abs(hy)) > 4.0:
            return m.group(0)
        fixed += 1
        tip_x = x0 + dx
        head_base_x = tip_x + hx
        head_r = abs(hy)
        shaft_r = max(2.2, min(head_r * 0.42, 6.0))
        new_d = (
            f"M{x0:.1f} {y0 - shaft_r:.1f} "
            f"L{head_base_x:.1f} {y0 - shaft_r:.1f} "
            f"L{head_base_x:.1f} {y0 - head_r:.1f} "
            f"L{tip_x:.1f} {y0:.1f} "
            f"L{head_base_x:.1f} {y0 + head_r:.1f} "
            f"L{head_base_x:.1f} {y0 + shaft_r:.1f} "
            f"L{x0:.1f} {y0 + shaft_r:.1f} Z"
        )
        return f'd="{new_d}"'

    out = pat_h.sub(_repl_h, svg_code)

    pat_v = re.compile(
        rf'd=["\']M\s*({num})[,\s]*({num})\s*v\s*({num})\s*l\s*({num})[,\s]*({num})\s*h\s*({num})\s*z["\']',
        re.IGNORECASE,
    )

    def _repl_v(m: re.Match[str]) -> str:
        nonlocal fixed
        x0, y0, dy, hx, hy, hh = [float(g) for g in m.groups()]
        if abs(dy) < 16 or abs(hx) < 3 or abs(abs(hh) - 2.0 * abs(hx)) > 4.0:
            return m.group(0)
        fixed += 1
        tip_y = y0 + dy
        head_base_y = tip_y + hy
        head_r = abs(hx)
        shaft_r = max(2.2, min(head_r * 0.42, 6.0))
        new_d = (
            f"M{x0 - shaft_r:.1f} {y0:.1f} "
            f"L{x0 - shaft_r:.1f} {head_base_y:.1f} "
            f"L{x0 - head_r:.1f} {head_base_y:.1f} "
            f"L{x0:.1f} {tip_y:.1f} "
            f"L{x0 + head_r:.1f} {head_base_y:.1f} "
            f"L{x0 + shaft_r:.1f} {head_base_y:.1f} "
            f"L{x0 + shaft_r:.1f} {y0:.1f} Z"
        )
        return f'd="{new_d}"'

    out = pat_v.sub(_repl_v, out)
    return out, fixed


def _set_font_size_css_safe(el: etree._Element, new_fs: float) -> None:
    """Set BOTH presentation attribute AND inline CSS `style` so it always overrides `.class { font-size: ... }`."""
    el.set("font-size", f"{new_fs:.1f}")
    cur = el.get("style") or ""
    cur = re.sub(r"font-size\s*:[^;]+;?", "", cur, flags=re.IGNORECASE).strip(" ;")
    el.set("style", f"{cur}; font-size:{new_fs:.1f}px" if cur else f"font-size:{new_fs:.1f}px")


def _text_bbox(
    t_el: etree._Element,
    css_rules: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    """Compute approximate rendered bounding box of a single-line `<text>` node in root SVG coords."""
    txt = "".join(t_el.itertext()).strip()
    if not txt or txt.startswith("<AF>"):
        return None
    tx = _parse_float(t_el.get("x"))
    ty = _parse_float(t_el.get("y"))
    if tx is None or ty is None:
        return None
    dx, dy = _cumulative_translate(t_el)
    ax, ay = tx + dx, ty + dy
    fs = _parse_float(_resolve_style_prop(t_el, "font-size", css_rules), 16.0) or 16.0
    bold = _is_bold_weight(_resolve_style_prop(t_el, "font-weight", css_rules))
    serif = _is_serif_family(_resolve_style_prop(t_el, "font-family", css_rules))
    anchor = (_resolve_style_prop(t_el, "text-anchor", css_rules) or "start").strip().lower()
    w = _measure_true_px(txt, fs, bold=bold, serif=serif)
    if anchor == "middle":
        x0, x1 = ax - w / 2.0, ax + w / 2.0
    elif anchor == "end":
        x0, x1 = ax - w, ax
    else:
        x0, x1 = ax, ax + w
    y0, y1 = ay - fs * 0.85, ay + fs * 0.25
    return {
        "el": t_el,
        "txt": txt,
        "tx": tx,
        "ty": ty,
        "dx": dx,
        "dy": dy,
        "ax": ax,
        "ay": ay,
        "fs": fs,
        "bold": bold,
        "serif": serif,
        "anchor": anchor,
        "w": w,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
    }


def autofit_text_in_rects(svg_code: str) -> tuple[str, int]:
    """Deterministically fit oversized <text> inside enclosing <rect> boxes, resolve sibling text collisions,
    keep text clear of embedded <image> icons, and clamp canvas edges (CSS-safe).
    """
    try:
        parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
        root = etree.fromstring(svg_code.encode("utf-8"), parser=parser)
    except Exception:
        return svg_code, 0

    css_rules = _parse_css_rules(root)
    vb = root.get("viewBox") or ""
    vb_nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", vb)]
    vb_x, vb_y, vb_w, vb_h = vb_nums if len(vb_nums) == 4 else (0.0, 0.0, 1600.0, 800.0)

    rects: list[tuple[float, float, float, float]] = []
    for r in root.xpath(".//*[local-name()='rect']"):
        if _is_inside_non_rendered_def(r):
            continue
        rw = _parse_float(r.get("width"), 0.0) or 0.0
        rh = _parse_float(r.get("height"), 0.0) or 0.0
        if rw <= 10 or rh <= 10 or (rw >= vb_w * 0.45 and rh >= vb_h * 0.45):
            continue
        rx = _parse_float(r.get("x"), 0.0) or 0.0
        ry = _parse_float(r.get("y"), 0.0) or 0.0
        dx, dy = _cumulative_translate(r)
        rects.append((rx + dx, ry + dy, rw, rh))

    images: list[tuple[float, float, float, float]] = []
    for im in root.xpath(".//*[local-name()='image']"):
        if _is_inside_non_rendered_def(im):
            continue
        iw = _parse_float(im.get("width"), 0.0) or 0.0
        ih = _parse_float(im.get("height"), 0.0) or 0.0
        if iw <= 8 or ih <= 8:
            continue
        ix = _parse_float(im.get("x"), 0.0) or 0.0
        iy = _parse_float(im.get("y"), 0.0) or 0.0
        dx, dy = _cumulative_translate(im)
        images.append((ix + dx, iy + dy, iw, ih))

    adjusted = 0

    # Pass 1: Fit every <text> inside its smallest enclosing <rect> + clamp canvas edges
    for t_el in root.xpath(".//*[local-name()='text']"):
        if _is_inside_non_rendered_def(t_el):
            continue
        info = _text_bbox(t_el, css_rules)
        if not info or len(info["txt"]) <= 1:
            continue

        ax, ay = info["ax"], info["ay"]
        dx = info["dx"]
        fs = info["fs"]
        bold, serif, anchor = info["bold"], info["serif"], info["anchor"]
        txt = info["txt"]
        est_w = info["w"]

        enclosing = [rc for rc in rects if rc[0] - 2 <= ax <= rc[0] + rc[2] + 2 and rc[1] - 2 <= ay <= rc[1] + rc[3] + 2]
        if enclosing:
            enclosing.sort(key=lambda rc: rc[2] * rc[3])
            bx, by, bw, bh = enclosing[0]
            pad = max(8.0, min(16.0, bw * 0.10))
            usable_w = max(12.0, bw - pad)
            tx0, tx1 = info["x0"], info["x1"]
            if est_w > usable_w or tx0 < bx + 3 or tx1 > bx + bw - 3:
                if est_w > usable_w:
                    new_fs = max(8.5, round(fs * (usable_w / est_w), 1))
                    _set_font_size_css_safe(t_el, new_fs)
                    fs = new_fs
                    est_w = _measure_true_px(txt, fs, bold=bold, serif=serif)
                    adjusted += 1
                if anchor == "middle":
                    half = est_w / 2.0
                    clamped_ax = max(bx + pad / 2 + half, min(bx + bw - pad / 2 - half, ax))
                    if abs(clamped_ax - ax) > 1.0:
                        t_el.set("x", f"{clamped_ax - dx:.1f}")
                        adjusted += 1
                elif anchor == "start" and (ax + est_w) > (bx + bw - 4):
                    avail = max(12.0, (bx + bw - 5) - ax)
                    if est_w > avail:
                        new_fs = max(8.5, round(fs * (avail / est_w), 1))
                        _set_font_size_css_safe(t_el, new_fs)
                        adjusted += 1

        # Canvas edge clamp (prevents clipped titles)
        info2 = _text_bbox(t_el, css_rules)
        if info2 and info2["anchor"] == "middle":
            if info2["x0"] < vb_x + 10 or info2["x1"] > vb_x + vb_w - 10:
                max_w = vb_w - 28
                if info2["w"] > max_w:
                    _set_font_size_css_safe(t_el, max(9.0, round(info2["fs"] * (max_w / info2["w"]), 1)))
                    adjusted += 1

    # Pass 2: Same-row sibling <text>-vs-<text> horizontal collision resolver (e.g. `Observed Free` touching `1`)
    text_boxes: list[dict[str, Any]] = []
    for t_el in root.xpath(".//*[local-name()='text']"):
        if _is_inside_non_rendered_def(t_el):
            continue
        # Skip multi-line <tspan> blocks with explicit dy>0 (handled vertically, not single horizontal line)
        tspans = t_el.xpath("./*[local-name()='tspan']")
        if len(tspans) >= 2 and any((_parse_float(ts.get("dy"), 0.0) or 0.0) >= 8.0 for ts in tspans):
            continue
        tb = _text_bbox(t_el, css_rules)
        if tb:
            text_boxes.append(tb)

    text_boxes.sort(key=lambda b: (round(b["ay"] / 8.0), b["x0"]))
    for i in range(len(text_boxes)):
        for j in range(i + 1, len(text_boxes)):
            a, b = text_boxes[i], text_boxes[j]
            max_fs = max(a["fs"], b["fs"])
            if abs(a["ay"] - b["ay"]) > max_fs * 0.60:
                continue
            left, right = (a, b) if a["x0"] <= b["x0"] else (b, a)
            # Only resolve genuine touching/overlapping horizontal neighbors
            if right["x0"] >= left["x1"] + 3.0 or right["x0"] <= left["x0"] + 4.0:
                continue
            min_gap = 4.0
            if left["anchor"] == "start" and len(left["txt"]) >= 3:
                avail = max(16.0, (right["x0"] - min_gap) - left["x0"])
                if left["w"] > avail:
                    new_fs = max(9.0, left["fs"] * 0.62, round(left["fs"] * (avail / left["w"]), 1))
                    if new_fs < left["fs"] - 0.2:
                        _set_font_size_css_safe(left["el"], new_fs)
                        left["fs"] = new_fs
                        left["w"] = _measure_true_px(left["txt"], new_fs, bold=left["bold"], serif=left["serif"])
                        left["x1"] = left["x0"] + left["w"]
                        adjusted += 1
            elif left["anchor"] == "middle" and right["anchor"] == "middle":
                overlap = (left["x1"] - right["x0"]) + min_gap
                if overlap > 0 and overlap < min(left["w"], right["w"]) * 0.75:
                    scale = max(0.72, (left["w"] - overlap * 0.55) / max(1.0, left["w"]))
                    new_fs_l = max(9.0, round(left["fs"] * scale, 1))
                    new_fs_r = max(9.0, round(right["fs"] * scale, 1))
                    _set_font_size_css_safe(left["el"], new_fs_l)
                    _set_font_size_css_safe(right["el"], new_fs_r)
                    left["fs"], right["fs"] = new_fs_l, new_fs_r
                    left["w"] = _measure_true_px(left["txt"], new_fs_l, bold=left["bold"], serif=left["serif"])
                    right["w"] = _measure_true_px(right["txt"], new_fs_r, bold=right["bold"], serif=right["serif"])
                    left["x0"], left["x1"] = left["ax"] - left["w"] / 2.0, left["ax"] + left["w"] / 2.0
                    right["x0"], right["x1"] = right["ax"] - right["w"] / 2.0, right["ax"] + right["w"] / 2.0
                    adjusted += 1

    # Pass 3: Keep `<text>` clear of overlapping `<image>` icons (e.g. `Occupancy` grazed by 3D cube on right)
    if images:
        for tb in text_boxes:
            if len(tb["txt"]) <= 1:
                continue
            for ix, iy, iw, ih in images:
                ox = min(tb["x1"], ix + iw) - max(tb["x0"], ix)
                oy = min(tb["y1"], iy + ih) - max(tb["y0"], iy)
                if ox <= 2.0 or oy <= 2.0:
                    continue
                # If text sits predominantly above the image (grazed bottom corner), nudge text up slightly!
                if tb["y0"] < iy and oy <= tb["fs"] * 0.95:
                    shift_up = oy + 3.0
                    if shift_up <= 18.0:
                        tb["el"].set("y", f"{tb['ty'] - shift_up:.1f}")
                        tb["ty"] -= shift_up
                        tb["ay"] -= shift_up
                        tb["y0"] -= shift_up
                        tb["y1"] -= shift_up
                        adjusted += 1
                # If text sits to the left of the image and its right tail grazes the icon, shrink slightly to clear!
                elif tb["x0"] < ix and ox <= tb["w"] * 0.45:
                    avail = max(12.0, (ix - 4.0) - tb["x0"])
                    if tb["w"] > avail:
                        new_fs = max(8.5, round(tb["fs"] * (avail / tb["w"]), 1))
                        if new_fs < tb["fs"] - 0.2:
                            _set_font_size_css_safe(tb["el"], new_fs)
                            tb["fs"] = new_fs
                            tb["w"] = _measure_true_px(tb["txt"], new_fs, bold=tb["bold"], serif=tb["serif"])
                            tb["x1"] = tb["x0"] + tb["w"]
                            adjusted += 1

    return etree.tostring(root, encoding="unicode"), adjusted


def postprocess_svg_deterministic(svg_code: str) -> tuple[str, dict[str, int]]:
    """Run all deterministic 0.05s SVG geometry repairs cleanly."""
    s = sanitize_svg_numeric_typos(svg_code)
    s, n_arr = fix_collapsed_arrow_paths(s)
    s, n_txt = autofit_text_in_rects(s)
    return s, {"arrows_fixed": n_arr, "texts_autofit": n_txt}
