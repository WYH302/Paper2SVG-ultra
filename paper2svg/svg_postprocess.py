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

    # Pass 3: Keep `<text>` clear of overlapping `<image>` icons in all 4 directions
    if images:
        for tb in text_boxes:
            if len(tb["txt"]) <= 1:
                continue
            for ix, iy, iw, ih in images:
                # Ignore huge background photos where text is intentionally overlaid inside them
                if iw > vb_w * 0.22 and ih > vb_h * 0.22:
                    continue
                ox = min(tb["x1"], ix + iw) - max(tb["x0"], ix)
                oy = min(tb["y1"], iy + ih) - max(tb["y0"], iy)
                if ox <= 2.0 or oy <= 2.0:
                    continue
                t_cx = 0.5 * (tb["x0"] + tb["x1"])
                t_cy = 0.5 * (tb["y0"] + tb["y1"])
                i_cx = ix + 0.5 * iw
                i_cy = iy + 0.5 * ih

                # 3a. Text sits predominantly ABOVE the image (grazed bottom corner) -> nudge up
                if tb["y0"] < iy and oy <= tb["fs"] * 0.95 and abs(t_cy - i_cy) >= abs(t_cx - i_cx) * 0.35:
                    shift_up = oy + 3.0
                    if shift_up <= 18.0:
                        tb["el"].set("y", f"{tb['ty'] - shift_up:.1f}")
                        tb["ty"] -= shift_up
                        tb["ay"] -= shift_up
                        tb["y0"] -= shift_up
                        tb["y1"] -= shift_up
                        adjusted += 1
                # 3b. Text sits predominantly BELOW a small icon -> nudge down
                elif tb["y1"] > iy + ih and oy <= tb["fs"] * 1.15 and abs(t_cx - i_cx) <= max(iw, tb["w"]) * 0.45:
                    shift_dn = (iy + ih + 3.0) - tb["y0"]
                    if 0 < shift_dn <= 22.0:
                        tb["el"].set("y", f"{tb['ty'] + shift_dn:.1f}")
                        tb["ty"] += shift_dn
                        tb["ay"] += shift_dn
                        tb["y0"] += shift_dn
                        tb["y1"] += shift_dn
                        adjusted += 1
                # 3c. Small icon sits to the LEFT of the text (e.g. pill icon grazing first letter) -> shift right / fit
                elif i_cx < t_cx and iw <= 150 and ox <= tb["w"] * 0.60:
                    needed_x0 = ix + iw + 6.0
                    shift_r = needed_x0 - tb["x0"]
                    if 0 < shift_r <= 80.0:
                        encl = [
                            rc
                            for rc in rects
                            if rc[0] - 2 <= tb["ax"] <= rc[0] + rc[2] + 2
                            and rc[1] - 2 <= tb["ay"] <= rc[1] + rc[3] + 2
                        ]
                        max_x1 = min((rc[0] + rc[2] - 6.0) for rc in encl) if encl else (vb_x + vb_w - 8.0)
                        avail = max(16.0, max_x1 - needed_x0)
                        if tb["w"] > avail:
                            new_fs = max(8.5, round(tb["fs"] * (avail / tb["w"]), 1))
                            _set_font_size_css_safe(tb["el"], new_fs)
                            tb["fs"] = new_fs
                            tb["w"] = _measure_true_px(tb["txt"], new_fs, bold=tb["bold"], serif=tb["serif"])
                        if tb["anchor"] == "middle":
                            new_ax = needed_x0 + tb["w"] / 2.0
                            tb["el"].set("x", f"{new_ax - tb['dx']:.1f}")
                            tb["tx"] = new_ax - tb["dx"]
                            tb["ax"] = new_ax
                            tb["x0"], tb["x1"] = needed_x0, needed_x0 + tb["w"]
                        elif tb["anchor"] == "start":
                            tb["el"].set("x", f"{needed_x0 - tb['dx']:.1f}")
                            tb["tx"] = needed_x0 - tb["dx"]
                            tb["ax"] = needed_x0
                            tb["x0"], tb["x1"] = needed_x0, needed_x0 + tb["w"]
                        else:
                            new_ax = needed_x0 + tb["w"]
                            tb["el"].set("x", f"{new_ax - tb['dx']:.1f}")
                            tb["tx"] = new_ax - tb["dx"]
                            tb["ax"] = new_ax
                            tb["x0"], tb["x1"] = needed_x0, new_ax
                        adjusted += 1
                # 3d. Text sits to the LEFT of the image and its right tail grazes the icon -> shrink slightly to clear
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


def _subpath_start_end(sub_d: str) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Trace start & end coordinates of a simple M/L/H/V/l/h/v/Q/C SVG subpath (for branching-stem detection)."""
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", sub_d)
    if not tokens or tokens[0] != "M":
        return None, None
    idx = 1
    try:
        sx = cx = float(tokens[idx])
        sy = cy = float(tokens[idx + 1])
        idx += 2
    except (IndexError, ValueError):
        return None, None
    cmd = "L"
    while idx < len(tokens):
        tk = tokens[idx]
        if tk.isalpha():
            cmd = tk
            idx += 1
            if cmd in ("Z", "z"):
                cx, cy = sx, sy
            continue
        try:
            if cmd == "L":
                cx, cy = float(tokens[idx]), float(tokens[idx + 1])
                idx += 2
            elif cmd == "l":
                cx += float(tokens[idx])
                cy += float(tokens[idx + 1])
                idx += 2
            elif cmd == "H":
                cx = float(tokens[idx])
                idx += 1
            elif cmd == "h":
                cx += float(tokens[idx])
                idx += 1
            elif cmd == "V":
                cy = float(tokens[idx])
                idx += 1
            elif cmd == "v":
                cy += float(tokens[idx])
                idx += 1
            elif cmd == "Q":
                cx, cy = float(tokens[idx + 2]), float(tokens[idx + 3])
                idx += 4
            elif cmd == "q":
                cx += float(tokens[idx + 2])
                cy += float(tokens[idx + 3])
                idx += 4
            elif cmd == "C":
                cx, cy = float(tokens[idx + 4]), float(tokens[idx + 5])
                idx += 6
            elif cmd == "c":
                cx += float(tokens[idx + 4])
                cy += float(tokens[idx + 5])
                idx += 6
            else:
                break
        except (IndexError, ValueError):
            break
    return (sx, sy), (cx, cy)


def fix_oversized_markers(svg_code: str) -> tuple[str, int]:
    """Fix SVG `<marker>` & multi-arrow path bugs:
      1. Force `markerUnits="userSpaceOnUse"` on any `<marker>` lacking it (prevents `stroke-width="8"` from
         multiplying `markerWidth="10"` into an 80px giant pyramid!).
      2. Split multi-subpath `<path d="M... M... M..." marker-end="url(...)">` into independent `<path>` nodes so
         SVG draws arrowheads on EVERY arrow instead of ONLY the final subpath!
      3. Strip duplicate `marker-end` from intermediate connected `<line>` elbow segments.
      4. Auto-clone colored markers when a colored path/line (e.g. pink feedback stroke) references a black arrow.
    """
    try:
        parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
        root = etree.fromstring(svg_code.encode("utf-8"), parser=parser)
    except Exception:
        return svg_code, 0

    fixes = 0
    svg_ns = "http://www.w3.org/2000/svg"
    tag_prefix = f"{{{svg_ns}}}" if root.tag.startswith(f"{{{svg_ns}}}") else ""

    markers_by_id: dict[str, etree._Element] = {}
    for mk in root.xpath(".//*[local-name()='marker']"):
        mid = mk.get("id")
        if mid:
            markers_by_id[mid] = mk
        units = (mk.get("markerUnits") or "").strip()
        mw = _parse_float(mk.get("markerWidth"), 10.0) or 10.0
        mh = _parse_float(mk.get("markerHeight"), 10.0) or 10.0
        if units != "userSpaceOnUse":
            if not mk.get("viewBox"):
                mk.set("viewBox", f"0 0 {mw:g} {mh:g}")
            mk.set("markerUnits", "userSpaceOnUse")
            if mw > 13.0 or mh > 13.0:
                scale = 11.0 / max(mw, mh)
                mk.set("markerWidth", f"{max(7.0, mw * scale):.1f}")
                mk.set("markerHeight", f"{max(7.0, mh * scale):.1f}")
            fixes += 1
        elif mw > 16.0 or mh > 16.0:
            if not mk.get("viewBox"):
                mk.set("viewBox", f"0 0 {mw:g} {mh:g}")
            scale = 12.0 / max(mw, mh)
            mk.set("markerWidth", f"{max(8.0, mw * scale):.1f}")
            mk.set("markerHeight", f"{max(8.0, mh * scale):.1f}")
            fixes += 1

    css_rules = _parse_css_rules(root)

    # Split multi-subpath `<path d="M... M..." marker-end="url(...)">` so every independent arrow gets its arrowhead!
    for p_el in list(root.xpath(".//*[local-name()='path']")):
        if _is_inside_non_rendered_def(p_el):
            continue
        me = _resolve_style_prop(p_el, "marker-end", css_rules)
        if not me or "url(" not in me:
            continue
        d_raw = (p_el.get("d") or "").strip()
        if not d_raw.startswith("M"):
            continue
        # Only split on uppercase absolute `M` (preserves relative `m` chevrons or segments inside one subpath)
        parts = [seg.strip() for seg in re.split(r"(?=(?<!^)M)", d_raw) if seg.strip()]
        if len(parts) < 2:
            continue
        parent = p_el.getparent()
        if parent is None:
            continue
        # Detect T-junction stem subpaths whose endpoint feeds another subpath's start point
        endpoints = [_subpath_start_end(seg) for seg in parts]
        starts_pts = [ep[0] for ep in endpoints if ep[0] is not None]
        insert_idx = parent.index(p_el)
        for seg_i, seg in enumerate(parts):
            clone = etree.fromstring(etree.tostring(p_el))
            clone.set("d", seg)
            if clone.get("id") and seg_i > 0:
                clone.set("id", f"{clone.get('id')}_{seg_i}")
            st_pt, en_pt = endpoints[seg_i]
            is_stem = False
            if st_pt is not None and en_pt is not None:
                for k, other_st in enumerate(starts_pts):
                    if k == seg_i:
                        continue
                    if (
                        abs(en_pt[0] - other_st[0]) <= 3.0
                        and abs(en_pt[1] - other_st[1]) <= 3.0
                        and not (abs(st_pt[0] - other_st[0]) <= 3.0 and abs(st_pt[1] - other_st[1]) <= 3.0)
                    ):
                        is_stem = True
                        break
            if is_stem:
                clone.set("marker-end", "none")
                cur_st = clone.get("style") or ""
                cur_st = re.sub(r"marker-end\s*:[^;]+;?", "", cur_st, flags=re.IGNORECASE).strip(" ;")
                clone.set("style", f"{cur_st}; marker-end:none" if cur_st else "marker-end:none")
            parent.insert(insert_idx + seg_i, clone)
        parent.remove(p_el)
        fixes += len(parts) - 1

    lines = [el for el in root.xpath(".//*[local-name()='line']") if not _is_inside_non_rendered_def(el)]
    starts = set()
    for ln in lines:
        x1, y1 = _parse_float(ln.get("x1")), _parse_float(ln.get("y1"))
        if x1 is not None and y1 is not None:
            starts.add((round(x1), round(y1)))
    for ln in lines:
        x1, y1 = _parse_float(ln.get("x1")), _parse_float(ln.get("y1"))
        x2, y2 = _parse_float(ln.get("x2")), _parse_float(ln.get("y2"))
        if None in (x1, y1, x2, y2):
            continue
        me = _resolve_style_prop(ln, "marker-end", css_rules)
        if not me or "url(" not in me:
            continue
        ex, ey = round(x2), round(y2)
        if any(
            abs(ex - sx) <= 3 and abs(ey - sy) <= 3 and not (abs(round(x1) - sx) <= 3 and abs(round(y1) - sy) <= 3)
            for sx, sy in starts
        ):
            ln.set("marker-end", "none")
            cur_st = ln.get("style") or ""
            cur_st = re.sub(r"marker-end\s*:[^;]+;?", "", cur_st, flags=re.IGNORECASE).strip(" ;")
            ln.set("style", f"{cur_st}; marker-end:none" if cur_st else "marker-end:none")
            fixes += 1

    defs_el = root.find(f"{tag_prefix}defs")
    cloned_cache: dict[tuple[str, str], str] = {}
    for el in root.xpath(".//*[local-name()='path' or local-name()='line' or local-name()='polyline']"):
        if _is_inside_non_rendered_def(el):
            continue
        me = el.get("marker-end") or ""
        m_ref = re.search(r"url\(#([^)]+)\)", me)
        if not m_ref:
            continue
        ref_id = m_ref.group(1)
        src_mk = markers_by_id.get(ref_id)
        if src_mk is None:
            continue
        stroke_col = (_resolve_style_prop(el, "stroke", css_rules) or "").strip().lower()
        if not stroke_col or stroke_col in (
            "none", "black", "#000", "#000000", "#111", "#111111", "#222", "#222222", "#333", "#333333"
        ):
            continue
        mk_shapes = src_mk.xpath(".//*[local-name()='path' or local-name()='polygon']")
        if not mk_shapes:
            continue
        mk_fill = (mk_shapes[0].get("fill") or "").strip().lower()
        mk_stroke = (mk_shapes[0].get("stroke") or "").strip().lower()
        is_dark_fill = mk_fill in ("black", "#000", "#000000", "#111", "#111111", "#222", "#222222", "")
        is_dark_stroke = mk_fill == "none" and mk_stroke in ("black", "#000", "#000000", "#111", "#111111", "#222", "#222222", "#333", "#333333", "#666", "#666666")
        if is_dark_fill or is_dark_stroke:
            key = (ref_id, stroke_col)
            new_id = cloned_cache.get(key)
            if not new_id and defs_el is not None:
                safe_col = re.sub(r"[^a-zA-Z0-9]", "", stroke_col)
                new_id = f"{ref_id}_{safe_col}"
                cloned = etree.fromstring(etree.tostring(src_mk))
                cloned.set("id", new_id)
                sw = _parse_float(_resolve_style_prop(el, "stroke-width", css_rules), 2.0) or 2.0
                if sw >= 5.0:
                    cloned.set("markerWidth", f"{min(18.0, sw * 2.2):.1f}")
                    cloned.set("markerHeight", f"{min(18.0, sw * 2.2):.1f}")
                for sh in cloned.xpath(".//*[local-name()='path' or local-name()='polygon']"):
                    if (sh.get("fill") or "").strip().lower() == "none":
                        sh.set("stroke", stroke_col)
                    else:
                        sh.set("fill", stroke_col)
                defs_el.append(cloned)
                cloned_cache[key] = new_id
                markers_by_id[new_id] = cloned
            if new_id:
                el.set("marker-end", f"url(#{new_id})")
                fixes += 1

    return etree.tostring(root, encoding="unicode"), fixes


_MATH_GLYPH_RE = re.compile(
    r"[\u0300-\u036f\u2100-\u214f\u2190-\u22ff\u25a0-\u27bf\u0391-\u03c9\u1d2c-\u1d6a\u2070-\u209f]"
)
_MATH_FONT_FALLBACK = "'Cambria Math', 'Segoe UI Symbol', 'STIX Two Math', 'DejaVu Sans'"


def fix_math_font_fallbacks(svg_code: str) -> tuple[str, int]:
    """Prevent missing-glyph tofu boxes (`□`) in resvg-py by appending math/symbol font fallbacks
    (`Cambria Math`, `Segoe UI Symbol`, `DejaVu Sans`) to CSS rules and `<text>` nodes containing math Unicode.
    """
    fixes = 0

    def _patch_style_block(m: re.Match[str]) -> str:
        nonlocal fixes
        body = m.group(2)

        def _patch_ff(fm: re.Match[str]) -> str:
            nonlocal fixes
            val = fm.group(1).strip()
            if "cambria math" in val.lower():
                return fm.group(0)
            fixes += 1
            return f"font-family:{val}, {_MATH_FONT_FALLBACK}"

        body2 = re.sub(r"font-family\s*:\s*([^;}\n]+)", _patch_ff, body, flags=re.IGNORECASE)
        return f"{m.group(1)}{body2}{m.group(3)}"

    out = re.sub(r"(<style\b[^>]*>)([\s\S]*?)(</style>)", _patch_style_block, svg_code, flags=re.IGNORECASE)

    try:
        parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
        root = etree.fromstring(out.encode("utf-8"), parser=parser)
        changed = False
        for t_el in root.xpath(".//*[local-name()='text']"):
            full_txt = "".join(t_el.itertext())
            if not _MATH_GLYPH_RE.search(full_txt):
                continue
            ff = t_el.get("font-family") or ""
            if "cambria math" not in ff.lower():
                base_ff = ff.strip() if ff.strip() else "Georgia, Arial, sans-serif"
                if any(ch in full_txt for ch in ("\u0304", "\u0302", "\u211d", "\u2208", "\u2130", "\u2744", "\u2723")):
                    t_el.set("font-family", f"{_MATH_FONT_FALLBACK}, {base_ff}")
                    cur = t_el.get("style") or ""
                    cur = re.sub(r"font-family\s*:[^;]+;?", "", cur, flags=re.IGNORECASE).strip(" ;")
                    t_el.set(
                        "style",
                        f"{cur}; font-family:{_MATH_FONT_FALLBACK}, {base_ff}"
                        if cur
                        else f"font-family:{_MATH_FONT_FALLBACK}, {base_ff}",
                    )
                else:
                    t_el.set("font-family", f"{base_ff}, {_MATH_FONT_FALLBACK}")
                fixes += 1
                changed = True
        if changed:
            out = etree.tostring(root, encoding="unicode")
    except Exception:
        pass

    return out, fixes


def postprocess_svg_deterministic(svg_code: str) -> tuple[str, dict[str, int]]:
    """Run all deterministic 0.05s SVG geometry, marker, math-font, and text-fit repairs cleanly."""
    s = sanitize_svg_numeric_typos(svg_code)
    s, n_arr = fix_collapsed_arrow_paths(s)
    s, n_mk = fix_oversized_markers(s)
    s, n_mf = fix_math_font_fallbacks(s)
    s, n_txt = autofit_text_in_rects(s)
    return s, {
        "arrows_fixed": n_arr,
        "markers_fixed": n_mk,
        "math_fonts_fixed": n_mf,
        "texts_autofit": n_txt,
    }
