"""Deterministic SVG Hard-Rule Linter enforcing the Golden 8 Principles + Two-Column Print Readability."""

from __future__ import annotations

import colorsys
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from lxml import etree
from PIL import Image, ImageFont

from .config import BANNED_FONTS, resolve_density_mode

# Non-rendered SVG container tags whose child geometry should be ignored during layout checks
NON_RENDERED_TAGS = {"defs", "clippath", "marker", "pattern", "mask", "symbol"}

# Meta-label patterns that should NEVER leak into rendered diagram text
FORBIDDEN_META_PATTERNS = [
    re.compile(r"\bZONE\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bSECTION\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bHERO\s+VISUAL\s+ASSETS\b", re.IGNORECASE),
    re.compile(r"\bLAYOUT\s+CONFIGURATION\b", re.IGNORECASE),
    re.compile(r"\bEXACT\s+TEXT\s+INVENTORY\b", re.IGNORECASE),
    re.compile(r"\bBEGIN\s+PROMPT\b", re.IGNORECASE),
    re.compile(r"\bEND\s+PROMPT\b", re.IGNORECASE),
    re.compile(r"\bBASELINE-GREY\b", re.IGNORECASE),
    re.compile(r"\bNOVEL-ACCENT\b", re.IGNORECASE),
    re.compile(r"<AF>\d+", re.IGNORECASE),
]

SERIF_HINTS = ("times", "georgia", "garamond", "palatino", "cambria", "serif")

# Standard 7-inch two-column figure width in typographic points (7 * 72 = 504pt)
TWO_COLUMN_WIDTH_PT = 504.0
MIN_PRINT_FONT_PT = 6.2
MIN_PRINT_FONT_PT_DETAILED = 4.05
MAX_RECOMMENDED_TEXT_ELEMENTS = 28
HARD_MAX_TEXT_ELEMENTS = 36
MIN_WHITESPACE_RATIO = 0.30
MIN_WHITESPACE_RATIO_DETAILED = 0.22


def measure_whitespace_ratio(png_path: Path, white_threshold: int = 246) -> float:
    """Measure percentage of open white/near-white breathing space on a rendered PNG."""
    with Image.open(png_path) as img:
        arr = np.asarray(img.convert("RGB"))
    is_white = np.all(arr >= white_threshold, axis=2)
    return float(np.mean(is_white))


@dataclass
class LintReport:
    valid_xml: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def passed_hard_checks(self) -> bool:
        return self.valid_xml and len(self.errors) == 0

    def to_prompt_feedback(self) -> str:
        lines = [
            "=== DETERMINISTIC SVG PROGRAMMATIC AUDIT REPORT ===",
            f"- XML Valid: {self.valid_xml}",
            f"- Hard Errors ({len(self.errors)}): {'NONE' if not self.errors else ''}",
        ]
        for idx, e in enumerate(self.errors, 1):
            lines.append(f"  [ERROR {idx}] {e}")
        lines.append(f"- Warnings ({len(self.warnings)}): {'NONE' if not self.warnings else ''}")
        for idx, w in enumerate(self.warnings, 1):
            lines.append(f"  [WARN {idx}] {w}")
        if self.stats:
            lines.append(f"- Stats: {self.stats}")
        return "\n".join(lines)


def _find_system_sans_font(bold: bool = False) -> Optional[str]:
    candidates = (
        [
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
            Path(r"C:\Windows\Fonts\calibrib.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ]
        if bold
        else [
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\segoeui.ttf"),
            Path(r"C:\Windows\Fonts\calibri.ttf"),
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Helvetica.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        ]
    )
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


_SYSTEM_FONT_PATH = _find_system_sans_font(bold=False)
_SYSTEM_FONT_BOLD_PATH = _find_system_sans_font(bold=True)


def _is_bold_weight(weight_raw: Optional[str]) -> bool:
    if not weight_raw:
        return False
    w = weight_raw.strip().lower()
    if w in ("bold", "bolder"):
        return True
    try:
        return float(w) >= 600
    except ValueError:
        return False


def _measure_text_px(text: str, font_size: float, *, bold: bool = False) -> float:
    """Measure realistic rendered pixel width using Pillow truetype metrics (regular or bold)."""
    if not text:
        return 0.0
    font_path = _SYSTEM_FONT_BOLD_PATH if (bold and _SYSTEM_FONT_BOLD_PATH) else _SYSTEM_FONT_PATH
    if font_path:
        try:
            font = ImageFont.truetype(font_path, max(8, int(round(font_size))))
            bbox = font.getbbox(text)
            measured = float(bbox[2] - bbox[0])
            # If bold requested but only regular font exists on host, apply ~8% bold width adjustment
            if bold and not _SYSTEM_FONT_BOLD_PATH:
                measured *= 1.08
            return measured
        except Exception:
            pass
    # Conservative fallback heuristic (CJK chars ~1.0em, Latin ~0.58em; +8% for bold)
    width = 0.0
    for ch in text:
        width += font_size * (1.02 if "\u4e00" <= ch <= "\u9fff" else 0.58)
    return width * (1.08 if bold else 1.0)


def _parse_float(val: Optional[str], default: Optional[float] = None) -> Optional[float]:
    if not val:
        return default
    m = re.search(r"[-+]?\d*\.?\d+", str(val))
    return float(m.group(0)) if m else default


def _parse_css_rules(svg_root: etree._Element) -> dict[str, dict[str, str]]:
    """Extract `.class` / grouped `.a, .b` declarations from `<style>` blocks."""
    rules: dict[str, dict[str, str]] = {}
    for style_el in svg_root.xpath(".//*[local-name()='style']"):
        css = style_el.text or ""
        for match in re.finditer(r"([^{}]+)\{([^}]*)\}", css):
            raw_selectors = match.group(1).strip()
            body = match.group(2)
            props: dict[str, str] = {}
            for decl in body.split(";"):
                if ":" in decl:
                    k, v = decl.split(":", 1)
                    props[k.strip().lower()] = v.strip()
            if not props:
                continue
            for sel in raw_selectors.split(","):
                sel_clean = sel.strip()
                if sel_clean:
                    rules.setdefault(sel_clean, {}).update(props)
    return rules


def _resolve_style_prop(el: etree._Element, prop: str, css_rules: dict[str, dict[str, str]]) -> Optional[str]:
    """Resolve an SVG attribute or inline style or class CSS property (walking up ancestors)."""
    curr: Optional[etree._Element] = el
    while curr is not None:
        # Inline style
        style_attr = curr.get("style") or ""
        for decl in style_attr.split(";"):
            if ":" in decl:
                k, v = decl.split(":", 1)
                if k.strip().lower() == prop:
                    return v.strip()
        # Direct attribute
        direct = curr.get(prop)
        if direct:
            return direct.strip()
        # Class rules
        cls_attr = curr.get("class") or ""
        for cls in cls_attr.split():
            if f".{cls}" in css_rules and prop in css_rules[f".{cls}"]:
                return css_rules[f".{cls}"][prop]
        curr = curr.getparent()
    # Root tag fallback
    if "text" in css_rules and prop in css_rules["text"]:
        return css_rules["text"][prop]
    return None


def _hex_to_hue_bucket(hex_color: str) -> Optional[int]:
    """Map chromatic hex color into 12 hue buckets (returns None for neutral white/grey/black)."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        return None
    try:
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
    except ValueError:
        return None
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    if sat < 0.16 or val < 0.18:
        return None  # Neutral white/grey/black
    return int(round(hue * 12)) % 12


def validate_svg_xml(svg_code: str) -> tuple[bool, list[str]]:
    """Strict XML syntax check using lxml."""
    try:
        etree.fromstring(svg_code.encode("utf-8"))
        return True, []
    except etree.XMLSyntaxError as exc:
        errs = [f"Line {e.line}, Col {e.column}: {e.message}" for e in exc.error_log]
        if not errs:
            errs = [str(exc)]
        return False, errs
    except Exception as exc:
        return False, [str(exc)]


def _is_inside_non_rendered_def(el: etree._Element) -> bool:
    """Return True if an SVG element lives inside <defs>, <clipPath>, <marker>, <pattern>, <mask>, or <symbol>."""
    curr = el.getparent()
    while curr is not None:
        tag = etree.QName(curr).localname.lower() if isinstance(curr.tag, str) else ""
        if tag in NON_RENDERED_TAGS:
            return True
        curr = curr.getparent()
    return False


def lint_svg(
    svg_code: str,
    exact_text_inventory: Optional[list[str]] = None,
    expected_viewbox: Optional[tuple[int, int]] = None,
    rendered_png_path: Optional[Path] = None,
    density: str = "auto",
) -> LintReport:
    """Run comprehensive programmatic checks on an SVG against the Golden 8 Principles
    with adaptive density calibration ('concise' vs 'detailed') and canvas out-of-bounds detection.
    """
    report = LintReport()
    ok, xml_errs = validate_svg_xml(svg_code)
    if not ok:
        report.valid_xml = False
        report.errors.extend([f"XML Syntax Error: {e}" for e in xml_errs])
        return report

    root = etree.fromstring(svg_code.encode("utf-8"))
    css_rules = _parse_css_rules(root)

    inv_list = [s for s in (exact_text_inventory or []) if isinstance(s, str) and s.strip()]
    resolved_density = resolve_density_mode(density, len(inv_list))
    min_print_pt = MIN_PRINT_FONT_PT_DETAILED if resolved_density == "detailed" else MIN_PRINT_FONT_PT
    min_whitespace = MIN_WHITESPACE_RATIO_DETAILED if resolved_density == "detailed" else MIN_WHITESPACE_RATIO

    # 1. ViewBox check
    vb = root.get("viewBox") or ""
    vb_nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", vb)]
    if len(vb_nums) != 4 or vb_nums[2] <= 0 or vb_nums[3] <= 0:
        report.errors.append(f"Invalid or missing viewBox attribute: '{vb}'")
        vb_x, vb_y, vb_w, vb_h = 0.0, 0.0, 1600.0, 680.0
    else:
        vb_x, vb_y, vb_w, vb_h = vb_nums[0], vb_nums[1], vb_nums[2], vb_nums[3]
        if expected_viewbox:
            ew, eh = expected_viewbox
            if abs(vb_w - ew) > 20 or abs(vb_h - eh) > 20:
                report.warnings.append(
                    f"viewBox is {int(vb_w)}x{int(vb_h)}, expected ~{ew}x{eh}."
                )

    pt_scale = TWO_COLUMN_WIDTH_PT / vb_w
    oob_issues: list[str] = []
    oob_tol = 2.0

    # Collect all visible <rect> bounding boxes (excluding <defs>/<clipPath>/<marker>) for overflow & OOB checks
    rects: list[tuple[float, float, float, float, str]] = []
    for rect in root.xpath(".//*[local-name()='rect']"):
        if _is_inside_non_rendered_def(rect):
            continue
        rx = _parse_float(rect.get("x"), 0.0) or 0.0
        ry = _parse_float(rect.get("y"), 0.0) or 0.0
        rw = _parse_float(rect.get("width"), 0.0) or 0.0
        rh = _parse_float(rect.get("height"), 0.0) or 0.0
        if rw <= 0 or rh <= 0:
            continue
        # Ignore full-canvas backdrop rect
        if rw >= vb_w * 0.94 and rh >= vb_h * 0.90:
            continue
        dx, dy = _cumulative_translate(rect)
        ax, ay = rx + dx, ry + dy
        rid = rect.get("id") or ""
        rects.append((ax, ay, rw, rh, rid))

        # Canvas boundary clipping check for visible cards/boxes
        if ax < vb_x - oob_tol or (ax + rw) > (vb_x + vb_w + oob_tol) or ay < vb_y - oob_tol or (ay + rh) > (vb_y + vb_h + oob_tol):
            label_ref = f"rect#{rid}" if rid else f"rect({ax:.0f},{ay:.0f},{rw:.0f}x{rh:.0f})"
            oob_issues.append(
                f"Canvas clipping: {label_ref} extends outside viewBox [0..{vb_w:.0f}, 0..{vb_h:.0f}] "
                f"(bounds x={ax:.0f}..{ax + rw:.0f}, y={ay:.0f}..{ay + rh:.0f}); shift inward or compact layout!"
            )

    # Check visible <image> elements for canvas clipping
    for img_el in root.xpath(".//*[local-name()='image']"):
        if _is_inside_non_rendered_def(img_el):
            continue
        ix = _parse_float(img_el.get("x"), 0.0) or 0.0
        iy = _parse_float(img_el.get("y"), 0.0) or 0.0
        iw = _parse_float(img_el.get("width"), 0.0) or 0.0
        ih = _parse_float(img_el.get("height"), 0.0) or 0.0
        if iw <= 0 or ih <= 0:
            continue
        dx, dy = _cumulative_translate(img_el)
        ax, ay = ix + dx, iy + dy
        if ax < vb_x - oob_tol or (ax + iw) > (vb_x + vb_w + oob_tol) or ay < vb_y - oob_tol or (ay + ih) > (vb_y + vb_h + oob_tol):
            iid = img_el.get("id") or "image"
            oob_issues.append(
                f"Canvas clipping: <image id='{iid}'> extends outside viewBox [0..{vb_w:.0f}, 0..{vb_h:.0f}] "
                f"(x={ax:.0f}..{ax + iw:.0f}, y={ay:.0f}..{ay + ih:.0f}); pull inside safe zone!"
            )

    # 2. Inspect all visible <text> elements
    rendered_texts: list[str] = []
    tiny_texts: list[str] = []
    wordy_texts: list[str] = []
    overflow_issues: list[str] = []
    squashed_texts: list[str] = []
    meta_leaks: list[str] = []
    seen_fonts: set[str] = set()
    box_pad = 8.0 if resolved_density == "detailed" else 14.0

    for text_el in root.xpath(".//*[local-name()='text']"):
        if _is_inside_non_rendered_def(text_el):
            continue
        # Inspect tspan lines separately for overflow, combined for inventory
        tspans = text_el.xpath("./*[local-name()='tspan']")
        line_strings = (
            ["".join(ts.itertext()).strip() for ts in tspans if "".join(ts.itertext()).strip()]
            if tspans
            else ["".join(text_el.itertext()).strip()]
        )
        full_text = " ".join(s for s in line_strings if s).strip()
        if not full_text:
            continue
        rendered_texts.append(full_text)

        # Ban horizontal glyph distortion (`textLength` / `lengthAdjust="spacingAndGlyphs"`)
        if text_el.get("textLength") or (text_el.get("lengthAdjust") or "").strip().lower() == "spacingandglyphs":
            squashed_texts.append(
                f"Text '{full_text}' uses textLength/lengthAdjust horizontal squashing (distorts typography); "
                "remove textLength and wrap cleanly onto 2 <tspan> lines or widen its container!"
            )

        # Check forbidden meta-labels
        for pat in FORBIDDEN_META_PATTERNS:
            if pat.search(full_text):
                meta_leaks.append(full_text)

        # Font family check
        ff = _resolve_style_prop(text_el, "font-family", css_rules) or ""
        if ff:
            seen_fonts.add(ff.lower())
            for banned in BANNED_FONTS:
                if banned in ff.lower():
                    report.errors.append(f"Banned non-academic font detected on '{full_text}': {ff}")

        # Printed font size check (adaptive to density tier; ceil guarantees suggested min_px always passes!)
        fs_raw = _resolve_style_prop(text_el, "font-size", css_rules)
        fs = _parse_float(fs_raw, 16.0) or 16.0
        printed_pt = fs * pt_scale
        min_px = math.ceil(min_print_pt / pt_scale - 1e-6)
        if printed_pt < min_print_pt - 1e-6 and len(full_text) > 2:
            tiny_texts.append(
                f"'{full_text}' uses font-size={fs:.0f}px (~{printed_pt:.1f}pt at 2-col print; "
                f"{resolved_density} floor={min_print_pt:.1f}pt); increase to >= {min_px}px"
            )

        # Sentence/wordiness check (Golden Principle 1: Structure >> Text)
        en_words = re.findall(r"[A-Za-z0-9_+-]+", full_text)
        cjk_chars = sum(1 for ch in full_text if "\u4e00" <= ch <= "\u9fff")
        if len(en_words) > 8 or cjk_chars > 22:
            wordy_texts.append(f"'{full_text}' looks like a sentence ({len(en_words)} words); shorten into crisp phrase")

        # Real-width text bounds & overflow checks (using bold metrics when font-weight >= 600)
        is_bold = _is_bold_weight(_resolve_style_prop(text_el, "font-weight", css_rules))
        tx = _parse_float(text_el.get("x"))
        ty = _parse_float(text_el.get("y"))
        if tx is not None and ty is not None:
            dx, dy = _cumulative_translate(text_el)
            ax, ay = tx + dx, ty + dy
            max_line = max(line_strings, key=lambda s: _measure_text_px(s, fs, bold=is_bold))
            est_w = _measure_text_px(max_line, fs, bold=is_bold)

            # Canvas edge clipping check for text
            anchor = (_resolve_style_prop(text_el, "text-anchor", css_rules) or "start").strip().lower()
            if anchor == "middle":
                tx0, tx1 = ax - est_w / 2.0, ax + est_w / 2.0
            elif anchor == "end":
                tx0, tx1 = ax - est_w, ax
            else:
                tx0, tx1 = ax, ax + est_w

            if tx0 < vb_x - oob_tol or tx1 > (vb_x + vb_w + oob_tol) or ay < vb_y - oob_tol or ay > (vb_y + vb_h + oob_tol):
                oob_issues.append(
                    f"Canvas clipping: Text '{max_line}' (x={tx0:.0f}..{tx1:.0f}, y={ay:.0f}) extends outside "
                    f"viewBox [0..{vb_w:.0f}, 0..{vb_h:.0f}]; shift inward or wrap onto 2 <tspan> lines!"
                )

            enclosing = [
                r for r in rects if r[0] - 4 <= ax <= r[0] + r[2] + 4 and r[1] - 4 <= ay <= r[1] + r[3] + 4
            ]
            if enclosing:
                # Smallest enclosing rect is the module box itself (not an outer stage container)
                enclosing.sort(key=lambda r: r[2] * r[3])
                bx, by, bw, bh, bid = enclosing[0]
                # Only check module-sized boxes (skip large macro zone panels)
                if bw < vb_w * 0.45 and bh < vb_h * 0.45:
                    usable_w = bw - box_pad
                    if est_w > usable_w and (est_w - usable_w) > 8.0:
                        overflow_issues.append(
                            f"Text '{max_line}' (~{est_w:.0f}px wide at {fs:.0f}px) overflows its box "
                            f"(width={bw:.0f}px at x={bx:.0f},y={by:.0f}) by ~{est_w - usable_w:.0f}px — "
                            "widen the box, split across 2 <tspan> lines, or adjust font within allowed tier."
                        )

    if meta_leaks:
        report.errors.append(f"Meta-label leakage rendered as visible text (MUST DELETE): {meta_leaks}")
    if oob_issues:
        report.errors.extend(oob_issues[:10])
    if squashed_texts:
        report.errors.extend(squashed_texts[:8])
    if tiny_texts:
        report.errors.extend(tiny_texts[:8])
    if overflow_issues:
        report.errors.extend(overflow_issues[:10])
    if wordy_texts:
        report.warnings.extend(wordy_texts[:6])

    # 3. Font unity check (no Serif + Sans-Serif mixing)
    has_serif = any(
        any(h in f for h in SERIF_HINTS) and "sans-serif" not in f for f in seen_fonts
    )
    has_sans = any("sans-serif" in f or "inter" in f or "arial" in f or "helvetica" in f for f in seen_fonts)
    if has_serif and has_sans:
        report.errors.append(f"Mixed Serif and Sans-Serif fonts detected: {sorted(seen_fonts)}")

    # 4. Color restraint check (allow up to 6 harmonious chromatic hue families for rich infographics)
    hex_colors = set(re.findall(r"#[0-9a-fA-F]{6}\b", svg_code))
    hue_buckets = {b for c in hex_colors if (b := _hex_to_hue_bucket(c)) is not None}
    if len(hue_buckets) > 6:
        report.warnings.append(
            f"Palette uses {len(hue_buckets)} distinct chromatic hue families ({len(hex_colors)} hex codes); "
            "ensure all hues match the reference palette harmoniously."
        )

    embedded_images = len(
        [im for im in root.xpath(".//*[local-name()='image']") if not _is_inside_non_rendered_def(im)]
    )

    # 5. Inventory coverage check (match across consecutive <text> siblings so 2-line stacked labels count as matched!)
    missing_labels: list[str] = []
    inv_checked = 0
    if inv_list:
        joined_sep = re.sub(r"\s+", " ", " || ".join(rendered_texts)).lower()
        joined_flat = re.sub(r"\s+", " ", " ".join(rendered_texts)).lower()
        for expected in inv_list:
            exp_norm = re.sub(r"\s+", " ", expected).strip().lower()
            if len(exp_norm) < 2:
                continue
            inv_checked += 1
            if exp_norm not in joined_sep and exp_norm not in joined_flat:
                missing_labels.append(expected)
        if missing_labels:
            report.warnings.append(
                f"Inventory labels not matched verbatim in SVG ({len(missing_labels)}/{inv_checked}): {missing_labels[:8]}"
            )

    inventory_coverage_ratio = (
        round((inv_checked - len(missing_labels)) / float(inv_checked), 3) if inv_checked > 0 else 1.0
    )
    total_text_chars = sum(len(re.sub(r"\s+", "", t)) for t in rendered_texts)

    # 6. Anti-crowding & whitespace breathing audit (inventory-aware so we NEVER contradict schema.md!)
    if inv_list:
        rec_max_texts = max(MAX_RECOMMENDED_TEXT_ELEMENTS, int(round(len(inv_list) * 1.15)))
        hard_max_texts = max(HARD_MAX_TEXT_ELEMENTS, int(round(len(inv_list) * 1.35)) + 6)
    elif resolved_density == "detailed":
        rec_max_texts = 65
        hard_max_texts = 85
    else:
        rec_max_texts = MAX_RECOMMENDED_TEXT_ELEMENTS
        hard_max_texts = HARD_MAX_TEXT_ELEMENTS

    if len(rendered_texts) > hard_max_texts:
        report.errors.append(
            f"Overcrowded figure: {len(rendered_texts)} visible text elements (max allowed <= {hard_max_texts} "
            f"for {resolved_density} tier). Remove non-inventory clutter or combine multi-line labels using <tspan>!"
        )
    elif len(rendered_texts) > rec_max_texts:
        report.warnings.append(
            f"Text density slightly high ({len(rendered_texts)} elements; target <= {rec_max_texts} for {resolved_density} tier)."
        )

    whitespace_ratio: Optional[float] = None
    if rendered_png_path is not None and rendered_png_path.is_file():
        whitespace_ratio = round(measure_whitespace_ratio(rendered_png_path), 3)
        if whitespace_ratio < min_whitespace:
            report.errors.append(
                f"Canvas is overcrowded: open white breathing space is only {whitespace_ratio * 100:.1f}% "
                f"(minimum {min_whitespace * 100:.0f}% for {resolved_density} tier). Increase gutters and outer margins!"
            )

    report.stats = {
        "viewBox": f"{int(vb_w)}x{int(vb_h)}",
        "aspect_ratio": round(vb_w / vb_h, 2),
        "density_mode": resolved_density,
        "text_elements": len(rendered_texts),
        "total_text_chars": total_text_chars,
        "rect_boxes": len(rects),
        "embedded_visual_assets": embedded_images,
        "chromatic_hue_families": len(hue_buckets),
        "inventory_total": inv_checked,
        "missing_inventory_count": len(missing_labels),
        "inventory_coverage_ratio": inventory_coverage_ratio,
        "out_of_bounds_count": len(oob_issues),
        **({"whitespace_ratio": whitespace_ratio} if whitespace_ratio is not None else {}),
    }
    return report


def _cumulative_translate(el: etree._Element) -> tuple[float, float]:
    """Sum simple `translate(dx, dy)` transforms from ancestors for coordinate checks."""
    dx, dy = 0.0, 0.0
    curr: Optional[etree._Element] = el
    while curr is not None:
        tf = curr.get("transform") or ""
        for m in re.finditer(r"translate\(\s*([-+]?\d*\.?\d+)(?:[\s,]+([-+]?\d*\.?\d+))?\s*\)", tf):
            dx += float(m.group(1))
            if m.group(2):
                dy += float(m.group(2))
        curr = curr.getparent()
    return dx, dy
