"""Deterministic SVG Hard-Rule Linter enforcing the Golden 8 Principles + Two-Column Print Readability."""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from lxml import etree
from PIL import Image, ImageFont

from .config import BANNED_FONTS

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
MAX_RECOMMENDED_TEXT_ELEMENTS = 28
HARD_MAX_TEXT_ELEMENTS = 36
MIN_WHITESPACE_RATIO = 0.30


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


def _find_system_sans_font() -> Optional[str]:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


_SYSTEM_FONT_PATH = _find_system_sans_font()


def _measure_text_px(text: str, font_size: float) -> float:
    """Measure realistic rendered pixel width using Pillow truetype metrics when available."""
    if not text:
        return 0.0
    if _SYSTEM_FONT_PATH:
        try:
            font = ImageFont.truetype(_SYSTEM_FONT_PATH, max(8, int(round(font_size))))
            bbox = font.getbbox(text)
            return float(bbox[2] - bbox[0])
        except Exception:
            pass
    # Conservative fallback heuristic (CJK chars ~1.0em, Latin ~0.58em)
    width = 0.0
    for ch in text:
        width += font_size * (1.02 if "\u4e00" <= ch <= "\u9fff" else 0.58)
    return width


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


def lint_svg(
    svg_code: str,
    exact_text_inventory: Optional[list[str]] = None,
    expected_viewbox: Optional[tuple[int, int]] = None,
    rendered_png_path: Optional[Path] = None,
) -> LintReport:
    """Run comprehensive programmatic checks on an SVG against the Golden 8 Principles."""
    report = LintReport()
    ok, xml_errs = validate_svg_xml(svg_code)
    if not ok:
        report.valid_xml = False
        report.errors.extend([f"XML Syntax Error: {e}" for e in xml_errs])
        return report

    root = etree.fromstring(svg_code.encode("utf-8"))
    css_rules = _parse_css_rules(root)

    # 1. ViewBox check
    vb = root.get("viewBox") or ""
    vb_nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", vb)]
    if len(vb_nums) != 4 or vb_nums[2] <= 0 or vb_nums[3] <= 0:
        report.errors.append(f"Invalid or missing viewBox attribute: '{vb}'")
        vb_w, vb_h = 1600.0, 680.0
    else:
        vb_w, vb_h = vb_nums[2], vb_nums[3]
        if expected_viewbox:
            ew, eh = expected_viewbox
            if abs(vb_w - ew) > 20 or abs(vb_h - eh) > 20:
                report.warnings.append(
                    f"viewBox is {int(vb_w)}x{int(vb_h)}, expected ~{ew}x{eh}."
                )

    pt_scale = TWO_COLUMN_WIDTH_PT / vb_w

    # Collect all <rect> bounding boxes (with simple local group translation) for text overflow checks
    rects: list[tuple[float, float, float, float, str]] = []
    for rect in root.xpath(".//*[local-name()='rect']"):
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
        rects.append((rx + dx, ry + dy, rw, rh, rect.get("id") or ""))

    # 2. Inspect all <text> elements
    rendered_texts: list[str] = []
    tiny_texts: list[str] = []
    wordy_texts: list[str] = []
    overflow_issues: list[str] = []
    meta_leaks: list[str] = []
    seen_fonts: set[str] = set()

    for text_el in root.xpath(".//*[local-name()='text']"):
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

        # Printed font size check
        fs_raw = _resolve_style_prop(text_el, "font-size", css_rules)
        fs = _parse_float(fs_raw, 16.0) or 16.0
        printed_pt = fs * pt_scale
        if printed_pt < MIN_PRINT_FONT_PT and len(full_text) > 2:
            min_px = int(round(MIN_PRINT_FONT_PT / pt_scale))
            tiny_texts.append(
                f"'{full_text}' uses font-size={fs:.0f}px (~{printed_pt:.1f}pt at 2-col print); increase to >= {min_px}px"
            )

        # Sentence/wordiness check (Golden Principle 1: Structure >> Text)
        en_words = re.findall(r"[A-Za-z0-9_+-]+", full_text)
        cjk_chars = sum(1 for ch in full_text if "\u4e00" <= ch <= "\u9fff")
        if len(en_words) > 8 or cjk_chars > 22:
            wordy_texts.append(f"'{full_text}' looks like a sentence ({len(en_words)} words); shorten into crisp phrase")

        # Real-width text overflow check against smallest enclosing <rect>
        tx = _parse_float(text_el.get("x"))
        ty = _parse_float(text_el.get("y"))
        if tx is not None and ty is not None:
            dx, dy = _cumulative_translate(text_el)
            ax, ay = tx + dx, ty + dy
            enclosing = [
                r for r in rects if r[0] - 4 <= ax <= r[0] + r[2] + 4 and r[1] - 4 <= ay <= r[1] + r[3] + 4
            ]
            if enclosing:
                # Smallest enclosing rect is the module box itself (not an outer stage container)
                enclosing.sort(key=lambda r: r[2] * r[3])
                bx, by, bw, bh, bid = enclosing[0]
                # Only check module-sized boxes (skip large macro zone panels)
                if bw < vb_w * 0.45 and bh < vb_h * 0.45:
                    max_line = max(line_strings, key=lambda s: _measure_text_px(s, fs))
                    est_w = _measure_text_px(max_line, fs)
                    usable_w = bw - 14.0
                    if est_w > usable_w and (est_w - usable_w) > 8.0:
                        overflow_issues.append(
                            f"Text '{max_line}' (~{est_w:.0f}px wide at {fs:.0f}px) overflows its box "
                            f"(width={bw:.0f}px at x={bx:.0f},y={by:.0f}) by ~{est_w - usable_w:.0f}px — "
                            "widen the box, split across 2 <tspan> lines, or slightly tighten label."
                        )

    if meta_leaks:
        report.errors.append(f"Meta-label leakage rendered as visible text (MUST DELETE): {meta_leaks}")
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

    embedded_images = len(root.xpath(".//*[local-name()='image']"))

    # 5. Inventory coverage check
    missing_labels: list[str] = []
    if exact_text_inventory:
        joined_norm = re.sub(r"\s+", " ", " || ".join(rendered_texts)).lower()
        for expected in exact_text_inventory:
            exp_norm = re.sub(r"\s+", " ", expected).strip().lower()
            if len(exp_norm) < 2:
                continue
            if exp_norm not in joined_norm:
                missing_labels.append(expected)
        if missing_labels:
            report.warnings.append(
                f"Inventory labels not matched verbatim in SVG ({len(missing_labels)}): {missing_labels[:8]}"
            )

    # 6. Anti-crowding & whitespace breathing audit
    if len(rendered_texts) > HARD_MAX_TEXT_ELEMENTS:
        report.errors.append(
            f"Overcrowded figure: {len(rendered_texts)} visible text elements (recommended <= {MAX_RECOMMENDED_TEXT_ELEMENTS}). "
            "Prune non-essential micro-labels/hyperparameters and expand whitespace!"
        )
    elif len(rendered_texts) > MAX_RECOMMENDED_TEXT_ELEMENTS:
        report.warnings.append(
            f"Text density slightly high ({len(rendered_texts)} labels; ideal top-venue overview has 12-26 crisp labels)."
        )

    whitespace_ratio: Optional[float] = None
    if rendered_png_path is not None and rendered_png_path.is_file():
        whitespace_ratio = round(measure_whitespace_ratio(rendered_png_path), 3)
        if whitespace_ratio < MIN_WHITESPACE_RATIO:
            report.errors.append(
                f"Canvas is overcrowded: open white breathing space is only {whitespace_ratio * 100:.1f}% "
                f"(minimum {MIN_WHITESPACE_RATIO * 100:.0f}%, ideal 35%-45%). Increase gutters, outer margins, and simplify clutter!"
            )

    report.stats = {
        "viewBox": f"{int(vb_w)}x{int(vb_h)}",
        "aspect_ratio": round(vb_w / vb_h, 2),
        "text_elements": len(rendered_texts),
        "rect_boxes": len(rects),
        "embedded_visual_assets": embedded_images,
        "chromatic_hue_families": len(hue_buckets),
        "missing_inventory_count": len(missing_labels),
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
