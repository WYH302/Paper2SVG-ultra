"""Automated unit & integration tests for Paper2SVG-ultra v3.0 (Pure AutoFigure-Edit + Deterministic Vector Repair)."""

from __future__ import annotations

import base64
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pymupdf
from fastapi.testclient import TestClient
from PIL import Image

import run as cli_run
from paper2svg import __version__
from paper2svg.autofigure_engine import (
    draw_samed_placeholder_image,
    merge_overlapping_boxes,
    replace_icons_in_svg,
    resolve_matting_backend,
    resolve_seg_backend,
    upscale_image_to_4k_if_needed,
)
from paper2svg.client import ResponsesAPIClient, encode_image_to_data_url, is_retryable_stream_error
from paper2svg.config import (
    Settings,
    load_settings,
    resolve_density_mode,
    resolve_stage_effort,
    sanitize_font_family,
)
from paper2svg.figure_gen import build_autofigure_step1_prompt, pick_single_style_reference, trim_uniform_outer_margin
from paper2svg.icon_utils import (
    detect_cc_frame_stripped,
    looks_like_opaque_photo_tile,
    smart_remove_border_background,
    strip_edge_text_bleed,
    strip_text_edges_on_box,
)
from paper2svg.pdf_parser import extract_method_section, load_method_text
from paper2svg.svg_linter import lint_svg
from paper2svg.svg_postprocess import (
    autofit_text_in_rects,
    fix_collapsed_arrow_paths,
    postprocess_svg_deterministic,
    sanitize_svg_numeric_typos,
)
from paper2svg.svg_renderer import render_dual_preview
from server import app


def test_font_guardrail_and_config_defaults() -> None:
    assert __version__ == "3.0.0"
    assert "comic" not in sanitize_font_family("Comic Sans MS").lower()
    assert "Inter" in sanitize_font_family("Comic Sans MS", lang="en")
    assert "Noto Sans SC" in sanitize_font_family("", lang="zh")
    assert sanitize_font_family("Helvetica, Arial, sans-serif") == "Helvetica, Arial, sans-serif"

    defaults = Settings()
    assert defaults.base_url == "https://api.openai.com/v1"
    assert defaults.actor_auth == ""
    assert defaults.svg_mode == "hybrid_rich"
    assert defaults.optimize_iterations == 2
    assert defaults.disable_response_storage is True

    loaded = load_settings()
    assert loaded.svg_mode in ("hybrid_rich", "pure_vector")
    assert loaded.reasoning_effort in ("", "none", "low", "medium", "high", "xhigh")

    client = ResponsesAPIClient(Settings(api_key="test-key", main_model="gpt-5.4", reasoning_effort="xhigh"))
    payload = client._apply_reasoning({"model": "gpt-5.4", "temperature": 0.2}, temperature=0.2)
    assert payload["reasoning"] == {"effort": "xhigh"}
    assert "temperature" not in payload

    assert resolve_stage_effort("xhigh", "low") == "low"
    assert resolve_stage_effort("xhigh", "medium") == "medium"
    assert resolve_density_mode("auto", 20) == "concise"
    assert resolve_density_mode("auto", 45) == "detailed"
    assert is_retryable_stream_error("503 service_unavailable_error overloaded")
    assert not is_retryable_stream_error("DAILY_LIMIT_EXCEEDED: quota exhausted")


def test_strict_method_section_slicer_arabic_and_ieee_roman() -> None:
    with tempfile.TemporaryDirectory() as td:
        # 1. IEEE Roman numerals (`III. METHODOLOGY` -> stops cleanly before `IV. EXPERIMENTS`)
        ieee_pdf = Path(td) / "ieee_paper.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        y = 50
        ieee_blocks = [
            ("PhysFlow-SPI: Physics-Consistent Flow Matching", 15, True),
            ("ABSTRACT", 12, True),
            ("We propose PhysFlow-SPI for single-pixel face reconstruction.", 10, False),
            ("I. INTRODUCTION", 12, True),
            ("Single-pixel imaging faces severe ill-posedness in extreme compression.", 10, False),
            ("II. RELATED WORK", 12, True),
            ("Prior GAN and diffusion models (Vaswani et al., 2017) ignore measurement physics.", 10, False),
            ("III. METHODOLOGY", 12, True),
            ("Our Dual-Domain Conditioning fuses spatial semantic tokens with covariance geometry.", 10, False),
            ("A. Adaptive Data Consistency", 11, True),
            ("At each ODE step we project onto the physical measurement subspace with lambda_adapt.", 10, False),
            ("IV. EXPERIMENTS", 12, True),
            ("Table I shows PSNR=31.42 dB and FID=4.81 on CelebA-HQ benchmark.", 10, False),
            ("REFERENCES", 12, True),
            ("[1] Vaswani et al. Attention Is All You Need.", 10, False),
        ]
        for txt, sz, bold in ieee_blocks:
            page.insert_text((50, y), txt, fontsize=sz, fontname="hebo" if bold else "helv")
            y += sz + 9
        doc.save(str(ieee_pdf))
        doc.close()

        info = extract_method_section(ieee_pdf)
        assert info["detected_lang"] == "en"
        assert "Dual-Domain Conditioning" in info["method_text"]
        assert "lambda_adapt" in info["method_text"]
        assert "PSNR=31.42" not in info["method_text"]
        assert "Vaswani et al." not in info["method_text"]
        # Full paper title & abstract are omitted when Method section exists (prevents useless top title banners!)
        assert "PhysFlow-SPI: Physics-Consistent Flow Matching" not in info["method_text"]
        assert "single-pixel face reconstruction" not in info["method_text"]
        assert any("EXPERIMENTS" in h for h in info["dropped_headings"])

        # 2. Arabic numbering (`3. Method` -> stops cleanly before `4. Evaluation`)
        arabic_pdf = Path(td) / "arabic_paper.pdf"
        doc2 = pymupdf.open()
        p2 = doc2.new_page()
        y = 50
        for txt, sz, bold in [
            ("SpecAlign: Sparse Routing", 15, True),
            ("2. Related Work", 12, True),
            ("Related survey text.", 10, False),
            ("3. Method", 12, True),
            ("Sparse Token Router aligns visual patches with text tokens.", 10, False),
            ("4. Evaluation", 12, True),
            ("Our ablation achieves 98.7% top-1 accuracy.", 10, False),
        ]:
            p2.insert_text((50, y), txt, fontsize=sz, fontname="hebo" if bold else "helv")
            y += sz + 10
        doc2.save(str(arabic_pdf))
        doc2.close()

        info2 = extract_method_section(arabic_pdf)
        assert "Sparse Token Router" in info2["method_text"]
        assert "98.7% top-1" not in info2["method_text"]

        # 3. Direct method.txt loader
        mtxt = Path(td) / "method.txt"
        mtxt.write_text("Stage 1 encodes radar point clouds into voxel grids.", encoding="utf-8")
        info3 = load_method_text(mtxt)
        assert "voxel grids" in info3["method_text"]


def test_step1_prompt_builder_and_trimmer() -> None:
    p_ref = build_autofigure_step1_prompt("Our encoder maps X to Z.", has_style_ref=True, has_subject_images=True)
    assert "closely imitate the visual (artistic) style" in p_ref
    assert "ORIGINALITY & ACADEMIC LAYOUT GUARD" in p_ref
    assert "NO TOP TITLE BANNER" in p_ref
    assert "NO BOTTOM BOASTING" in p_ref
    assert "OWN PAPER SUBJECT IMAGE" in p_ref

    p_zero = build_autofigure_step1_prompt("Our encoder maps X to Z.", has_style_ref=False)
    assert "professional academic journal style figure" in p_zero
    assert "NO TOP TITLE BANNER" in p_zero
    assert "NO BOTTOM BOASTING" in p_zero

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        r1 = td_path / "01_ref.png"
        r2 = td_path / "02_ref.png"
        Image.new("RGB", (400, 200), (240, 240, 240)).save(r1)
        Image.new("RGB", (400, 200), (230, 230, 250)).save(r2)
        assert pick_single_style_reference(style_ref=r2, refs_dir=td_path) == r2
        assert pick_single_style_reference(refs_dir=td_path) == r1

    # Outer white margin trim
    canvas = np.full((300, 500, 3), 255, dtype=np.uint8)
    canvas[60:240, 80:420] = [40, 90, 200]
    trimmed = trim_uniform_outer_margin(Image.fromarray(canvas, mode="RGB"), pad=10)
    assert trimmed.width <= 365 and trimmed.height <= 205


def test_icon_utils_cc_photo_protection_and_floodfill() -> None:
    # 1. Border flood-fill keeps white interior 100% opaque
    arr = np.full((120, 160, 3), [239, 246, 255], dtype=np.uint8)
    arr[25:95, 35:125] = [37, 99, 235]
    arr[45:75, 55:105] = [255, 255, 255]
    cleaned, _ = smart_remove_border_background(Image.fromarray(arr, mode="RGB"))
    c_arr = np.array(cleaned)
    assert cleaned.width <= 94 and cleaned.height <= 74
    assert c_arr[c_arr.shape[0] // 2, c_arr.shape[1] // 2, 3] == 255

    # 2. Opaque photo tile protection vs white-bg icon
    rng = np.random.default_rng(42)
    photo_tile = Image.fromarray(rng.integers(20, 180, size=(80, 80, 3), dtype=np.uint8), mode="RGB")
    icon_tile = Image.new("RGB", (80, 80), (255, 255, 255))
    assert looks_like_opaque_photo_tile(photo_tile) is True
    assert looks_like_opaque_photo_tile(icon_tile) is False

    # 3. Strip open C-border + grazed bottom caption while keeping closed 4-sided frame untouched
    closed_frame = np.full((64, 84, 3), 255, dtype=np.uint8)
    closed_frame[:3, :] = 20
    closed_frame[-3:, :] = 20
    closed_frame[:, :3] = 20
    closed_frame[:, -3:] = 20
    closed_frame[14:50, 16:68] = 80
    cf_out, cf_trims = strip_edge_text_bleed(Image.fromarray(closed_frame, mode="RGB"))
    assert cf_trims == (0, 0, 0, 0) and cf_out.size == (84, 64)

    open_c = np.full((74, 77, 3), 255, dtype=np.uint8)
    open_c[:3, :] = 20
    open_c[-3:, :] = 20
    open_c[:, :3] = 20
    open_c[14:60, 18:56] = 30
    open_c[28:32, 72:77] = 20
    oc_out, oc_trims = strip_edge_text_bleed(Image.fromarray(open_c, mode="RGB"))
    assert oc_trims[0] >= 12 and oc_trims[2] >= 10 and oc_out.width <= 45

    # 4. Frame-stripped CC strips a 600px outer panel border and still finds all 3 disjoint colored icons inside
    syn = np.full((300, 700, 3), 255, dtype=np.uint8)
    syn[20:23, 20:680] = [30, 30, 30]
    syn[277:280, 20:680] = [30, 30, 30]
    syn[20:280, 20:23] = [30, 30, 30]
    syn[20:280, 677:680] = [30, 30, 30]
    syn[80:210, 55:185] = [60, 120, 220]
    syn[80:210, 280:410] = [220, 110, 60]
    syn[80:210, 500:630] = [70, 180, 120]
    cc_boxes = detect_cc_frame_stripped(Image.fromarray(syn, mode="RGB"))
    assert len(cc_boxes) == 3, f"Expected 3 icons inside stripped panel frame, got {len(cc_boxes)}: {cc_boxes}"

    stripped = strip_text_edges_on_box(syn.astype(np.float32), cc_boxes[0])
    assert stripped["bg_mode"] == "remove_bg"


def test_deterministic_svg_postprocess_arrows_autofit_siblings_and_images() -> None:
    # 1. Double-decimal typo repair
    assert sanitize_svg_numeric_typos('<rect width="95..0" height="12.0"/>') == '<rect width="95.0" height="12.0"/>'

    # 2. Collapsed triangle-wedge arrow repair (including compact negative coords like `l-10-11`)
    wedge_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300"><path d="M618 78h117l-10-11v22z" fill="#fff" stroke="#111"/></svg>'
    fixed_arr_svg, n_arr = fix_collapsed_arrow_paths(wedge_svg)
    assert n_arr == 1
    assert "L725.0" in fixed_arr_svg and "L735.0 78.0" in fixed_arr_svg

    # 3. CSS-safe box autofit + same-row sibling collision + image overlap nudger
    crowded_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 320" width="900" height="320">
      <style>
        .lbl { font-family: Arial, sans-serif; font-size: 22px; font-weight: 700; }
      </style>
      <rect x="0" y="0" width="900" height="320" fill="#FFFFFF"/>
      <rect x="40" y="40" width="110" height="44" rx="6" fill="#EFF6FF" stroke="#1E3A8A"/>
      <text x="95" y="68" text-anchor="middle" class="lbl">D4-Augmented Encoder</text>
      <rect x="40" y="120" width="130" height="36" rx="4" fill="#F0FDF4" stroke="#166534"/>
      <text x="48" y="144" text-anchor="start" font-family="Arial, sans-serif" font-size="16">Observed Free</text>
      <text x="148" y="144" text-anchor="start" font-family="Arial, sans-serif" font-size="16">1</text>
      <text x="500" y="108" text-anchor="start" font-family="Arial, sans-serif" font-size="22" font-weight="700">Occupancy</text>
      <image x="570" y="100" width="120" height="100" href="data:image/png;base64,AAAA"/>
    </svg>"""
    fixed_svg, stats = postprocess_svg_deterministic(crowded_svg)
    assert stats["texts_autofit"] >= 3, f"Expected box autofit + sibling + image nudge, got {stats}"
    assert "style=\"font-size:" in fixed_svg  # CSS-safe inline style overrides `.lbl`!
    # Verify `Occupancy` was nudged up or shrunk away from image y=100..200
    assert 'y="108"' not in fixed_svg or 'style="font-size:' in fixed_svg


def test_autofigure_swap_1to1_encoding_and_backends() -> None:
    small = Image.new("RGB", (1200, 600), (255, 255, 255))
    up, did_up = upscale_image_to_4k_if_needed(small, target_long_edge=3840)
    assert did_up and up.size == (3840, 1920)

    merged = merge_overlapping_boxes(
        [
            {"x1": 100, "y1": 50, "x2": 220, "y2": 170, "score": 0.9},
            {"x1": 104, "y1": 52, "x2": 218, "y2": 168, "score": 0.7},
            {"x1": 400, "y1": 50, "x2": 520, "y2": 170, "score": 0.85},
        ],
        overlap_threshold=0.80,
    )
    assert len(merged) == 2 and merged[0]["label"] == "<AF>01" and merged[1]["label"] == "<AF>02"

    # 1:1 native pixel encoding up to 4096px
    wide = Image.new("RGB", (2162, 445), (255, 255, 255))
    uri = encode_image_to_data_url(wide)
    dec = Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert dec.size == (2162, 445)

    assert resolve_seg_backend("grounding_dino") == "grounding_dino"
    assert resolve_seg_backend("cc_zero_gpu") == "cc_zero_gpu"
    assert resolve_matting_backend("birefnet") == "birefnet"
    assert resolve_matting_backend("floodfill") == "floodfill"

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        assert draw_samed_placeholder_image(small, merged, td_path / "samed.png").is_file()
        icon_png = td_path / "icon_AF01_nobg.png"
        ic_arr = np.zeros((60, 60, 4), dtype=np.uint8)
        ic_arr[10:50, 10:50] = [37, 99, 235, 255]
        Image.fromarray(ic_arr, mode="RGBA").save(icon_png)

        tmpl = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400" width="800" height="400">
          <rect width="800" height="400" fill="#FFFFFF"/>
          <g id="AF01" transform="translate(20, 10)">
            <rect x="80" y="40" width="120" height="120" fill="#808080" stroke="black" stroke-width="2"/>
            <text x="140" y="100" text-anchor="middle" fill="white">&lt;AF&gt;01</text>
          </g>
        </svg>"""
        swapped, cnt = replace_icons_in_svg(
            tmpl,
            [
                {
                    "id": 0,
                    "label": "<AF>01",
                    "label_clean": "AF01",
                    "x1": 100,
                    "y1": 50,
                    "width": 120,
                    "height": 120,
                    "orig_x1": 100,
                    "orig_y1": 50,
                    "orig_width": 120,
                    "orig_height": 120,
                    "bg_mode": "remove_bg",
                    "rounded_rx": 0,
                    "nobg_path": str(icon_png),
                }
            ],
            figure_size=(800, 400),
        )
        assert cnt == 1 and 'id="icon_AF01"' in swapped and 'x="100.0" y="50.0"' in swapped

        out_full = td_path / "final.png"
        out_2col = td_path / "print_2col.png"
        render_dual_preview(swapped, out_full, out_2col, viewbox=(800, 400), full_zoom=1.0)
        with Image.open(out_full) as r_im:
            r_arr = np.array(r_im.convert("RGB"))
            patch = r_arr[60:160, 110:210]
            assert int(np.sum((patch[:, :, 2] > 180) & (patch[:, :, 0] < 100))) > 500


def test_svg_linter_cli_and_fastapi_workbench() -> None:
    touching_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 680" width="1600" height="680">
      <rect width="1600" height="680" fill="#FFFFFF"/>
      <text x="500" y="300" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="700">Identity</text>
      <text x="580" y="300" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="700">Geometry</text>
    </svg>"""
    rep = lint_svg(touching_svg, expected_viewbox=(1600, 680))
    assert any("Text collision" in e for e in rep.errors)

    parser = cli_run.build_parser()
    parsed = parser.parse_args(
        [
            "--paper", "paper.pdf",
            "--style-ref", "ref.png",
            "--subject-images", "in.png", "out.png",
            "--seg-backend", "grounding_dino",
            "--matting-backend", "birefnet",
            "--optimize-iterations", "2",
            "--pause-after-step1",
        ]
    )
    assert parsed.paper == Path("paper.pdf")
    assert parsed.style_ref == Path("ref.png")
    assert parsed.subject_images == [Path("in.png"), Path("out.png")]
    assert parsed.pause_after_step1 is True
    assert parsed.optimize_iterations == 2

    client = TestClient(app)
    r_home = client.get("/")
    assert r_home.status_code == 200 and "Paper2SVG-ultra" in r_home.text
    r_cfg = client.get("/api/config")
    assert r_cfg.status_code == 200
    cfg = r_cfg.json()
    assert cfg["version"] == "3.0.0"
    assert cfg["svg_edit_url"] == "/svg-edit/editor/index.html"
    assert client.get("/svg-edit/editor/index.html").status_code == 200
    r_qr = client.get("/assets/sponsor_wechat.jpg")
    assert r_qr.status_code == 200 and r_qr.headers["content-type"].startswith("image/jpeg")
    assert client.get("/api/runs").status_code == 200
