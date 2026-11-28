"""Local CUDA GPU Worker for Paper2SVG-ultra v3.0 (SAM3 / GroundingDINO-Base / RMBG-2.0 / BiRefNet).

Can be invoked EITHER directly in-process (when PyTorch+CUDA is installed in the current venv) OR as a lightweight
subprocess via `GPU_PYTHON` in `.env` (so the main Paper2SVG-ultra venv stays fast & dependency-free while still
harnessing a local NVIDIA GPU such as an RTX 3090!).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

# Curated specific visual object nouns for GroundingDINO (NEVER generic 'icon.' / 'grid.' which fire on text cards!)
GROUNDING_DINO_PROMPTS: list[tuple[str, float]] = [
    ("person.", 0.24),
    ("face.", 0.22),
    ("blurred photo.", 0.20),
    ("photograph.", 0.24),
    ("car.", 0.25),
    ("cube.", 0.20),
    ("3d block.", 0.22),
    ("robot.", 0.24),
    ("microchip.", 0.24),
    ("gear.", 0.24),
    ("shield.", 0.24),
    ("snowflake.", 0.22),
    ("book.", 0.25),
    ("document.", 0.25),
    ("database.", 0.24),
    ("magnifying glass.", 0.24),
    ("satellite dish.", 0.20),
    ("antenna tower.", 0.23),
    ("waveform.", 0.24),
    ("heatmap.", 0.22),
    ("matrix grid.", 0.22),
    ("noise pattern.", 0.22),
    ("bar chart.", 0.20),
    ("radar chart.", 0.23),
    ("ellipse.", 0.25),
    ("checkerboard.", 0.22),
]


def _get_hf_token() -> str | None:
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    return None


def run_sam3_local_gpu(
    image_path: Path,
    prompts: list[str],
    min_score: float = 0.45,
) -> list[dict[str, Any]]:
    """Run upstream Meta `facebook/sam3` on local CUDA GPU."""
    import sam3
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    img = Image.open(image_path).convert("RGB")
    sam3_dir = Path(sam3.__path__[0]) if hasattr(sam3, "__path__") else Path(sam3.__file__).parent
    bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(device=device, bpe_path=str(bpe_path) if bpe_path.exists() else None)
    processor = Sam3Processor(model, device=device)
    state = processor.set_image(img)

    detected: list[dict[str, Any]] = []
    for prompt in prompts:
        out = processor.set_text_prompt(state=state, prompt=prompt)
        boxes = out["boxes"].cpu().numpy() if isinstance(out["boxes"], torch.Tensor) else out["boxes"]
        scores = out["scores"].cpu().numpy() if isinstance(out["scores"], torch.Tensor) else out["scores"]
        for box, sc in zip(boxes, scores):
            if float(sc) >= min_score:
                detected.append(
                    {
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                        "score": round(float(sc), 4),
                        "prompt": f"sam3:{prompt}",
                    }
                )
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return detected


def run_grounding_dino_gpu(image_path: Path) -> list[dict[str, Any]]:
    """Run open zero-token `IDEA-Research/grounding-dino-base` on local GPU (~5s/image on RTX 3090)."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(det_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(det_id).to(device).eval()

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    detected: list[dict[str, Any]] = []

    for prompt, thresh in GROUNDING_DINO_PROMPTS:
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=thresh,
            text_threshold=max(0.18, thresh - 0.03),
            target_sizes=[(h, w)],
        )[0]
        labels = res.get("text_labels") or res.get("labels") or []
        for sc, lb, bx in zip(res["scores"].tolist(), labels, res["boxes"].tolist()):
            x1, y1, x2, y2 = [int(round(v)) for v in bx]
            bw, bh = x2 - x1, y2 - y1
            if bw < 22 or bh < 22 or bw > w * 0.28 or bh > h * 0.45:
                continue
            detected.append(
                {
                    "x1": max(0, x1),
                    "y1": max(0, y1),
                    "x2": min(w, x2),
                    "y2": min(h, y2),
                    "score": round(float(sc), 4),
                    "prompt": f"gd:{lb}",
                }
            )

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return detected


def run_neural_matting_batch_gpu(
    pairs: list[tuple[Path, Path]],
    *,
    backend: str = "birefnet",
    rmbg_model_path: str = "",
) -> int:
    """Batch-matte icon crops on local CUDA GPU via `briaai/RMBG-2.0` or zero-token `ZhengPeng7/BiRefNet`."""
    if not pairs:
        return 0
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = _get_hf_token()
    if backend == "rmbg2":
        repo = rmbg_model_path if (rmbg_model_path and Path(rmbg_model_path).exists()) else "briaai/RMBG-2.0"
        model = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True, token=hf_token).eval().to(device)
    else:
        model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).eval().to(device)

    tf = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model_dtype = next(model.parameters()).dtype
    done = 0
    for src_path, dst_path in pairs:
        rgb = Image.open(src_path).convert("RGB")
        inp = tf(rgb).unsqueeze(0).to(device=device, dtype=model_dtype)
        with torch.no_grad():
            pred = model(inp)[-1].float().sigmoid().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(rgb.size)
        out = rgb.copy()
        out.putalpha(mask)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst_path, format="PNG")
        done += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return done


def _probe_capabilities() -> dict[str, Any]:
    import importlib.util

    has_torch = importlib.util.find_spec("torch") is not None
    cuda = False
    gpu_name = ""
    if has_torch:
        import torch

        cuda = bool(torch.cuda.is_available())
        if cuda:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "CUDA GPU"
    has_transformers = importlib.util.find_spec("transformers") is not None
    has_sam3 = importlib.util.find_spec("sam3") is not None
    has_kornia = importlib.util.find_spec("kornia") is not None
    has_timm = importlib.util.find_spec("timm") is not None
    has_rembg = importlib.util.find_spec("rembg") is not None
    return {
        "python": sys.executable,
        "torch": has_torch,
        "cuda": cuda,
        "gpu_name": gpu_name,
        "transformers": has_transformers,
        "sam3": has_sam3,
        "kornia": has_kornia,
        "timm": has_timm,
        "rembg": has_rembg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper2SVG-ultra v3.0 GPU Worker")
    parser.add_argument("--task", required=True, choices=["probe", "sam3", "grounding_dino", "rmbg2", "birefnet"])
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--prompts", type=str, default="icon,person,robot,animal,diagram,chart")
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--manifest", type=Path, default=None, help="JSON list of {src, dst} pairs for matting")
    parser.add_argument("--rmbg-model-path", type=str, default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.task == "probe":
        info = _probe_capabilities()
        txt = json.dumps(info, ensure_ascii=False)
        if args.out:
            args.out.write_text(txt, encoding="utf-8")
        else:
            print(txt)
        return 0

    if args.task == "sam3":
        assert args.image is not None and args.out is not None
        prompt_list = [p.strip() for p in args.prompts.split(",") if p.strip()] or ["icon"]
        boxes = run_sam3_local_gpu(args.image, prompt_list, min_score=args.min_score)
        args.out.write_text(json.dumps(boxes, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    if args.task == "grounding_dino":
        assert args.image is not None and args.out is not None
        boxes = run_grounding_dino_gpu(args.image)
        args.out.write_text(json.dumps(boxes, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    if args.task in ("rmbg2", "birefnet"):
        assert args.manifest is not None
        items = json.loads(args.manifest.read_text(encoding="utf-8"))
        pairs = [(Path(it["src"]), Path(it["dst"])) for it in items]
        done = run_neural_matting_batch_gpu(pairs, backend=args.task, rmbg_model_path=args.rmbg_model_path)
        if args.out:
            args.out.write_text(json.dumps({"matted": done}, ensure_ascii=False), encoding="utf-8")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
