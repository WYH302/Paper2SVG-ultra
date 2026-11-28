"""FastAPI Web Workbench for Paper2SVG-ultra v3.0 (Pure AutoFigure-Edit + Local GPU/Zero-GPU + svg-edit)."""

from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from paper2svg import __version__
from paper2svg.autofigure_engine import probe_gpu_capabilities, resolve_matting_backend, resolve_seg_backend
from paper2svg.config import PROJECT_ROOT, load_settings
from paper2svg.pipeline import run_pipeline
from paper2svg.svg_linter import validate_svg_xml
from paper2svg.svg_renderer import render_dual_preview

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
WEB_DIR = PROJECT_ROOT / "web"
ASSETS_DIR = PROJECT_ROOT / "assets"


def _find_svg_edit_dir() -> Optional[Path]:
    for candidate in (
        WEB_DIR / "vendor" / "svg-edit",
        PROJECT_ROOT / "others" / "repos" / "AutoFigure-Edit" / "web" / "vendor" / "svg-edit",
    ):
        if (candidate / "editor" / "index.html").is_file():
            return candidate
    return None


SVG_EDIT_DIR = _find_svg_edit_dir()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=f"Paper2SVG-ultra v{__version__} Workbench")

if SVG_EDIT_DIR is not None:
    app.mount("/svg-edit", StaticFiles(directory=str(SVG_EDIT_DIR), html=True), name="svg-edit")


@dataclass
class JobState:
    job_id: str
    output_dir: Path
    status: str = "running"  # running | paused_after_step1 | completed | failed
    error: Optional[str] = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[queue.Queue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, event_type: str, data: dict[str, Any]) -> None:
        item = {"event": event_type, "data": data}
        with self.lock:
            self.events.append(item)
            for q in list(self.subscribers):
                q.put(item)


JOBS: dict[str, JobState] = {}

ALLOWED_ARTIFACTS = {
    "final.svg",
    "final.png",
    "print_preview_2col.png",
    "figure.png",
    "figure_raw.png",
    "samed.png",
    "samed_candidates.png",
    "boxlib.json",
    "template.svg",
    "optimized_template.svg",
    "final_pre_polish.svg",
    "final_pre_polish.png",
    "method_section.txt",
    "step1_prompt.txt",
    "step1_meta.json",
    "qa.md",
    "run_manifest.json",
}


def _safe_job_dir(job_id: str) -> Path:
    if not re.match(r"^[A-Za-z0-9_-]{3,80}$", job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id")
    path = (OUTPUTS_DIR / job_id).resolve()
    if OUTPUTS_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="Path escape blocked")
    return path


def _start_pipeline_thread(
    job: JobState,
    *,
    paper_pdf: Optional[Path] = None,
    method_file: Optional[Path] = None,
    method_text_override: Optional[str] = None,
    input_figure: Optional[Path] = None,
    style_ref: Optional[Path] = None,
    subject_images: Optional[list[Path]] = None,
    lang: str = "auto",
    optimize_iterations: int = 2,
    main_model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    image_model: Optional[str] = None,
    svg_mode: Optional[str] = None,
    seg_backend: Optional[str] = None,
    matting_backend: Optional[str] = None,
    from_step: int = 1,
    pause_after_step1: bool = False,
) -> None:
    def _worker() -> None:
        try:
            settings = load_settings()
            if main_model:
                settings.main_model = main_model.strip()
            if reasoning_effort is not None and reasoning_effort.strip():
                settings.reasoning_effort = reasoning_effort.strip().lower()
            if image_model:
                settings.image_tool_model = image_model.strip()
            if svg_mode and svg_mode.strip():
                settings.svg_mode = svg_mode.strip().lower()
            if seg_backend and seg_backend.strip():
                settings.seg_backend = seg_backend.strip().lower()
            if matting_backend and matting_backend.strip():
                settings.matting_backend = matting_backend.strip().lower()

            res = run_pipeline(
                paper_pdf=paper_pdf,
                method_file=method_file,
                method_text_override=method_text_override,
                input_figure=input_figure,
                style_ref=style_ref,
                subject_images=subject_images,
                output_dir=job.output_dir,
                settings=settings,
                lang=lang,
                svg_mode_override=settings.svg_mode,
                seg_backend_override=settings.seg_backend,
                matting_backend_override=settings.matting_backend,
                optimize_iterations=optimize_iterations,
                from_step=from_step,
                stop_after_step=1 if pause_after_step1 else None,
                on_event=lambda evt_type, payload: job.push(evt_type, payload),
            )
            job.status = "paused_after_step1" if pause_after_step1 else "completed"
            job.push(
                "done",
                {
                    "status": job.status,
                    "job_id": job.job_id,
                    "seg_backend": res.seg_backend,
                    "matting_backend": res.matting_backend,
                    "viewbox": list(res.viewbox),
                    "elapsed_seconds": res.elapsed_seconds,
                },
            )
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.push("error", {"status": "failed", "error": str(exc)})

    threading.Thread(target=_worker, daemon=True).start()


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/assets/sponsor_wechat.jpg")
def get_sponsor_qr() -> FileResponse:
    qr_path = ASSETS_DIR / "sponsor_wechat.jpg"
    if not qr_path.is_file():
        raise HTTPException(status_code=404, detail="Sponsor QR not found")
    return FileResponse(qr_path, media_type="image/jpeg")


@app.get("/api/config")
def get_config() -> JSONResponse:
    s = load_settings()
    caps = probe_gpu_capabilities(s.gpu_python)
    svg_edit_url = (
        "/svg-edit/editor/index.html"
        if (SVG_EDIT_DIR is not None and (SVG_EDIT_DIR / "editor" / "index.html").is_file())
        else None
    )
    return JSONResponse(
        {
            "version": __version__,
            "main_model": s.main_model,
            "reasoning_effort": s.reasoning_effort,
            "image_tool_model": s.image_tool_model,
            "svg_mode": s.svg_mode,
            "seg_backend": s.seg_backend,
            "resolved_seg_backend": resolve_seg_backend(s.seg_backend, s.sam_api_key, gpu_python=s.gpu_python),
            "matting_backend": s.matting_backend,
            "resolved_matting_backend": resolve_matting_backend(s.matting_backend, s.rmbg_model_path, gpu_python=s.gpu_python),
            "gpu_name": caps.get("gpu_name") or "",
            "optimize_iterations": s.optimize_iterations,
            "font_family": s.font_family,
            "lang": s.lang,
            "api_configured": bool(s.api_key),
            "svg_edit_url": svg_edit_url,
        }
    )


@app.get("/api/runs")
def list_runs() -> JSONResponse:
    runs: list[dict[str, Any]] = []
    for d in sorted(OUTPUTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        manifest_file = d / "run_manifest.json"
        meta: dict[str, Any] = {}
        if manifest_file.is_file():
            try:
                meta = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        live_job = JOBS.get(d.name)
        status = live_job.status if live_job else meta.get("status", "completed" if (d / "final.svg").is_file() else "unknown")
        src_name = d.name
        for k in ("paper_pdf", "method_file", "input_figure"):
            if meta.get(k):
                src_name = Path(meta[k]).name
                break
        runs.append(
            {
                "job_id": d.name,
                "status": status,
                "created_at": meta.get("created_at", datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds")),
                "seg_backend": meta.get("seg_backend", ""),
                "matting_backend": meta.get("matting_backend", ""),
                "viewbox": meta.get("viewbox"),
                "paper_name": src_name,
                "has_final_svg": (d / "final.svg").is_file(),
                "has_figure": (d / "figure.png").is_file(),
                "has_method": (d / "method_section.txt").is_file(),
            }
        )
    return JSONResponse({"runs": runs[:50]})


@app.post("/api/run")
async def start_run(
    paper_pdf: Optional[UploadFile] = File(default=None),
    method_text: str = Form(default=""),
    input_figure: Optional[UploadFile] = File(default=None),
    style_ref: Optional[UploadFile] = File(default=None),
    style_ref_path: str = Form(default=""),
    subject_images: list[UploadFile] = File(default=[]),
    lang: str = Form(default="auto"),
    optimize_iterations: int = Form(default=2),
    main_model: str = Form(default=""),
    reasoning_effort: str = Form(default=""),
    image_model: str = Form(default=""),
    svg_mode: str = Form(default="hybrid_rich"),
    seg_backend: str = Form(default="auto"),
    matting_backend: str = Form(default="auto"),
    pause_after_step1: bool = Form(default=False),
) -> JSONResponse:
    has_pdf = paper_pdf is not None and bool(paper_pdf.filename)
    has_txt = bool(method_text.strip())
    has_fig = input_figure is not None and bool(input_figure.filename)
    if not has_pdf and not has_txt and not has_fig:
        raise HTTPException(status_code=400, detail="Please upload a paper PDF, paste Method text, or upload a Step-1 figure.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    orig_stem = Path((paper_pdf.filename if has_pdf else (input_figure.filename if has_fig else "method")) or "paper").stem  # type: ignore[union-attr]
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", orig_stem)[:28].strip("_") or "paper"
    job_id = f"{safe_stem}_{stamp}_{short_uid}"

    upload_dir = UPLOADS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_pdf: Optional[Path] = None
    if has_pdf and paper_pdf is not None:
        saved_pdf = upload_dir / Path(paper_pdf.filename or "paper.pdf").name
        saved_pdf.write_bytes(await paper_pdf.read())

    saved_fig: Optional[Path] = None
    if has_fig and input_figure is not None:
        saved_fig = upload_dir / Path(input_figure.filename or "figure.png").name
        saved_fig.write_bytes(await input_figure.read())

    saved_style_ref: Optional[Path] = None
    if style_ref is not None and style_ref.filename:
        saved_style_ref = upload_dir / f"style_ref_{Path(style_ref.filename).name}"
        saved_style_ref.write_bytes(await style_ref.read())
    elif style_ref_path.strip():
        cand = Path(style_ref_path.strip())
        if not cand.is_file():
            raise HTTPException(status_code=400, detail=f"Style reference file does not exist: {cand}")
        saved_style_ref = cand

    saved_subjects: list[Path] = []
    valid_subs = [f for f in subject_images if f and f.filename]
    if valid_subs:
        subj_dir = upload_dir / "subjects"
        subj_dir.mkdir(parents=True, exist_ok=True)
        for idx, sf in enumerate(valid_subs, 1):
            p = subj_dir / f"{idx:02d}_{Path(sf.filename or f'subj_{idx}.png').name}"
            p.write_bytes(await sf.read())
            saved_subjects.append(p)

    out_dir = OUTPUTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    job = JobState(job_id=job_id, output_dir=out_dir)
    JOBS[job_id] = job

    _start_pipeline_thread(
        job,
        paper_pdf=saved_pdf,
        method_text_override=method_text.strip() or None,
        input_figure=saved_fig,
        style_ref=saved_style_ref,
        subject_images=saved_subjects or None,
        lang=lang,
        optimize_iterations=max(0, min(4, int(optimize_iterations))),
        main_model=main_model.strip() or None,
        reasoning_effort=reasoning_effort.strip() or None,
        image_model=image_model.strip() or None,
        svg_mode=svg_mode.strip() or "hybrid_rich",
        seg_backend=seg_backend.strip() or "auto",
        matting_backend=matting_backend.strip() or "auto",
        from_step=1,
        pause_after_step1=pause_after_step1,
    )
    return JSONResponse({"job_id": job_id, "status": "running"})


class ResumeRequest(BaseModel):
    from_step: int = 2
    method_text: Optional[str] = None
    optimize_iterations: int = 2
    main_model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    image_model: Optional[str] = None
    svg_mode: Optional[str] = None
    seg_backend: Optional[str] = None
    matting_backend: Optional[str] = None
    pause_after_step1: bool = False


@app.post("/api/runs/{job_id}/resume")
def resume_run(job_id: str, req: ResumeRequest) -> JSONResponse:
    out_dir = _safe_job_dir(job_id)
    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    if req.method_text is not None:
        (out_dir / "method_section.txt").write_text(req.method_text, encoding="utf-8")

    manifest_file = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else {}
    style_ref = Path(manifest["style_ref"]) if manifest.get("style_ref") else None
    subjects = [Path(p) for p in manifest.get("subject_images", []) if Path(p).is_file()]

    job = JobState(job_id=job_id, output_dir=out_dir)
    JOBS[job_id] = job

    _start_pipeline_thread(
        job,
        method_text_override=req.method_text,
        style_ref=style_ref,
        subject_images=subjects or None,
        lang=manifest.get("lang", "auto"),
        optimize_iterations=max(0, min(4, req.optimize_iterations)),
        main_model=req.main_model or manifest.get("main_model"),
        reasoning_effort=req.reasoning_effort or manifest.get("reasoning_effort"),
        image_model=req.image_model or manifest.get("image_tool_model"),
        svg_mode=req.svg_mode or manifest.get("svg_mode"),
        seg_backend=req.seg_backend or manifest.get("seg_backend"),
        matting_backend=req.matting_backend or manifest.get("matting_backend"),
        from_step=max(1, min(2, req.from_step)),
        pause_after_step1=req.pause_after_step1,
    )
    return JSONResponse({"job_id": job_id, "status": "running", "from_step": req.from_step})


@app.get("/api/runs/{job_id}/events")
async def stream_events(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if job is None:
        out_dir = _safe_job_dir(job_id)
        status = "completed" if (out_dir / "final.svg").is_file() else "unknown"

        async def _static_gen():
            yield f"event: done\ndata: {json.dumps({'status': status, 'job_id': job_id})}\n\n"

        return StreamingResponse(_static_gen(), media_type="text/event-stream")

    sub_q: queue.Queue = queue.Queue()
    with job.lock:
        replay = list(job.events)
        job.subscribers.append(sub_q)

    async def _event_gen():
        try:
            for item in replay:
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                if item["event"] in ("done", "error"):
                    return
            while True:
                try:
                    item = await asyncio.to_thread(sub_q.get, True, 15.0)
                    yield f"event: {item['event']}\ndata: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
                    if item["event"] in ("done", "error"):
                        return
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with job.lock:
                if sub_q in job.subscribers:
                    job.subscribers.remove(sub_q)

    return StreamingResponse(_event_gen(), media_type="text/event-stream")


@app.get("/api/runs/{job_id}/artifacts/{filename}")
def get_artifact(job_id: str, filename: str):
    out_dir = _safe_job_dir(job_id)
    if filename not in ALLOWED_ARTIFACTS and not re.match(r"^iter_\d+(_full|_2col)?\.(svg|png)$", filename):
        raise HTTPException(status_code=403, detail="Artifact not permitted")
    target = (out_dir / filename).resolve()
    if out_dir.resolve() not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {filename}")

    if filename.endswith(".svg"):
        return PlainTextResponse(target.read_text(encoding="utf-8"), media_type="image/svg+xml")
    if filename.endswith((".md", ".txt", ".json")):
        return PlainTextResponse(target.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")
    return FileResponse(target)


class SaveSvgRequest(BaseModel):
    svg_code: str


@app.post("/api/runs/{job_id}/save-svg")
def save_edited_svg(job_id: str, req: SaveSvgRequest) -> JSONResponse:
    """Persist manual edits from embedded svg-edit back to final.svg & refresh PNG previews."""
    out_dir = _safe_job_dir(job_id)
    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    ok, errs = validate_svg_xml(req.svg_code)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Invalid SVG XML: {errs[:3]}")

    (out_dir / "final.svg").write_text(req.svg_code, encoding="utf-8")
    m = re.search(r'viewBox="[^"]*?(\d+(?:\.\d+)?)[\s,]+(\d+(?:\.\d+)?)\s*"', req.svg_code)
    vb = (int(float(m.group(1))), int(float(m.group(2)))) if m else (1600, 800)
    render_dual_preview(req.svg_code, out_dir / "final.png", out_dir / "print_preview_2col.png", viewbox=vb, full_zoom=2.0)
    return JSONResponse({"saved": True, "viewbox": list(vb)})
