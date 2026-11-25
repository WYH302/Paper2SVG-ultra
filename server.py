"""FastAPI Web Workbench for Paper2SVG-ultra with bundled svg-edit interactive canvas."""

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

from paper2svg.archetypes import ARCHETYPES
from paper2svg.config import PROJECT_ROOT, load_settings
from paper2svg.pipeline import run_pipeline
from paper2svg.svg_linter import validate_svg_xml
from paper2svg.svg_renderer import render_dual_preview

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
WEB_DIR = PROJECT_ROOT / "web"

def _find_svg_edit_dir() -> Optional[Path]:
    for candidate in (
        WEB_DIR / "vendor" / "svg-edit",
        PROJECT_ROOT / "others" / "repos" / "AutoFigure-Edit" / "web" / "vendor" / "svg-edit",
        PROJECT_ROOT / "repos" / "AutoFigure-Edit" / "web" / "vendor" / "svg-edit",
    ):
        if (candidate / "editor" / "index.html").is_file():
            return candidate
    return None

SVG_EDIT_DIR = _find_svg_edit_dir()

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Paper2SVG-ultra Workbench")

# Mount bundled MIT-licensed svg-edit canvas
if SVG_EDIT_DIR is not None:
    app.mount("/svg-edit", StaticFiles(directory=str(SVG_EDIT_DIR), html=True), name="svg-edit")


@dataclass
class JobState:
    job_id: str
    output_dir: Path
    status: str = "running"  # running | paused_after_schema | completed | failed
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
    "stage1_draft.png",
    "stage1_draft_raw.png",
    "stage1_draft_ruler.png",
    "stage1_draft_verify.png",
    "stage1_draft_verify_calibrated.png",
    "stage1_draft_masked.png",
    "assets_manifest.json",
    "raw.svg",
    "schema.md",
    "style_guide.md",
    "style_dna.json",
    "paper_method.md",
    "paper_analysis.json",
    "qa.md",
    "run_manifest.json",
    "stage4_render_prompt.txt",
    "stage4_meta.json",
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
    paper_pdf: Path,
    refs_dir: Optional[Path],
    lang: str,
    archetype: Optional[str],
    critic_rounds: int,
    main_model: Optional[str],
    reasoning_effort: Optional[str],
    image_model: Optional[str],
    svg_mode: Optional[str],
    density: Optional[str] = None,
    primary_ref: Optional[Path] = None,
    from_stage: int = 1,
    pause_after_schema: bool = False,
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
            if density and density.strip():
                settings.density = density.strip().lower()

            def _on_event(evt_type: str, payload: dict[str, Any]) -> None:
                job.push(evt_type, payload)

            res = run_pipeline(
                paper_pdf=paper_pdf,
                refs_dir=refs_dir,
                output_dir=job.output_dir,
                settings=settings,
                lang=lang,
                archetype_override=archetype or None,
                svg_mode_override=settings.svg_mode,
                density_override=settings.density,
                primary_ref=primary_ref,
                critic_rounds=critic_rounds,
                from_stage=from_stage,
                stop_after_stage=3 if pause_after_schema else None,
                on_event=_on_event,
            )
            job.status = "paused_after_schema" if pause_after_schema else "completed"
            job.push(
                "done",
                {
                    "status": job.status,
                    "job_id": job.job_id,
                    "archetype": res.archetype,
                    "viewbox": list(res.viewbox),
                    "elapsed_seconds": res.elapsed_seconds,
                },
            )
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.push("error", {"status": "failed", "error": str(exc)})

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/config")
def get_config() -> JSONResponse:
    s = load_settings()
    svg_edit_url = "/svg-edit/editor/index.html" if (SVG_EDIT_DIR is not None and (SVG_EDIT_DIR / "editor" / "index.html").is_file()) else None
    return JSONResponse(
        {
            "main_model": s.main_model,
            "reasoning_effort": s.reasoning_effort,
            "image_tool_model": s.image_tool_model,
            "image_quality": s.image_quality,
            "svg_mode": s.svg_mode,
            "density": s.density,
            "critic_rounds": s.critic_rounds,
            "font_family": s.font_family,
            "lang": s.lang,
            "api_configured": bool(s.api_key),
            "svg_edit_url": svg_edit_url,
            "archetypes": [
                {
                    "key": a.key,
                    "zh_name": a.zh_name,
                    "en_name": a.en_name,
                    "story_trigger": a.story_trigger,
                    "is_wide_banner": a.is_wide_banner,
                }
                for a in ARCHETYPES.values()
            ],
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
        runs.append(
            {
                "job_id": d.name,
                "status": status,
                "created_at": meta.get("created_at", datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds")),
                "archetype": meta.get("archetype", ""),
                "archetype_name": meta.get("archetype_name", ""),
                "viewbox": meta.get("viewbox"),
                "paper_name": Path(meta["paper_pdf"]).name if meta.get("paper_pdf") else d.name,
                "has_final_svg": (d / "final.svg").is_file(),
                "has_schema": (d / "schema.md").is_file(),
                "has_draft": (d / "stage1_draft.png").is_file(),
            }
        )
    return JSONResponse({"runs": runs[:50]})


@app.post("/api/run")
async def start_run(
    paper_pdf: UploadFile = File(...),
    ref_images: list[UploadFile] = File(default=[]),
    refs_dir_path: str = Form(default=""),
    primary_ref_path: str = Form(default=""),
    lang: str = Form(default="auto"),
    archetype: str = Form(default=""),
    critic_rounds: int = Form(default=2),
    main_model: str = Form(default=""),
    reasoning_effort: str = Form(default=""),
    image_model: str = Form(default=""),
    svg_mode: str = Form(default=""),
    density: str = Form(default="auto"),
    pause_after_schema: bool = Form(default=False),
) -> JSONResponse:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    orig_stem = Path(paper_pdf.filename or "paper.pdf").stem
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", orig_stem)[:28].strip("_") or "paper"
    job_id = f"{safe_stem}_{stamp}_{short_uid}"

    upload_dir = UPLOADS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf = upload_dir / (Path(paper_pdf.filename or "paper.pdf").name)
    saved_pdf.write_bytes(await paper_pdf.read())

    # Resolve reference folder: uploaded files take priority; else optional local folder path
    resolved_refs_dir: Optional[Path] = None
    valid_uploads = [f for f in ref_images if f and f.filename]
    if valid_uploads:
        refs_upload_dir = upload_dir / "refs"
        refs_upload_dir.mkdir(parents=True, exist_ok=True)
        for idx, rf in enumerate(valid_uploads, 1):
            fname = Path(rf.filename or f"ref_{idx}.png").name
            (refs_upload_dir / f"{idx:02d}_{fname}").write_bytes(await rf.read())
        resolved_refs_dir = refs_upload_dir
    elif refs_dir_path.strip():
        candidate = Path(refs_dir_path.strip())
        if not candidate.exists():
            raise HTTPException(status_code=400, detail=f"Reference folder does not exist: {candidate}")
        resolved_refs_dir = candidate

    resolved_primary_ref: Optional[Path] = None
    if primary_ref_path.strip():
        p_cand = Path(primary_ref_path.strip())
        if not p_cand.is_file():
            raise HTTPException(status_code=400, detail=f"Primary reference image does not exist: {p_cand}")
        resolved_primary_ref = p_cand

    out_dir = OUTPUTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    job = JobState(job_id=job_id, output_dir=out_dir)
    JOBS[job_id] = job

    _start_pipeline_thread(
        job,
        paper_pdf=saved_pdf,
        refs_dir=resolved_refs_dir,
        lang=lang,
        archetype=archetype.strip() or None,
        critic_rounds=max(1, min(4, int(critic_rounds))),
        main_model=main_model.strip() or None,
        reasoning_effort=reasoning_effort.strip() or None,
        image_model=image_model.strip() or None,
        svg_mode=svg_mode.strip() or None,
        density=density.strip() or None,
        primary_ref=resolved_primary_ref,
        from_stage=1,
        pause_after_schema=pause_after_schema,
    )
    return JSONResponse({"job_id": job_id, "status": "running"})


class ResumeRequest(BaseModel):
    from_stage: int = 4
    schema_md: Optional[str] = None
    critic_rounds: int = 2
    image_model: Optional[str] = None
    main_model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    svg_mode: Optional[str] = None
    density: Optional[str] = None


@app.post("/api/runs/{job_id}/resume")
def resume_run(job_id: str, req: ResumeRequest) -> JSONResponse:
    out_dir = _safe_job_dir(job_id)
    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    if req.schema_md is not None:
        (out_dir / "schema.md").write_text(req.schema_md, encoding="utf-8")

    manifest_file = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else {}
    paper_pdf = Path(manifest.get("paper_pdf", ""))
    refs_dir = Path(manifest["refs_dir"]) if manifest.get("refs_dir") else None
    primary_ref = Path(manifest["primary_ref"]) if manifest.get("primary_ref") else None

    job = JobState(job_id=job_id, output_dir=out_dir)
    JOBS[job_id] = job

    _start_pipeline_thread(
        job,
        paper_pdf=paper_pdf,
        refs_dir=refs_dir,
        lang=manifest.get("lang", "auto"),
        archetype=manifest.get("archetype") or None,
        critic_rounds=max(1, min(4, req.critic_rounds)),
        main_model=req.main_model or manifest.get("main_model"),
        reasoning_effort=req.reasoning_effort or manifest.get("reasoning_effort"),
        image_model=req.image_model or manifest.get("image_tool_model"),
        svg_mode=req.svg_mode or manifest.get("svg_mode"),
        density=req.density or manifest.get("density"),
        primary_ref=primary_ref,
        from_stage=max(1, min(6, req.from_stage)),
        pause_after_schema=False,
    )
    return JSONResponse({"job_id": job_id, "status": "running", "from_stage": req.from_stage})


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
    """Persist edits from the embedded svg-edit canvas back to final.svg & re-render PNG previews."""
    out_dir = _safe_job_dir(job_id)
    if not out_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    ok, errs = validate_svg_xml(req.svg_code)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Invalid SVG XML: {errs[:3]}")

    final_svg_path = out_dir / "final.svg"
    final_svg_path.write_text(req.svg_code, encoding="utf-8")

    # Parse viewBox for re-render
    m = re.search(r'viewBox="[^"]*?(\d+(?:\.\d+)?)[\s,]+(\d+(?:\.\d+)?)\s*"', req.svg_code)
    vb = (int(float(m.group(1))), int(float(m.group(2)))) if m else (1600, 680)
    render_dual_preview(
        req.svg_code,
        out_dir / "final.png",
        out_dir / "print_preview_2col.png",
        viewbox=vb,
        full_zoom=2.0,
    )
    return JSONResponse({"saved": True, "viewbox": list(vb)})
