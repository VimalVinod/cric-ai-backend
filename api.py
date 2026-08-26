"""
api.py

Cric AI - FastAPI Backend (Background Task Optimized for Hugging Face/Render)
"""

import json
import uuid
import shutil
import asyncio
import gc
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "api_uploads"
RESULT_DIR = BASE_DIR / "api_results"
REPORT_DIR = BASE_DIR / "api_reports"
ANNOTATED_DIR = BASE_DIR / "api_annotated"

PHASE1_ENGINE = BASE_DIR / "phase1_engine.py"
ANNOTATOR = BASE_DIR / "annotate_video.py"
PHASE6_REPORT = BASE_DIR / "phase6_report.py"
MODEL_PATH = BASE_DIR / "yolov8n-pose.pt"

# Create directories
for dir_path in [UPLOAD_DIR, RESULT_DIR, REPORT_DIR, ANNOTATED_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Cric AI API",
    description="AI-powered cricket bowling biomechanics analysis API",
    version="1.2.0", # Bumped to reflect background task update
)

# ============================================================
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cricai-amber.vercel.app",
        "http://localhost:3000", # Added for local frontend testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store to track background progress
jobs_db: Dict[str, Any] = {}

# ============================================================
# ROOT & HEALTH
# ============================================================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Cric AI API",
        "version": "1.2.0",
        "message": "Cric AI bowling biomechanics API is running.",
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Cric AI",
        "phase1_engine": PHASE1_ENGINE.exists(),
        "annotator": ANNOTATOR.exists(),
        "phase6_report": PHASE6_REPORT.exists(),
        "pose_model": MODEL_PATH.exists(),
    }

# ============================================================
# ANALYZE VIDEO (BACKGROUND TASK)
# ============================================================
@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    arm: str = "right",
):
    print("=" * 60, flush=True)
    print("ANALYZE REQUEST RECEIVED", flush=True)
    print(f"Filename: {file.filename}", flush=True)
    print(f"Arm: {arm}", flush=True)
    print("=" * 60, flush=True)

    arm = arm.lower().strip()
    if arm not in ("right", "left"):
        raise HTTPException(status_code=400, detail="arm must be 'right' or 'left'")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename supplied.")

    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    extension = Path(file.filename).suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    analysis_id = uuid.uuid4().hex[:12]

    video_path = UPLOAD_DIR / f"{analysis_id}{extension}"
    json_path = RESULT_DIR / f"{analysis_id}_phase1_result.json"
    annotated_path = ANNOTATED_DIR / f"{analysis_id}_annotated.mp4"

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Unable to save uploaded video: {error}")

    print(f"VIDEO SAVED: {video_path} | SIZE: {video_path.stat().st_size} bytes", flush=True)

    # Initialize job state
    jobs_db[analysis_id] = {
        "status": "queued",
        "progress": 0,
        "arm": arm,
        "video_path": str(video_path),
        "json_path": str(json_path),
        "annotated_path": str(annotated_path)
    }

    # Push heavy processing to background (returns instantly to frontend)
    background_tasks.add_task(process_analysis_pipeline, analysis_id)

    return {
        "status": "queued",
        "analysisId": analysis_id,
        "message": "Analysis started. Poll /result/{analysisId} for status and progress."
    }

async def process_analysis_pipeline(analysis_id: str):
    """Runs Phase 1 and Annotation in the background without blocking the server."""
    job = jobs_db[analysis_id]
    
    try:
        job["status"] = "processing"
        job["progress"] = 10

        # ========================================================
        # PHASE 1 ANALYSIS
        # ========================================================
        print("=" * 60, flush=True)
        print(f"STARTING PHASE 1 for {analysis_id}", flush=True)
        print("=" * 60, flush=True)

        phase1_command = [
            sys.executable, str(PHASE1_ENGINE), job["video_path"],
            "--arm", job["arm"], "--output", job["json_path"],
            "--model", str(MODEL_PATH),
        ]

        # Use asyncio to run subprocess without blocking the main event loop
        process = await asyncio.create_subprocess_exec(
            *phase1_command,
            cwd=str(BASE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception(f"Phase 1 failed: {stderr.decode()}")

        print(f"PHASE 1 COMPLETED for {analysis_id}", flush=True)
        job["progress"] = 60

        # ========================================================
        # ANNOTATED VIDEO
        # ========================================================
        print("=" * 60, flush=True)
        print(f"STARTING ANNOTATION for {analysis_id}", flush=True)
        print("=" * 60, flush=True)

        annotate_command = [
            sys.executable, str(ANNOTATOR), job["video_path"],
            "--arm", job["arm"], "--output", job["annotated_path"],
            "--model", str(MODEL_PATH),
        ]

        ann_process = await asyncio.create_subprocess_exec(
            *annotate_command,
            cwd=str(BASE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await ann_process.communicate()

        print(f"ANNOTATION COMPLETED for {analysis_id}", flush=True)
        job["progress"] = 100
        job["status"] = "completed"

    except Exception as error:
        print(f"ERROR in job {analysis_id}: {error}", flush=True)
        job["status"] = "failed"
        job["error"] = str(error)
        
    finally:
        # CRITICAL: Prevent memory leaks on Hugging Face / Render
        gc.collect()

# ============================================================
# GET RESULT / STATUS
# ============================================================
@app.get("/result/{analysis_id}")
async def get_result(analysis_id: str):
    # Check memory store first (for status/progress)
    if analysis_id in jobs_db:
        job = jobs_db[analysis_id]
        
        if job["status"] in ("queued", "processing"):
            return {
                "status": job["status"],
                "analysisId": analysis_id,
                "progress": job["progress"],
                "message": "Analysis is still running. Please wait."
            }
            
        if job["status"] == "failed":
            raise HTTPException(status_code=500, detail=job.get("error", "Analysis failed"))
            
        # If completed, read and return the JSON
        if job["status"] == "completed":
            try:
                with open(job["json_path"], "r", encoding="utf-8") as f:
                    result = json.load(f)
                    
                result["api"] = {
                    "analysisId": analysis_id,
                    "arm": job["arm"],
                    "video": f"/video/{analysis_id}",
                    "annotatedVideo": f"/video/{analysis_id}",
                    "result": f"/result/{analysis_id}",
                    "report": f"/report/{analysis_id}",
                }
                return result
            except Exception as error:
                raise HTTPException(status_code=500, detail=f"Unable to read result: {error}")

    # Fallback: If server restarted but files exist on disk
    json_path = RESULT_DIR / f"{analysis_id}_phase1_result.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
            
    raise HTTPException(status_code=404, detail="Analysis not found or still processing.")

# ============================================================
# GET ANNOTATED VIDEO
# ============================================================
@app.get("/video/{analysis_id}")
def get_annotated_video(analysis_id: str):
    video_path = ANNOTATED_DIR / f"{analysis_id}_annotated.mp4"
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found. Run analysis first.")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"cric_ai_{analysis_id}_analysis.mp4",
    )

# ============================================================
# GENERATE PDF REPORT
# ============================================================
@app.post("/report/{analysis_id}")
def generate_report(analysis_id: str):
    # Note: For PDF generation, we keep it synchronous as it's usually triggered manually after analysis
    import subprocess
    json_path = RESULT_DIR / f"{analysis_id}_phase1_result.json"

    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    report_name = f"{analysis_id}_report"
    command = [
        sys.executable, str(PHASE6_REPORT), str(json_path),
        "--out-name", report_name, "--out-dir", str(REPORT_DIR),
    ]

    try:
        process = subprocess.run(command, cwd=str(BASE_DIR), capture_output=True, text=True)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Unable to generate report: {error}")

    if process.returncode != 0:
        raise HTTPException(status_code=500, detail={"message": "Report generation failed.", "stderr": process.stderr})

    return {
        "status": "ok",
        "analysisId": analysis_id,
        "pdf": f"/report/{analysis_id}/pdf",
        "html": f"/report/{analysis_id}/html",
        "message": "Report generated successfully.",
    }

# ============================================================
# DOWNLOAD PDF
# ============================================================
@app.get("/report/{analysis_id}/pdf")
def download_pdf(analysis_id: str):
    pdf_path = REPORT_DIR / f"{analysis_id}_report.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF report not found. Generate the report first.")

    return FileResponse(path=str(pdf_path), media_type="application/pdf", filename=f"cric_ai_report_{analysis_id}.pdf")

# ============================================================
# VIEW HTML REPORT
# ============================================================
@app.get("/report/{analysis_id}/html")
def get_html_report(analysis_id: str):
    html_path = REPORT_DIR / f"{analysis_id}_report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="HTML report not found. Generate the report first.")

    return FileResponse(path=str(html_path), media_type="text/html")
