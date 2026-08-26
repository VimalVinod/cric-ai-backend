"""
api.py - Cric AI FastAPI Backend (Render Optimized)
"""
import json
import uuid
import shutil
import asyncio
import gc
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PATHS & CONFIG
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

for dir_path in [UPLOAD_DIR, RESULT_DIR, REPORT_DIR, ANNOTATED_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================
# APP & SECURITY
# ============================================================
app = FastAPI(title="Cric AI API", version="1.3.0")

# Rate Limiter Setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait before analyzing again."}
    )

# CORS Setup
ALLOWED_ORIGINS = ["https://cricai-amber.vercel.app", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs_db: Dict[str, Any] = {}

# ============================================================
# ROOT & HEALTH
# ============================================================
@app.get("/")
def root():
    return {"status": "ok", "service": "Cric AI API", "version": "1.3.0"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "phase1_engine": PHASE1_ENGINE.exists(),
        "annotator": ANNOTATOR.exists(),
        "pose_model": MODEL_PATH.exists(),
    }

# ============================================================
# ANALYZE VIDEO (Background Task + Rate Limited)
# ============================================================
@app.post("/analyze")
@limiter.limit("3/minute")
async def analyze_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    arm: str = "right",
):
    print(f"[ANALYZE] Received: {file.filename}, Arm: {arm}", flush=True)

    arm = arm.lower().strip()
    if arm not in ("right", "left"):
        raise HTTPException(status_code=400, detail="arm must be 'right' or 'left'")

    allowed_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Unsupported format.")

    analysis_id = uuid.uuid4().hex[:12]
    video_path = UPLOAD_DIR / f"{analysis_id}{ext}"
    json_path = RESULT_DIR / f"{analysis_id}_phase1_result.json"
    annotated_path = ANNOTATED_DIR / f"{analysis_id}_annotated.mp4"

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {e}")

    jobs_db[analysis_id] = {
        "status": "queued", "progress": 0, "arm": arm,
        "video_path": str(video_path), "json_path": str(json_path),
        "annotated_path": str(annotated_path)
    }

    background_tasks.add_task(process_pipeline, analysis_id)
    return {"status": "queued", "analysisId": analysis_id}

async def process_pipeline(aid: str):
    job = jobs_db[aid]
    try:
        job["status"] = "processing"; job["progress"] = 10
        
        # Phase 1
        cmd1 = [sys.executable, str(PHASE1_ENGINE), job["video_path"], 
                "--arm", job["arm"], "--output", job["json_path"], "--model", str(MODEL_PATH)]
        p1 = await asyncio.create_subprocess_exec(*cmd1, cwd=str(BASE_DIR), 
                                                  stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await p1.communicate()
        if p1.returncode != 0: raise Exception(f"Phase1 fail: {err.decode()}")
        
        job["progress"] = 60
        
        # Annotation
        cmd2 = [sys.executable, str(ANNOTATOR), job["video_path"], 
                "--arm", job["arm"], "--output", job["annotated_path"], "--model", str(MODEL_PATH)]
        p2 = await asyncio.create_subprocess_exec(*cmd2, cwd=str(BASE_DIR), 
                                                  stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await p2.communicate()
        
        job["progress"] = 100; job["status"] = "completed"
        print(f"[DONE] {aid}", flush=True)
        
    except Exception as e:
        print(f"[ERROR] {aid}: {e}", flush=True)
        job["status"] = "failed"; job["error"] = str(e)
    finally:
        gc.collect()

# ============================================================
# RESULTS & MEDIA
# ============================================================
@app.get("/result/{aid}")
async def get_result(aid: str):
    # 1. Check disk first (most reliable for subprocess workflows)
    json_path = RESULT_DIR / f"{aid}_phase1_result.json"
    
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            
            result_data["api"] = {
                "analysisId": aid,
                "video": f"/video/{aid}",
                "annotatedVideo": f"/video/{aid}"
            }
            return result_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read result: {e}")
    
    # 2. Fallback: Check memory store (for status/progress)
    if aid in jobs_db:
        j = jobs_db[aid]
        if j["status"] == "failed":
            raise HTTPException(status_code=500, detail=j.get("error"))
        if j["status"] in ("queued", "processing"):
            return {"status": j["status"], "progress": j.get("progress", 0)}
    
    raise HTTPException(status_code=404, detail="Analysis result not found.")

@app.get("/video/{aid}")
def get_video(aid: str):
    vp = ANNOTATED_DIR / f"{aid}_annotated.mp4"
    if not vp.exists(): raise HTTPException(status_code=404, detail="Video not ready")
    return FileResponse(str(vp), media_type="video/mp4", filename=f"cric_{aid}.mp4")

@app.post("/report/{aid}")
def gen_report(aid: str):
    jp = RESULT_DIR / f"{aid}_phase1_result.json"
    if not jp.exists(): raise HTTPException(status_code=404, detail="No result")
    cmd = [sys.executable, str(PHASE6_REPORT), str(jp), "--out-name", f"{aid}_report", "--out-dir", str(REPORT_DIR)]
    p = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    if p.returncode != 0: raise HTTPException(status_code=500, detail=p.stderr)
    return {"status": "ok", "pdf": f"/report/{aid}/pdf"}

@app.get("/report/{aid}/pdf")
def dl_pdf(aid: str):
    pp = REPORT_DIR / f"{aid}_report.pdf"
    if not pp.exists(): raise HTTPException(status_code=404, detail="Generate report first")
    return FileResponse(str(pp), media_type="application/pdf")

@app.get("/report/{aid}/html")
def view_html(aid: str):
    hp = REPORT_DIR / f"{aid}_report.html"
    if not hp.exists(): raise HTTPException(status_code=404, detail="Generate report first")
    return FileResponse(str(hp), media_type="text/html")
