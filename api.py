"""
api.py

Cric AI - FastAPI Backend

Pipeline:

    Video Upload
         ↓
    Phase 1 Analysis
         ↓
    Annotated Video Generation
         ↓
    Phase 1 JSON
         ↓
    Phase 6 HTML/PDF Report
         ↓
    API Response

Endpoints:

    GET  /
    GET  /health

    POST /analyze
    GET  /result/{analysis_id}

    GET  /video/{analysis_id}

    POST /report/{analysis_id}
    GET  /report/{analysis_id}/pdf
    GET  /report/{analysis_id}/html
"""

import json
import uuid
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
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
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Cric AI API",
    description="AI-powered cricket bowling biomechanics analysis API",
    version="1.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cricai-amber.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Cric AI API",
        "version": "1.1.0",
        "message": "Cric AI bowling biomechanics API is running.",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

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
# ANALYZE VIDEO
# ============================================================

@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    arm: str = "right",
):
    """
    Upload bowling video.

    Pipeline:

        Upload
          ↓
        Phase 1
          ↓
        Annotated Video
          ↓
        JSON response
    """

    # ========================================================
    # DEBUG LOG - REQUEST RECEIVED
    # ========================================================

    print("=" * 60, flush=True)
    print("ANALYZE REQUEST RECEIVED", flush=True)
    print(f"Filename: {file.filename}", flush=True)
    print(f"Arm: {arm}", flush=True)
    print("=" * 60, flush=True)


    # ========================================================
    # Validate arm
    # ========================================================

    arm = arm.lower().strip()

    if arm not in ("right", "left"):

        raise HTTPException(
            status_code=400,
            detail="arm must be 'right' or 'left'",
        )


    # ========================================================
    # Validate file
    # ========================================================

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )


    allowed_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Use MP4, MOV, AVI, MKV or WEBM."
            ),
        )


    # ========================================================
    # Create analysis ID
    # ========================================================

    analysis_id = uuid.uuid4().hex[:12]


    # ========================================================
    # File paths
    # ========================================================

    video_path = (
        UPLOAD_DIR /
        f"{analysis_id}{extension}"
    )

    json_path = (
        RESULT_DIR /
        f"{analysis_id}_phase1_result.json"
    )

    annotated_path = (
        ANNOTATED_DIR /
        f"{analysis_id}_annotated.mp4"
    )


    # ========================================================
    # Save uploaded video
    # ========================================================

    try:

        with open(video_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

    except Exception as error:

        print(
            f"ERROR SAVING VIDEO: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Unable to save uploaded video: {error}",
        )


    # ========================================================
    # DEBUG LOG - VIDEO SAVED
    # ========================================================

    print(
        f"VIDEO SAVED: {video_path} | "
        f"SIZE: {video_path.stat().st_size} bytes",
        flush=True,
    )


    # ========================================================
    # PHASE 1 ANALYSIS
    # ========================================================

    phase1_command = [
        sys.executable,
        str(PHASE1_ENGINE),

        str(video_path),

        "--arm",
        arm,

        "--output",
        str(json_path),

        "--model",
        str(MODEL_PATH),
    ]


    # ========================================================
    # DEBUG LOG - START PHASE 1
    # ========================================================

    print("=" * 60, flush=True)
    print("STARTING PHASE 1", flush=True)
    print(f"Command: {' '.join(phase1_command)}", flush=True)
    print("=" * 60, flush=True)


    try:

        process = subprocess.run(
            phase1_command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )

    except Exception as error:

        print(
            f"ERROR STARTING PHASE 1: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Unable to start Phase 1: {error}",
        )


    # ========================================================
    # DEBUG LOG - PHASE 1 FINISHED
    # ========================================================

    print(
        "=" * 60,
        flush=True,
    )

    print(
        f"PHASE 1 FINISHED: RETURN CODE {process.returncode}",
        flush=True,
    )

    print(
        f"PHASE 1 STDOUT LENGTH: {len(process.stdout)}",
        flush=True,
    )

    print(
        f"PHASE 1 STDERR LENGTH: {len(process.stderr)}",
        flush=True,
    )

    print(
        "=" * 60,
        flush=True,
    )


    # ========================================================
    # Phase 1 failure
    # ========================================================

    if process.returncode != 0:

        print("PHASE 1 STDOUT", flush=True)
        print(process.stdout, flush=True)

        print("PHASE 1 STDERR", flush=True)
        print(process.stderr, flush=True)

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Phase 1 analysis failed.",
                "stdout": process.stdout,
                "stderr": process.stderr,
            },
        )


    # ========================================================
    # Check JSON
    # ========================================================

    if not json_path.exists():

        print(
            "PHASE 1 FINISHED BUT JSON WAS NOT GENERATED",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Phase 1 completed but "
                "no JSON result was generated."
            ),
        )


    print(
        f"PHASE 1 JSON CREATED: {json_path}",
        flush=True,
    )


    # ========================================================
    # Read Phase 1 result
    # ========================================================

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as f:

            result = json.load(f)

    except Exception as error:

        print(
            f"ERROR READING PHASE 1 JSON: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Unable to read Phase 1 JSON: {error}",
        )


    print(
        "PHASE 1 JSON READ SUCCESSFULLY",
        flush=True,
    )


    # ========================================================
    # ANNOTATED VIDEO
    # ========================================================

    print("=" * 60, flush=True)
    print("STARTING ANNOTATED VIDEO GENERATION", flush=True)
    print("=" * 60, flush=True)

    annotate_command = [
        sys.executable,
        str(ANNOTATOR),

        str(video_path),

        "--arm",
        arm,

        "--output",
        str(annotated_path),

        "--model",
        str(MODEL_PATH),
    ]


    print(
        f"Annotation command: {' '.join(annotate_command)}",
        flush=True,
    )


    try:

        annotation_process = subprocess.run(
            annotate_command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )

    except Exception as error:

        print(
            f"ERROR STARTING ANNOTATION: {error}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Unable to start video annotation: {error}",
        )


    # ========================================================
    # DEBUG LOG - ANNOTATION FINISHED
    # ========================================================

    print("=" * 60, flush=True)

    print(
        f"ANNOTATION FINISHED: RETURN CODE "
        f"{annotation_process.returncode}",
        flush=True,
    )

    print(
        f"ANNOTATION STDOUT LENGTH: "
        f"{len(annotation_process.stdout)}",
        flush=True,
    )

    print(
        f"ANNOTATION STDERR LENGTH: "
        f"{len(annotation_process.stderr)}",
        flush=True,
    )

    print("=" * 60, flush=True)


    # ========================================================
    # Annotation failure
    # ========================================================

    if annotation_process.returncode != 0:

        print(
            "ANNOTATION STDOUT",
            flush=True,
        )

        print(
            annotation_process.stdout,
            flush=True,
        )

        print(
            "ANNOTATION STDERR",
            flush=True,
        )

        print(
            annotation_process.stderr,
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Annotated video generation failed.",
                "stdout": annotation_process.stdout,
                "stderr": annotation_process.stderr,
            },
        )


    # ========================================================
    # Check annotated video
    # ========================================================

    if not annotated_path.exists():

        print(
            "ANNOTATION FINISHED BUT VIDEO WAS NOT GENERATED",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Annotation completed but "
                "annotated video was not generated."
            ),
        )


    print(
        f"ANNOTATED VIDEO CREATED: {annotated_path}",
        flush=True,
    )

    print(
        f"ANNOTATED VIDEO SIZE: "
        f"{annotated_path.stat().st_size} bytes",
        flush=True,
    )


    # ========================================================
    # Add API metadata
    # ========================================================

    result["api"] = {

        "analysisId": analysis_id,

        "originalFilename": file.filename,

        "arm": arm,

        "video": f"/video/{analysis_id}",

        "annotatedVideo": f"/video/{analysis_id}",

        "result": f"/result/{analysis_id}",

        "report": f"/report/{analysis_id}",

        "pdf": f"/report/{analysis_id}/pdf",

        "html": f"/report/{analysis_id}/html",
    }


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    print("=" * 60, flush=True)
    print("ANALYSIS COMPLETED SUCCESSFULLY", flush=True)
    print(f"Analysis ID: {analysis_id}", flush=True)
    print("=" * 60, flush=True)

    return {

        "status": "ok",

        "analysisId": analysis_id,

        "video": f"/video/{analysis_id}",

        "annotatedVideo": f"/video/{analysis_id}",

        "result": result,

        "report": {

            "generate": f"/report/{analysis_id}",

            "pdf": f"/report/{analysis_id}/pdf",

            "html": f"/report/{analysis_id}/html",
        },

        "message": "Cric AI analysis completed successfully.",
    }


# ============================================================
# GET RESULT
# ============================================================

@app.get("/result/{analysis_id}")
def get_result(analysis_id: str):

    json_path = (
        RESULT_DIR /
        f"{analysis_id}_phase1_result.json"
    )

    if not json_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Analysis result not found.",
        )


    try:

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as f:

            result = json.load(f)

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to read result: {error}",
        )


    return result


# ============================================================
# GET ANNOTATED VIDEO
# ============================================================

@app.get("/video/{analysis_id}")
def get_annotated_video(analysis_id: str):

    video_path = (
        ANNOTATED_DIR /
        f"{analysis_id}_annotated.mp4"
    )

    if not video_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Annotated video not found. "
                "Run analysis first."
            ),
        )


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

    json_path = (
        RESULT_DIR /
        f"{analysis_id}_phase1_result.json"
    )


    if not json_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Analysis result not found.",
        )


    report_name = f"{analysis_id}_report"

    report_dir = REPORT_DIR


    command = [

        sys.executable,

        str(PHASE6_REPORT),

        str(json_path),

        "--out-name",
        report_name,

        "--out-dir",
        str(report_dir),
    ]


    try:

        process = subprocess.run(

            command,

            cwd=str(BASE_DIR),

            capture_output=True,

            text=True,
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to generate report: {error}",
        )


    if process.returncode != 0:

        raise HTTPException(

            status_code=500,

            detail={

                "message": "Report generation failed.",

                "stdout": process.stdout,

                "stderr": process.stderr,
            },
        )


    pdf_path = (
        REPORT_DIR /
        f"{report_name}.pdf"
    )

    html_path = (
        REPORT_DIR /
        f"{report_name}.html"
    )


    if not pdf_path.exists():

        raise HTTPException(

            status_code=500,

            detail=(
                "Phase 6 completed but "
                "PDF was not generated."
            ),
        )


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

    pdf_path = (
        REPORT_DIR /
        f"{analysis_id}_report.pdf"
    )


    if not pdf_path.exists():

        raise HTTPException(

            status_code=404,

            detail=(
                "PDF report not found. "
                "Generate the report first."
            ),
        )


    return FileResponse(

        path=str(pdf_path),

        media_type="application/pdf",

        filename=f"cric_ai_report_{analysis_id}.pdf",
    )


# ============================================================
# VIEW HTML REPORT
# ============================================================

@app.get("/report/{analysis_id}/html")
def get_html_report(analysis_id: str):

    html_path = (
        REPORT_DIR /
        f"{analysis_id}_report.html"
    )


    if not html_path.exists():

        raise HTTPException(

            status_code=404,

            detail=(
                "HTML report not found. "
                "Generate the report first."
            ),
        )


    return FileResponse(

        path=str(html_path),

        media_type="text/html",
    )
