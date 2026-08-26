import json, uuid, shutil, asyncio, gc, sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Cric AI API", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "api_uploads"
RESULT_DIR = BASE_DIR / "api_results"
IMAGE_DIR = BASE_DIR / "api_images" # Changed from ANNOTATED_DIR

for d in [UPLOAD_DIR, RESULT_DIR, IMAGE_DIR]:
    d.mkdir(exist_ok=True)

jobs = {}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...), arm: str = "right"):
    aid = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix
    v_path = UPLOAD_DIR / f"{aid}{ext}"
    j_path = RESULT_DIR / f"{aid}.json"
    i_path = IMAGE_DIR / f"{aid}_release.jpg" # Changed to .jpg
    
    with open(v_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    jobs[aid] = {"status": "processing", "progress": 10}
    background_tasks.add_task(run_pipeline, aid, str(v_path), str(j_path), str(i_path), arm)
    
    return {"analysisId": aid, "status": "queued"}

async def run_pipeline(aid, v_path, j_path, i_path, arm):
    try:
        # 1. Run Phase 1 (Biomechanics)
        p1 = await asyncio.create_subprocess_exec(
            sys.executable, "phase1_engine.py", v_path, "--arm", arm, "--output", j_path, "--model", "yolov8n-pose.pt"
        )
        await p1.communicate()
        jobs[aid]["progress"] = 60
        
        # 2. Run Release Image Generation (Ultra-lightweight)
        p2 = await asyncio.create_subprocess_exec(
            sys.executable, "draw_release_image.py", v_path, j_path, "--output", i_path, "--arm", arm
        )
        await p2.communicate()
        
        jobs[aid] = {"status": "completed", "progress": 100}
    except Exception as e:
        jobs[aid] = {"status": "failed", "error": str(e)}
    finally:
        gc.collect()

@app.get("/result/{aid}")
def get_result(aid: str):
    j_path = RESULT_DIR / f"{aid}.json"
    if j_path.exists():
        with open(j_path) as f:
            data = json.load(f)
            data["api"] = {"analysisId": aid, "releaseImage": f"/image/{aid}"}
            return data
    return jobs.get(aid, {"status": "not_found"})

@app.get("/image/{aid}") # Changed from /video/
def get_release_image(aid: str):
    i_path = IMAGE_DIR / f"{aid}_release.jpg"
    if i_path.exists():
        return FileResponse(str(i_path), media_type="image/jpeg", filename=f"release_{aid}.jpg")
    raise HTTPException(404, "Image not ready yet")
