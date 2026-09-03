"""
main.py

FastAPI backend for SatQuery AI. This is the entry point Railway will run.

Flow for a single-image question:
  1. User uploads image + question
  2. router.py decides the task type
  3. specialist model (rs_inference.py) analyzes the image
  4. gemini_client.py turns the structured result into a natural answer
  5. audit_log.py records every step

Run locally with:
    uvicorn backend.main:app --reload

Railway runs it via the Procfile (see repo root).
"""

import os
import sys
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# allow importing sibling packages (agent/, inference/) when run from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download

from agent.router import classify_task, TaskType
from agent.gemini_client import ask_gemini
from agent.audit_log import log_step, read_trail
from inference.rs_inference import RSClassifier

app = FastAPI(title="SatQuery AI")

# Your model lives on Hugging Face -- Railway needs to pull it at startup
# since it deploys from your GitHub repo, not your laptop's local files.
HF_REPO_ID = os.environ.get("HF_REPO_ID", "KowhickMaran/rs-eurosat-classifier")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "rs-eurosat-classifier")


def ensure_model_downloaded():
    checkpoint_path = os.path.join(MODEL_DIR, "rs_classifier.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Model not found locally -- downloading {HF_REPO_ID} from Hugging Face...")
        snapshot_download(repo_id=HF_REPO_ID, local_dir=MODEL_DIR)
    return checkpoint_path

# loosen CORS for the hackathon demo frontend; tighten before real deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_classifier = None  # lazy-loaded so the app can boot even before the model file exists


def get_classifier():
    global _classifier
    if _classifier is None:
        checkpoint_path = ensure_model_downloaded()
        _classifier = RSClassifier(checkpoint_path)
    return _classifier


@app.get("/")
def health_check():
    return {"status": "ok", "service": "SatQuery AI"}


@app.get("/audit-trail")
def get_audit_trail():
    return {"entries": read_trail()}


@app.post("/analyze")
async def analyze(question: str = Form(...), image: UploadFile = File(...)):
    """
    Main endpoint: single-image VQA path.
    Change-detection / fusion endpoints will be added once those pieces exist.
    """
    task = classify_task(question, num_images=1)

    if task != TaskType.SINGLE_IMAGE_VQA:
        raise HTTPException(
            status_code=501,
            detail=f"Task type '{task.value}' isn't wired up yet -- only single-image VQA works so far.",
        )

    # save upload to a temp file so PIL/rasterio can open it
    suffix = os.path.splitext(image.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        classifier = get_classifier()
        specialist_result = classifier.predict(tmp_path)
        log_step("specialist_model_call", specialist_result)

        answer = ask_gemini(question, specialist_result)
        log_step("gemini_call", {"question": question, "answer": answer})

        return {
            "answer": answer,
            "specialist_result": specialist_result,
            "task_type": task.value,
        }
    finally:
        os.remove(tmp_path)
