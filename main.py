import os
import uuid
import shutil
import yaml
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uvicorn

from app_main import SecureRedactionPolicyManager, SecureInfoRedactionPipeline

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Secure Redaction API",
    description="An API to securely redact PII and PHI from PDF documents.",
    version="1.0.0"
)

# --- CORS Configuration ---
# Allows the frontend (running on a different port) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Temporary Storage ---
# A simple in-memory dictionary to track session files.
# In a production environment, you might use a database or a more robust cache like Redis.
TEMP_STORAGE: Dict[str, Dict[str, str]] = {}
TEMP_BASE_DIR = "temp_processing_files"
os.makedirs(TEMP_BASE_DIR, exist_ok=True)


# --- API Endpoints ---

@app.post("/redact/")
async def redact_document(
        text_redaction_mode: str ="dummy_replacement",
        visual_redaction_mode: str = "image",
        create_overlay_pdf: bool = Form(...),
        file: UploadFile = File(...)
):
    """
    Accepts a PDF and redaction settings, processes the document,
    and returns paths to the generated files.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    # 1. Create a unique session directory for this request
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_BASE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_pdf_path = os.path.join(session_dir, file.filename)
    output_pdf_name = f"redacted_{file.filename}"
    output_pdf_path = os.path.join(session_dir, output_pdf_name)
    policy_file_path = os.path.join(session_dir, "redaction_policies.yaml")
    log_file_path = os.path.join(session_dir, "redaction_log.json")

    try:
        # 2. Save the uploaded file
        with open(input_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Create the dynamic policy file from form data
        policy_config = {
            'global_settings': {
                'text_redaction_mode': text_redaction_mode,
                'visual_redaction_mode': visual_redaction_mode,
                'create_overlay_pdf': create_overlay_pdf
            }
        }
        with open(policy_file_path, 'w') as f:
            yaml.dump(policy_config, f)

        # 4. Initialize and run the redaction pipeline
        policy_manager = SecureRedactionPolicyManager(policy_file_path)
        pipeline = SecureInfoRedactionPipeline(input_pdf_path, policy_manager)

        # The process method will create the output_pdf and potentially an overlay
        processing_log = pipeline.process(output_pdf_path)
        pipeline.save_processing_log(log_file_path)

        # 5. Store file paths for later download
        overlay_pdf_name = output_pdf_name.replace('.pdf', '_overlay.pdf')
        overlay_pdf_path = os.path.join(session_dir, overlay_pdf_name)

        TEMP_STORAGE[session_id] = {
            "redacted": output_pdf_name,
            "log": "redaction_log.json",
        }
        if create_overlay_pdf and os.path.exists(overlay_pdf_path):
            TEMP_STORAGE[session_id]["overlay"] = overlay_pdf_name

        return {
            "session_id": session_id,
            "message": "Redaction processing complete.",
            "files": TEMP_STORAGE[session_id],
            "log_summary": processing_log['metrics']
        }

    except Exception as e:
        # Clean up the session directory on failure
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")


@app.get("/download/{session_id}/{file_key}")
async def download_file(session_id: str, file_key: str):
    """
    Downloads a specific file generated during a redaction session.
    """
    if session_id not in TEMP_STORAGE or file_key not in TEMP_STORAGE[session_id]:
        raise HTTPException(status_code=404, detail="File not found or session expired.")

    file_name = TEMP_STORAGE[session_id][file_key]
    file_path = os.path.join(TEMP_BASE_DIR, session_id, file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server.")

    return FileResponse(path=file_path, filename=file_name, media_type='application/octet-stream')


