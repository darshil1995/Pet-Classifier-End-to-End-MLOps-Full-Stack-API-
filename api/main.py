from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import uuid
import aiofiles  # Add this for async file writing
from src.predict import Predictor

app = FastAPI(title="Pet Classifier UI")
templates = Jinja2Templates(directory="api/templates")

try:
    predictor = Predictor()
except Exception:
    predictor = None


class PredictionResponse(BaseModel):
    filename: str
    label: str
    confidence: float


# --- New Route for UI ---
@app.get("/", response_class=HTMLResponse)
async def render_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- Prediction Route ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    temp_path = f"temp_{uuid.uuid4()}.jpg"

    try:
        # Warning-free async file writing
        async with aiofiles.open(temp_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        label, confidence = predictor.predict(temp_path)
        return {"filename": file.filename, "label": label, "confidence": confidence}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)