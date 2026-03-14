# api/main.py

import os
import io
import boto3
import mlflow
import mlflow.pytorch
import nibabel as nib
import numpy as np
import torch
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from contextlib import asynccontextmanager

# ── Config ────────────────────────────────────────────
MLFLOW_URI    = os.getenv("MLFLOW_URI",    "http://your-mlflow-ip:5000")
MODEL_URI     = os.getenv("MODEL_URI",     "models:/mgmt-prediction/1")
DEVICE        = os.getenv("DEVICE",        "cpu")   # CPU is fine for inference

app = FastAPI(
    title="MGMT Prediction API",
    description="Predicts MGMT promoter methylation status from MRI NIfTI files",
    version="1.0.0"
)

# ── Load model at startup ──────────────────────────────
# Runs once when the container starts — not on every request
model = None


# on_event is deprecated, use lifespan event handlers instead.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Loading model from MLflow: {MODEL_URI}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.pytorch.load_model(MODEL_URI, map_location=DEVICE)
    model.eval()
    print("✅ Model loaded")
    yield
    print("Shutting down API...")


# ── Preprocessing ──────────────────────────────────────
def preprocess_nifti(nifti_bytes: bytes) -> torch.Tensor:
    # Write bytes to a temp file (nibabel needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(nifti_bytes)
        tmp_path = f.name

    img = nib.load(tmp_path).get_fdata()
    os.unlink(tmp_path)  # cleanup

    # Middle slice
    mid_slice = img[:, :, img.shape[2] // 2]

    # Normalize
    mid_slice = (mid_slice - mid_slice.min()) / \
                (mid_slice.max() - mid_slice.min() + 1e-8)

    # 3-channel tensor for ResNet
    mid_slice = np.stack([mid_slice] * 3, axis=0).astype(np.float32)

    tensor = torch.tensor(mid_slice).unsqueeze(0)  # add batch dim → (1, 3, H, W)
    tensor = transforms.functional.resize(tensor, [224, 224])
    return tensor.to(DEVICE)


# ── Routes ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": DEVICE
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii or .nii.gz files accepted"
        )

    # Read uploaded file
    contents = await file.read()

    try:
        tensor = preprocess_nifti(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {str(e)}")

    # Inference
    with torch.no_grad():
        logits      = model(tensor)
        probs       = torch.softmax(logits, dim=1)
        pred_class  = torch.argmax(probs, dim=1).item()
        confidence  = probs[0][pred_class].item()

    return {
        "mgmt_methylated":  bool(pred_class == 1),
        "predicted_class":  pred_class,
        "confidence":       round(confidence, 4),
        "probabilities": {
            "not_methylated": round(probs[0][0].item(), 4),
            "methylated":     round(probs[0][1].item(), 4),
        }
    }