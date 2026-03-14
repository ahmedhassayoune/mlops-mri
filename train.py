# train.py

import os
import boto3
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import pandas as pd

# ── Config ────────────────────────────────────────────
S3_BUCKET      = "s3-mlops-mri-ahassayoune"
PROCESSED_KEY  = "processed/train"
LABELS_KEY     = "raw/train_labels.csv"
LOCAL_DATA_DIR = "/tmp/nifti"
MLFLOW_URI     = "http://172.31.37.214:5000"  # ← paste your MLflow IP
MODALITY       = "FLAIR"   # start with one modality
BATCH_SIZE     = 16
EPOCHS         = 10
LR             = 1e-4
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────
class MRIDataset(Dataset):
    def __init__(self, patient_ids, labels, data_dir, transform=None):
        self.patient_ids = patient_ids
        self.labels      = labels
        self.data_dir    = data_dir
        self.transform   = transform

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label      = self.labels[idx]

        # Load NIfTI file
        nii_path = os.path.join(
            self.data_dir, patient_id, MODALITY, "image.nii.gz"
        )
        img = nib.load(nii_path).get_fdata()

        # Take middle slice as 2D input (most informative slice)
        mid_slice = img[:, :, img.shape[2] // 2]

        # Normalize to [0, 1]
        mid_slice = (mid_slice - mid_slice.min()) / (mid_slice.max() - mid_slice.min() + 1e-8)

        # Convert to 3-channel tensor (ResNet expects RGB)
        mid_slice = np.stack([mid_slice] * 3, axis=0).astype(np.float32)
        tensor    = torch.tensor(mid_slice)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.long)


# ── Download data from S3 ─────────────────────────────
def download_data(patient_ids):
    s3 = boto3.client("s3")
    for pid in patient_ids:
        local_dir = f"{LOCAL_DATA_DIR}/{pid}/{MODALITY}"
        os.makedirs(local_dir, exist_ok=True)

        prefix   = f"{PROCESSED_KEY}/{pid}/{MODALITY}/"
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

        for obj in response.get("Contents", []):
            if not obj["Key"].endswith(".nii.gz"):
                continue
            local_path = f"{local_dir}/image.nii.gz"
            s3.download_file(S3_BUCKET, obj["Key"], local_path)

    print(f"✅ Downloaded {len(patient_ids)} patients")


# ── Model ─────────────────────────────────────────────
def build_model():
    model = models.resnet18(pretrained=True)

    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for binary classification (MGMT: 0 or 1)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    return model.to(DEVICE)


# ── Training Loop ─────────────────────────────────────
def train():
    # Load labels
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, LABELS_KEY, "/tmp/train_labels.csv")
    df = pd.read_csv("/tmp/train_labels.csv")

    patient_ids = df["BraTS21ID"].astype(str).str.zfill(5).tolist()[:32]
    labels      = df["MGMT_value"].tolist()[:32]

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        patient_ids, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Download data
    download_data(X_train + X_val)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    train_ds = MRIDataset(X_train, y_train, LOCAL_DATA_DIR, transform)
    val_ds   = MRIDataset(X_val,   y_val,   LOCAL_DATA_DIR, transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    # ── MLflow run ────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("mgmt-prediction")

    with mlflow.start_run(run_name="resnet18-flair-baseline"):

        # Log hyperparameters
        mlflow.log_params({
            "model":      "resnet18",
            "modality":   MODALITY,
            "epochs":     EPOCHS,
            "lr":         LR,
            "batch_size": BATCH_SIZE,
            "device":     DEVICE,
        })

        for epoch in range(EPOCHS):
            # ── Train ─────────────────────────────────
            model.train()
            train_loss, train_correct = 0, 0

            for X, y in train_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out  = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                train_loss    += loss.item()
                train_correct += (out.argmax(1) == y).sum().item()

            # ── Validate ──────────────────────────────
            model.eval()
            val_loss, val_correct = 0, 0

            with torch.no_grad():
                for X, y in val_dl:
                    X, y  = X.to(DEVICE), y.to(DEVICE)
                    out   = model(X)
                    loss  = criterion(out, y)
                    val_loss    += loss.item()
                    val_correct += (out.argmax(1) == y).sum().item()

            # ── Metrics ───────────────────────────────
            train_acc = train_correct / len(train_ds)
            val_acc   = val_correct   / len(val_ds)
            t_loss    = train_loss    / len(train_dl)
            v_loss    = val_loss      / len(val_dl)

            print(f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Train Loss: {t_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {v_loss:.4f} Acc: {val_acc:.4f}")

            # Log per-epoch metrics to MLflow
            mlflow.log_metrics({
                "train_loss": t_loss,
                "train_acc":  train_acc,
                "val_loss":   v_loss,
                "val_acc":    val_acc,
            }, step=epoch)

        # Log final model artifact to S3 via MLflow
        mlflow.pytorch.log_model(model, name="resnet18-flair")
        print("✅ Model logged to MLflow")


if __name__ == "__main__":
    train()