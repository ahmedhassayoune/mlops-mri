"""
DAG: DICOM → NIfTI conversion pipeline with dynamic task mapping.

Discovers every patient ID under raw/train/ in S3, then processes each
patient independently:  check → download → convert → upload → cleanup.

Already-processed patients are skipped automatically.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess

import boto3
from airflow.sdk import dag, task
from airflow.exceptions import AirflowSkipException
from pendulum import datetime

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────
S3_BUCKET = "s3-mlops-mri-ahassayoune"
RAW_PREFIX = "raw/train"
PROCESSED_PREFIX = "processed/train"
LOCAL_RAW = "/tmp/mri_raw"
LOCAL_PROCESSED = "/tmp/mri_processed"
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

# Tune based on worker resources / S3 rate limits
MAX_PARALLEL_PATIENTS = 4


# ── Helper functions (pure logic, no Airflow deps) ────
def _check_if_processed(patient_id: str) -> bool:
    """Return True if all modalities already exist in S3."""
    s3 = boto3.client("s3")
    for modality in MODALITIES:
        resp = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=f"{PROCESSED_PREFIX}/{patient_id}/{modality}/",
            MaxKeys=1,
        )
        if resp.get("KeyCount", 0) == 0:
            log.info("Patient %s needs processing (missing %s)", patient_id, modality)
            return False
    return True


def _download_from_s3(patient_id: str) -> None:
    """Download all DICOM files for one patient."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for modality in MODALITIES:
        local_dir = os.path.join(LOCAL_RAW, patient_id, modality)
        os.makedirs(local_dir, exist_ok=True)

        for page in paginator.paginate(
            Bucket=S3_BUCKET,
            Prefix=f"{RAW_PREFIX}/{patient_id}/{modality}/",
        ):
            for obj in page.get("Contents", []):
                filename = obj["Key"].rsplit("/", 1)[-1]
                s3.download_file(
                    S3_BUCKET,
                    obj["Key"],
                    os.path.join(local_dir, filename),
                )

    log.info("Downloaded patient %s from S3", patient_id)


def _convert_dicom_to_nifti(patient_id: str) -> None:
    """Run dcm2niix on each modality for one patient."""
    for modality in MODALITIES:
        input_dir = os.path.join(LOCAL_RAW, patient_id, modality)
        output_dir = os.path.join(LOCAL_PROCESSED, patient_id, modality)
        os.makedirs(output_dir, exist_ok=True)

        result = subprocess.run(
            ["dcm2niix", "-z", "y", "-o", output_dir, input_dir],
            capture_output=True,
            text=True,
            check=False,
        )
        log.info(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(
                f"dcm2niix failed for {patient_id}/{modality}: {result.stderr}"
            )

    log.info("Converted patient %s to NIfTI", patient_id)


def _upload_to_s3(patient_id: str) -> None:
    """Upload every NIfTI produced for one patient."""
    s3 = boto3.client("s3")
    patient_root = os.path.join(LOCAL_PROCESSED, patient_id)

    for root, _dirs, files in os.walk(patient_root):
        for fname in files:
            local_path = os.path.join(root, fname)
            relative = os.path.relpath(local_path, LOCAL_PROCESSED)
            s3_key = f"{PROCESSED_PREFIX}/{relative}"
            s3.upload_file(local_path, S3_BUCKET, s3_key)
            log.info("Uploaded → s3://%s/%s", S3_BUCKET, s3_key)

    log.info("Uploaded patient %s to S3", patient_id)


def _cleanup(patient_id: str) -> None:
    """Remove raw & processed temp dirs for one patient."""
    for base in (LOCAL_RAW, LOCAL_PROCESSED):
        path = os.path.join(base, patient_id)
        if os.path.isdir(path):
            shutil.rmtree(path)
            log.info("Cleaned up %s", path)


@dag(
    dag_id="dicom_to_nifti_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mri", "dicom", "processing"],
    max_active_tasks=MAX_PARALLEL_PATIENTS,
)
def dicom_to_nifti_pipeline():
    """DICOM → NIfTI for every patient in the S3 dataset."""

    @task
    def discover_patients() -> list[str]:
        """List every patient-id folder under raw/train/ in S3."""
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        patient_ids: set[str] = set()

        for page in paginator.paginate(
            Bucket=S3_BUCKET, Prefix=f"{RAW_PREFIX}/", Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                patient_id = cp["Prefix"].rstrip("/").split("/")[-1]
                patient_ids.add(patient_id)

        result = sorted(patient_ids)
        log.info("Discovered %d patients: %s", len(result), result)
        return result[:4]

    @task
    def process_patient(patient_id: str) -> str:
        """Full per-patient pipeline: check → download → convert → upload.
        Cleanup always runs, even on failure."""
        if _check_if_processed(patient_id):
            raise AirflowSkipException(
                f"Patient {patient_id} already fully processed — skipping"
            )

        try:
            _download_from_s3(patient_id)
            _convert_dicom_to_nifti(patient_id)
            _upload_to_s3(patient_id)
        finally:
            _cleanup(patient_id)

        log.info("Patient %s processed successfully", patient_id)
        return patient_id

    # ── DAG wiring with dynamic task mapping ──────────
    ids = discover_patients()
    process_patient.expand(patient_id=ids)


dicom_to_nifti_pipeline()