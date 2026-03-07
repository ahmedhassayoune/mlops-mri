from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import boto3
import os

# ── Config ────────────────────────────────────────────
S3_BUCKET = "s3-mlops-mri-ahassayoune"
RAW_PREFIX = "raw/train"
PROCESSED_PREFIX = "processed/train"
LOCAL_RAW = "/tmp/mri_raw"
LOCAL_PROCESSED = "/tmp/mri_processed"

# ── Task 1: Download a sample from S3 ─────────────────
def download_from_s3(**context):
    s3 = boto3.client("s3")
    os.makedirs(LOCAL_RAW, exist_ok=True)

    # Start with one patient for testing
    patient_id = "00000"
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    for modality in modalities:
        prefix = f"{RAW_PREFIX}/{patient_id}/{modality}/"
        local_dir = f"{LOCAL_RAW}/{patient_id}/{modality}"
        os.makedirs(local_dir, exist_ok=True)

        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        for obj in response.get("Contents", []):
            filename = obj["Key"].split("/")[-1]
            s3.download_file(S3_BUCKET, obj["Key"], f"{local_dir}/{filename}")

    print(f"✅ Downloaded patient {patient_id} from S3")

# ── Task 2: Convert DICOM → NIfTI ─────────────────────
def convert_dicom_to_nifti(**context):
    patient_id = "00000"
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    for modality in modalities:
        input_dir = f"{LOCAL_RAW}/{patient_id}/{modality}"
        output_dir = f"{LOCAL_PROCESSED}/{patient_id}/{modality}"
        os.makedirs(output_dir, exist_ok=True)

        # dcm2niix does the heavy lifting
        result = subprocess.run([
            "dcm2niix",
            "-z", "y",          # gzip output → .nii.gz
            "-o", output_dir,   # output directory
            input_dir           # input DICOM directory
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.returncode != 0:
            raise Exception(f"dcm2niix failed: {result.stderr}")

    print(f"✅ Converted patient {patient_id} to NIfTI")

# ── Task 3: Upload NIfTI files back to S3 ─────────────
def upload_to_s3(**context):
    s3 = boto3.client("s3")
    patient_id = "00000"

    for root, dirs, files in os.walk(f"{LOCAL_PROCESSED}/{patient_id}"):
        for file in files:
            local_path = os.path.join(root, file)
            # Reconstruct the S3 key
            relative_path = local_path.replace(f"{LOCAL_PROCESSED}/", "")
            s3_key = f"{PROCESSED_PREFIX}/{relative_path}"

            s3.upload_file(local_path, S3_BUCKET, s3_key)
            print(f"  uploaded → s3://{S3_BUCKET}/{s3_key}")

    print(f"✅ Uploaded processed files to S3")

# ── DAG Definition ─────────────────────────────────────
with DAG(
    dag_id="dicom_to_nifti_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,          # manual trigger for now
    catchup=False,
    tags=["mri", "dicom", "processing"],
) as dag:

    t1 = PythonOperator(
        task_id="download_from_s3",
        python_callable=download_from_s3,
    )

    t2 = PythonOperator(
        task_id="convert_dicom_to_nifti",
        python_callable=convert_dicom_to_nifti,
    )

    t3 = PythonOperator(
        task_id="upload_to_s3",
        python_callable=upload_to_s3,
    )

    # Define execution order
    t1 >> t2 >> t3