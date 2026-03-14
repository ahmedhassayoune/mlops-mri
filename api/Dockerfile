FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

# Env defaults
ENV MLFLOW_URI="http://your-mlflow-ip:5000" \
    MODEL_URI="models:/mgmt-prediction/1" \
    DEVICE="cpu"

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]