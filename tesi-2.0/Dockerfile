FROM python:3.11-slim

LABEL description="Tesi_2.0 — EEG Brain-Age / Sex classification pipeline"
LABEL maintainer="Samuele"

# System deps per MNE + scipy + networkx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia prima requirements per cache layer
COPY requirements_docker.txt .

RUN pip install --no-cache-dir -r requirements_docker.txt

# Copia codebase (escludi dati pesanti via .dockerignore)
COPY . .

# fsaverage MNE data (scaricata in build da MNE auto-download)
ENV MNE_DATA=/mne_data
RUN python3 -c "import mne; mne.datasets.fetch_fsaverage(verbose=False)" 2>/dev/null || true

# Mount points (da usare con -v)
VOLUME ["/app/data", "/mne_data"]

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-c", "print('Tesi_2.0 container ready. Run: python3 scripts/run_ml_age_sex_n179.py --dry-run')"]
