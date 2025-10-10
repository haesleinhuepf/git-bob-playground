FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Optional: pre-install if requirements exist to speed up caches
COPY requirements*.txt ./ 
RUN python -m pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    pip install --no-cache-dir jupyterlab

COPY . .

EXPOSE 8888 8000 5000

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
