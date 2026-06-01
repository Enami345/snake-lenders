FROM python:3.11-slim

WORKDIR /app

# System deps for numpy/torch (no GUI needed — pygame excluded)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

# HuggingFace Spaces uses port 7860 by default
ENV PORT=7860

EXPOSE 7860

CMD ["python", "server.py"]
