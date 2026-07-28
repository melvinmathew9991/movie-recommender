# Dockerfile -- containerizes the FastAPI recommendation service.
#
# This is genuine Docker infrastructure (build + run a real image), distinct
# from the Dev Containers used in MEDBOT (which are a VS Code development
# convenience, not production containerization). Worth keeping that
# distinction clear rather than treating them as equivalent.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/

# PREREQUISITE: both paths below are gitignored regenerable artifacts, so they
# are absent from a fresh clone and these COPY lines will fail. Run
#   python src/generate_sample_data.py && python src/train.py
# before building. CI does exactly this in the docker-build job.
COPY models/ models/
COPY data/raw/movies.csv data/raw/movies.csv

EXPOSE 8000

# Healthcheck hits the same /health endpoint the API exposes for monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
