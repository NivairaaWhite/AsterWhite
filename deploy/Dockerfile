# AsterWhite serving image.
#
# CPU (default):
#   docker build -f deploy/Dockerfile -t asterwhite:cpu .
#
# GPU (CUDA): override BASE_IMAGE with a matching torch build in
# requirements.txt (the pinned torch>=2.2 wheel from PyPI is CPU-only;
# for CUDA, either install a CUDA-enabled torch wheel via an extra
# --index-url in a modified requirements install step, or start FROM a
# pytorch/pytorch:*-cuda*-runtime base that already bundles one):
#   docker build -f deploy/Dockerfile --build-arg BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime -t asterwhite:gpu .
#
# This Dockerfile is authored and buildable, but not verified against a
# live Kubernetes cluster or real GPU hardware in this repository's
# CPU-only development environment -- see README.md's scope notes.

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# Build tooling only where the base image doesn't already provide it;
# harmless no-op on images that do (e.g. pytorch/pytorch bases).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 asterwhite \
    && chown -R asterwhite:asterwhite /app
USER asterwhite

# ASTERWHITE_SERVE_CHECKPOINT_DIR is deliberately NOT set: unset means "serve the bundled
# pretrained model" (pretrained/asterwhite-small, copied in by `COPY . .` above). To serve your own
# model instead, set it to a checkpoint/bundle directory (-e ASTERWHITE_SERVE_CHECKPOINT_DIR=/data/model
# plus a volume); its stored config supplies the architecture, and a tokenizer/ folder inside it is
# found automatically. (Setting it to the EMPTY string means untrained random weights.)
# ASTERWHITE_SERVE_PROFILE is likewise unset: the bundle's own serving defaults apply. Set it
# (e.g. production_inference) only to override sampling defaults / concurrency / timeouts.
ENV ASTERWHITE_SERVE_DEVICE=auto \
    ASTERWHITE_API_KEYS="" \
    ASTERWHITE_JSON_LOGS=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

ENTRYPOINT ["python", "train.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
