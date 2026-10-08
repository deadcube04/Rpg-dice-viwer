FROM python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH=/app/.venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    BENTOML_HOME=/opt/bentoml \
    BENTOML_DO_NOT_TRACK=true \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    DICE_STORAGE_BACKEND=local \
    DICE_SQLITE_PATH=/state/feedback.db \
    DICE_LOCAL_IMAGE_DIR=/state/images

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY service.py /app/service.py
COPY src/dice_viewer /app/src/dice_viewer
COPY artifacts-next/dice_bundle-e33njkwb56yiiaa2.bentomodel /tmp/dice_bundle.bentomodel
RUN python -m dice_viewer.deployment import-model /tmp/dice_bundle.bentomodel \
    && rm /tmp/dice_bundle.bentomodel \
    && groupadd --gid 1034 dice \
    && useradd --uid 1034 --gid dice --create-home dice \
    && mkdir -p /state/images /tmp/ultralytics \
    && chown -R dice:dice /state /opt/bentoml /tmp/ultralytics

USER 1034:1034
EXPOSE 3000
CMD ["python", "-m", "dice_viewer.deployment", "serve"]
