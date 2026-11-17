FROM python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce
COPY --from=ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/dice/src PATH=/opt/venv/bin:$PATH \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy \
    BENTOML_HOME=/state/bentoml BENTOML_DO_NOT_TRACK=true \
    YOLO_CONFIG_DIR=/state/ultralytics YOLO_AUTOINSTALL=false \
    MPLCONFIGDIR=/state/matplotlib DICE_PROJECT_ROOT=/opt/dice \
    DICE_STATE_DIR=/state DICE_DEVICE=cpu OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/v1/environment/ /opt/environment/
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then export SSL_CERT_FILE=/run/secrets/build_ca; fi; \
    uv sync --project /opt/environment --locked --no-install-project --no-dev --system-certs

COPY . /opt/dice/
ARG DICE_BUNDLE_SHA256
ARG DICE_RELEASE_ID
ENV DICE_BUNDLE_SHA256=${DICE_BUNDLE_SHA256}
LABEL io.dice-viewer.release=${DICE_RELEASE_ID} io.dice-viewer.bundle-sha256=${DICE_BUNDLE_SHA256}
RUN python -m dice_viewer.recognition.operations verify \
    && groupadd --gid 10001 dice && useradd --uid 10001 --gid 10001 --no-create-home dice \
    && mkdir -p /state && chown 10001:10001 /state
USER 10001:10001
WORKDIR /opt/dice
EXPOSE 3000
CMD ["python", "-m", "dice_viewer.recognition.operations", "serve"]
