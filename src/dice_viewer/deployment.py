"""Import the frozen release and diagnose the standalone GPU deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen

from dice_viewer.active_model import ACTIVE_MODEL_TAG

BUNDLE_SHA256 = "4d306b197cf4a2d245dfae23fd5c6993dc4d32eb59129a25a861de4a5308cd4d"


def model_folder() -> Path:
    import bentoml

    try:
        folder = Path(bentoml.models.get(ACTIVE_MODEL_TAG).path)
    except bentoml.exceptions.NotFound as exc:
        raise RuntimeError(
            f"Modelo ausente: {ACTIVE_MODEL_TAG}. Reconstrua com docker compose up -d --build."
        ) from exc
    for name in ("detector.pt", "reader.safetensors", "reader.json", "threshold.json"):
        if not (folder / name).is_file():
            raise RuntimeError(f"Bundle incompleto: falta {name}. Reconstrua a imagem.")
    return folder


def import_model(archive: Path) -> None:
    import bentoml

    if not archive.is_file():
        raise RuntimeError("Bundle ausente. Inclua o .bentomodel versionado no clone.")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != BUNDLE_SHA256:
        raise RuntimeError(f"SHA-256 do bundle divergente: esperado {BUNDLE_SHA256}; recebido {digest}.")
    model = bentoml.models.import_model(str(archive))
    if str(model.tag) != ACTIVE_MODEL_TAG:
        raise RuntimeError(f"Versão divergente no bundle: {model.tag}; esperado {ACTIVE_MODEL_TAG}.")
    model_folder()
    print(f"Bundle verificado e importado: {model.tag} (SHA-256 {digest})", flush=True)


def serve() -> None:
    import torch

    from dice_viewer.storage import FeedbackStore

    if os.environ.get("DICE_MODEL_TAG", ACTIVE_MODEL_TAG) != ACTIVE_MODEL_TAG:
        raise RuntimeError("DICE_MODEL_TAG difere do modelo empacotado. Use a configuração Compose da entrega.")
    model_folder()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU NVIDIA indisponível. Confira nvidia-smi, atualize o driver/WSL2 "
            "e habilite o backend WSL2 no Docker Desktop."
        )
    try:
        torch.zeros(1, device="cuda:0").sum().item()
    except RuntimeError as exc:
        raise RuntimeError("CUDA não conseguiu executar na GPU. Confira driver e compatibilidade da placa.") from exc
    try:
        store = FeedbackStore()
        store.ready()
    except Exception as exc:
        raise RuntimeError("Armazenamento indisponível. Confira as permissões e o espaço do volume /state.") from exc
    print(f"Modelo {ACTIVE_MODEL_TAG}; GPU {torch.cuda.get_device_name(0)}; armazenamento {store.backend} pronto.", flush=True)
    os.execvp("bentoml", ["bentoml", "serve", "service:DiceService", "--host", "0.0.0.0", "--port", "3000"])


def healthcheck() -> None:
    with urlopen("http://127.0.0.1:3000/health", timeout=8) as response:
        result = json.load(response)
    if result.get("status") != "ready" or result.get("model_version") != ACTIVE_MODEL_TAG:
        raise RuntimeError("Serviço não está pronto com o modelo esperado.")
    if result.get("storage_ready") is not True:
        raise RuntimeError("Armazenamento não está pronto.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import-model")
    importer.add_argument("archive", type=Path)
    commands.add_parser("serve")
    commands.add_parser("healthcheck")
    args = parser.parse_args()
    try:
        if args.command == "import-model":
            import_model(args.archive)
        elif args.command == "serve":
            serve()
        else:
            healthcheck()
    except Exception as exc:
        print(f"Falha na implantação: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
