"""Restore and run the approved model using only versioned files and Docker."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "models/v1-initial-20261117-r3"
PUBLIC_MINIO = "cgr.dev/chainguard/minio@sha256:4cf4831a2bbcf13ddca09c1cbcc9faff716dd3c4247e0babc32864b8ee8e0034"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore():
    manifest = json.loads((DUMP / "manifest.json").read_text(encoding="utf-8"))
    identifier = manifest["release_id"]
    if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in identifier):
        raise ValueError("Invalid release identifier")
    target = ROOT / "releases" / manifest["release_id"]
    sys.path.insert(0, str(ROOT / "src"))
    from dice_viewer.recognition.package import verify_package
    if target.exists():
        verify_package(target, manifest["release_sha256"])
        return target
    parent = ROOT / "releases"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix="restore-") as scratch:
        stage = Path(scratch)
        archive = stage / "model.tar.gz"
        with archive.open("wb") as output:
            for item in manifest["parts"]:
                name = item["name"]
                if Path(name).name != name or "/" in name or "\\" in name:
                    raise ValueError("Invalid dump part name")
                part = DUMP / name
                if part.stat().st_size != item["bytes"] or sha(part) != item["sha256"]:
                    raise ValueError(f"Dump part changed: {name}")
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output)
        if sha(archive) != manifest["archive_sha256"]:
            raise ValueError("Dump archive checksum mismatch")
        extracted = stage / "payload"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                destination = (extracted / member.name).resolve()
                if not destination.is_relative_to(extracted.resolve()) or not (member.isfile() or member.isdir()):
                    raise ValueError("Unsafe archive member")
                if "\\" in member.name or ":" in member.name:
                    raise ValueError("Nonportable archive path")
            source.extractall(extracted, filter="data")
        verify_package(extracted, manifest["release_sha256"])
        shutil.copytree(extracted, target)
        verify_package(target, manifest["release_sha256"])
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("restore", "run"))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--project", default="dice-model")
    parser.add_argument("--port", type=int, default=23000)
    args = parser.parse_args()
    if not args.project or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.project):
        parser.error("Use a lowercase Compose project name")
    if not 1024 <= args.port <= 65533:
        parser.error("Port must be 1024..65533")
    target = restore()
    print(f"Approved model restored and verified: {target}", flush=True)
    if args.command == "restore":
        return
    subprocess.run(["docker", "pull", "--platform", "linux/amd64", PUBLIC_MINIO], check=True)
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", PUBLIC_MINIO], text=True))[0]
    spec = importlib.util.spec_from_file_location("v1_manager", ROOT / "deploy/v1/manage.py")
    manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manager)
    manager.MINIO_IMAGE = image["Id"]
    manager.STATE = ROOT / "runtime/portable" / args.project
    os.environ["DICE_COMPOSE_PROJECT"] = args.project
    os.environ["DICE_API_PORT"] = str(args.port)
    os.environ["DICE_MINIO_PORT"] = str(args.port + 1)
    os.environ["DICE_MINIO_CONSOLE_PORT"] = str(args.port + 2)
    build_record = manager.STATE / "builds" / f"{target.name}.json"
    if not build_record.exists() or json.loads(build_record.read_text()).get("exit_code") != 0:
        manager.build(target.name)
    manager.activate(target.name, cuda=args.cuda)


if __name__ == "__main__":
    main()
