"""Build and activate pinned local V1 releases without changing Git or volumes."""

import argparse
import json
import os
import secrets
import shutil
import ssl
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dice_viewer.recognition.package import digest, verify_package  # noqa: E402

STATE = ROOT / "runtime/v1"
MINIO_IMAGE = "sha256:a3475aaa2694039238a5728913f45f58c45539012c74228796408fb8d9147ba6"


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(args, capture=True):
    return subprocess.run(args, cwd=ROOT, check=True, text=True,
                          encoding="utf-8", stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def release(identifier):
    if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in identifier):
        raise ValueError("Invalid release ID")
    folder = ROOT / "releases" / identifier
    sha = digest(folder / "RELEASE.json")
    return folder, sha, verify_package(folder, sha)


def configure():
    folder = STATE / "secrets"
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in (("minio-user", "dice-v1-local"), ("minio-password", secrets.token_urlsafe(36))):
        path = folder / name
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(value)
    print("Local secret files ready; existing credentials preserved", flush=True)


def build(identifier):
    folder, sha, manifest = release(identifier)
    directory = STATE / "builds"
    directory.mkdir(parents=True, exist_ok=True)
    existing = directory / f"{identifier}.json"
    if existing.exists() and json.loads(existing.read_text()).get("exit_code") == 0:
        raise ValueError("This release already has an immutable image; create a new release ID instead of rebuilding")
    log = directory / f"{identifier}.log"
    iidfile = directory / f"{identifier}.iid"
    record = {"release_id": identifier, "manifest_sha256": sha, "started_at_utc": utc()}
    started = time.perf_counter()
    command = ["docker", "build", "--progress=plain", "--platform", "linux/amd64",
               "--build-arg", f"DICE_BUNDLE_SHA256={sha}", "--build-arg", f"DICE_RELEASE_ID={identifier}",
               "--iidfile", str(iidfile), "-t", f"dice-v1:{identifier}",
               "-f", str(folder / "deploy/v1/Dockerfile"), str(folder)]
    if sys.platform == "win32":
        # Public CA certificates only; no private key or credentials are exported.
        # This temporary build mount is absent from the resulting image.
        ca = STATE / "build-ca.pem"
        certificates = [ssl.DER_cert_to_PEM_cert(cert) for store in ("ROOT", "CA")
                        for cert, encoding, trust in ssl.enum_certificates(store) if encoding == "x509_asn"]
        ca.write_text("".join(certificates), encoding="ascii")
        command[2:2] = ["--secret", f"id=build_ca,src={ca}"]
    print(f"Building {identifier}; output: {log}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    record.update(finished_at_utc=utc(), seconds=time.perf_counter() - started, exit_code=result.returncode)
    if result.returncode == 0:
        record["image_id"] = iidfile.read_text().strip()
        image = json.loads(run(["docker", "image", "inspect", record["image_id"]]).stdout)[0]
        labels = image["Config"].get("Labels", {})
        if labels.get("io.dice-viewer.bundle-sha256") != sha or labels.get("io.dice-viewer.release") != identifier:
            raise ValueError("Built image labels do not match release")
        record["image_bytes"] = image["Size"]
    write_json(directory / f"{identifier}.json", record)
    if result.returncode:
        raise RuntimeError(f"Docker build failed; inspect {log}")
    print(json.dumps(record, indent=2), flush=True)


def compose(folder, envfile, cuda):
    args = ["docker", "compose", "--project-name", os.environ.get("DICE_COMPOSE_PROJECT", "dice-v1"), "--project-directory", str(ROOT),
            "--env-file", str(envfile), "-f", str(folder / "deploy/v1/compose.yaml")]
    if cuda:
        args += ["-f", str(folder / "deploy/v1/compose.cuda.yaml")]
    return args


def activate(identifier, cuda=False, action="activate", historical=None):
    configure()
    folder, sha, manifest = release(identifier)
    built = ({"exit_code": 0, "manifest_sha256": historical["manifest_sha256"], "image_id": historical["image_id"]}
             if historical else json.loads((STATE / "builds" / f"{identifier}.json").read_text()))
    if built.get("exit_code") != 0 or built["manifest_sha256"] != sha:
        raise ValueError("Successful build of the exact release is required")
    image = json.loads(run(["docker", "image", "inspect", built["image_id"]]).stdout)[0]
    labels = image["Config"]["Labels"]
    if labels.get("io.dice-viewer.bundle-sha256") != sha or labels.get("io.dice-viewer.release") != identifier:
        raise ValueError("Image digest does not match release")
    envfile = STATE / "releases" / f"{identifier}.env"
    envfile.parent.mkdir(parents=True, exist_ok=True)
    envfile.write_text("\n".join((f"DICE_IMAGE={built['image_id']}", f"DICE_BUNDLE_SHA256={sha}",
                                  f"DICE_MINIO_IMAGE={MINIO_IMAGE}",
                                  f"DICE_SECRETS_DIR={(STATE / 'secrets').as_posix()}",
                                  "DICE_API_PORT=" + os.environ.get("DICE_API_PORT", "23000"),
                                  "DICE_MINIO_PORT=" + os.environ.get("DICE_MINIO_PORT", "29000"),
                                  "DICE_MINIO_CONSOLE_PORT=" + os.environ.get("DICE_MINIO_CONSOLE_PORT", "29001"))) + "\n")
    previous = json.loads((STATE / "active.json").read_text()) if (STATE / "active.json").exists() else None
    started = time.perf_counter()
    event = {"action": action, "release_id": identifier, "manifest_sha256": sha,
             "image_id": built["image_id"], "cuda": cuda, "started_at_utc": utc(), "api_port": int(os.environ.get("DICE_API_PORT", "23000"))}
    commands = compose(folder, envfile, cuda)
    pending = STATE / "pending.json"
    write_json(pending, {"target": event, "last_healthy": previous, "status": "activating"})
    try:
        run(commands + ["up", "-d", "--wait", "--wait-timeout", "180"], capture=False)
        container = run(commands + ["ps", "-q", "api"]).stdout.strip()
        inspected = json.loads(run(["docker", "inspect", container]).stdout)[0]
        if inspected["Image"] != built["image_id"] or inspected["State"]["Health"]["Status"] != "healthy":
            raise RuntimeError("Expected immutable image is not healthy")
        startup = json.loads(run(["docker", "exec", container, "python", "-m",
                                  "dice_viewer.recognition.operations", "identify"]).stdout)
        if startup["release_sha256"] != sha:
            raise RuntimeError("Runtime state does not match the pinned release")
        if cuda and not startup["resolved_device"].startswith("cuda"):
            raise RuntimeError("CUDA override did not activate CUDA")
        event.update(status="healthy", container_id=container, startup=startup)
    except Exception:
        event.update(status="failed", finished_at_utc=utc(), seconds=time.perf_counter() - started)
        write_json(pending, {"target": event, "last_healthy": previous, "status": "failed"})
        with (STATE / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        raise
    event.update(finished_at_utc=utc(), seconds=time.perf_counter() - started, previous=previous)
    write_json(STATE / "active.json", event)
    pending.unlink(missing_ok=True)
    with (STATE / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({k: v for k, v in event.items() if k != "previous"}) + "\n")
    print(json.dumps({k: v for k, v in event.items() if k != "previous"}, indent=2), flush=True)


def preserve(identifier):
    folder, sha, manifest = release(identifier)
    built = json.loads((STATE / "builds" / f"{identifier}.json").read_text())
    if built.get("exit_code") != 0 or built["manifest_sha256"] != sha:
        raise ValueError("A successful exact build is required for recovery export")
    target = STATE / "recovery" / identifier
    target.mkdir(parents=True, exist_ok=False)
    images = {"api": built["image_id"], "minio": MINIO_IMAGE}
    print("Preserving immutable Docker images and release sources", flush=True)
    for name, image_id in images.items():
        run(["docker", "image", "save", "-o", str(target / f"{name}.tar"), image_id], capture=False)
    archive = Path(shutil.make_archive(str(target / "bundle"), "gztar", root_dir=folder))
    index = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
             for p in target.iterdir() if p.is_file()}
    record = {"release_id": identifier, "manifest_sha256": sha, "images": images, "files": index,
              "created_at_utc": utc(), "data_recovery_requires": "Preserved Docker volumes and runtime/v1/secrets"}
    write_json(target / "RECOVERY.json", record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("configure")
    builder = commands.add_parser("build")
    builder.add_argument("release_id")
    exporter = commands.add_parser("preserve")
    exporter.add_argument("release_id")
    activator = commands.add_parser("activate")
    activator.add_argument("release_id")
    activator.add_argument("--cuda", action="store_true")
    commands.add_parser("rollback")
    commands.add_parser("status")
    args = parser.parse_args()
    if args.command == "configure":
        configure()
    elif args.command == "build":
        build(args.release_id)
    elif args.command == "activate":
        activate(args.release_id, args.cuda)
    elif args.command == "preserve":
        preserve(args.release_id)
    elif args.command == "rollback":
        pending_path = STATE / "pending.json"
        pending = json.loads(pending_path.read_text()) if pending_path.exists() else None
        current = json.loads((STATE / "active.json").read_text()) if (STATE / "active.json").exists() else {}
        previous = pending.get("last_healthy") if pending else current.get("previous")
        if not previous:
            raise RuntimeError("No preceding approved V1 deployment exists; preserve this initial release for recovery")
        activate(previous["release_id"], previous["cuda"], action="rollback", historical=previous)
    else:
        print((STATE / "active.json").read_text())


if __name__ == "__main__":
    main()
