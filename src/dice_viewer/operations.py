"""Human timing study and GPU-container HTTP latency measurement."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from dice_viewer.dataset import load_rows, numeric_value, split_for, valid_value
from dice_viewer.evaluation import percentile


def samples(raw: Path, limit: int = 30) -> list[dict]:
    rows = [row for row in load_rows(raw) if split_for(row) == "test" and row.get("values_trusted")
            and len(row["dice"]) == 1 and valid_value(row["dice"][0])]
    rows.sort(key=lambda row: row["file_name"])
    if len(rows) < limit:
        raise ValueError(f"Only {len(rows)} eligible single-die held-out photos; requested {limit}")
    return rows[:limit]


def post_image(url: str, path: Path) -> dict:
    boundary = uuid.uuid4().hex
    image_bytes = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n").encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()
    request = Request(url.rstrip("/") + "/v1/predictions?details=true", body,
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def post_feedback(url: str, prediction_id: str, sides: int, value: int, confirmed: bool) -> None:
    data = json.dumps({"confirmed": confirmed, "sides": sides, "value": value}).encode()
    request = Request(url.rstrip("/") + f"/v1/predictions/{prediction_id}/feedback", data,
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=30) as response:
        response.read()


def study_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("manual", "api", "compare"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.mode == "compare":
        manual = json.loads((args.output / "study-manual.json").read_text(encoding="utf-8"))
        api = json.loads((args.output / "study-api.json").read_text(encoding="utf-8"))
        if [row["file"] for row in manual["rows"]] != [row["file"] for row in api["rows"]]:
            raise ValueError("Both modes must use the same photos in the same order")
        reduction = 1 - api["median_seconds"] / manual["median_seconds"]
        report = {"manual_median_seconds": manual["median_seconds"], "api_median_seconds": api["median_seconds"],
                  "time_reduction": reduction, "manual_errors": manual["errors"], "api_errors": api["errors"],
                  "goal_reduction_50pct_met": reduction >= 0.5}
        (args.output / "study-comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    if os.name != "nt":
        raise RuntimeError("The interactive photo-opening study is configured for Windows")
    measurements: list[dict] = []
    for row in samples(args.raw, args.count):
        path = args.raw / row["file_name"]
        os.startfile(path)  # Opens the participant-visible photo.
        print(f"Photo: {path.name}")
        started = time.perf_counter()
        if args.mode == "manual":
            answer = input("Type sides and value (example: 20 14): ").strip()
            sides, value = map(int, answer.split())
        else:
            prediction = post_image(args.url, path)
            print(f"API: {prediction['status']} {prediction['sides']} {prediction['value']}")
            answer = input("Enter to accept, or type corrected sides and value: ").strip()
            if not answer and (prediction["sides"] is None or prediction["value"] is None):
                answer = input("No readable result; enter sides and value: ").strip()
                if not answer:
                    raise ValueError("A human result is required for an unreadable photo")
            if answer:
                sides, value = map(int, answer.split())
                post_feedback(args.url, prediction["prediction_id"], sides, value, False)
            else:
                sides, value = prediction["sides"], prediction["value"]
                if sides is not None and value is not None:
                    post_feedback(args.url, prediction["prediction_id"], sides, value, True)
        elapsed = time.perf_counter() - started
        truth = (int(row["dice"][0]["type"][1:]), numeric_value(row["dice"][0]))
        measurements.append({"file": row["file_name"], "seconds": elapsed, "correct": (sides, value) == truth})
        print(f"Elapsed: {elapsed:.3f}s")
    report = {"mode": args.mode, "count": len(measurements),
              "median_seconds": statistics.median(item["seconds"] for item in measurements),
              "errors": sum(not item["correct"] for item in measurements), "rows": measurements}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"study-{args.mode}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    images = [args.raw / row["file_name"] for row in samples(args.raw)]
    roundtrip: list[float] = []
    inference: list[float] = []
    for index in range(args.warmup + args.requests):
        started = time.perf_counter()
        result = post_image(args.url, images[index % len(images)])
        elapsed_ms = (time.perf_counter() - started) * 1000
        if index >= args.warmup:
            roundtrip.append(elapsed_ms)
            inference.append(float(result["inference_ms"]))
    report = {"warmup": args.warmup, "requests": args.requests,
              "inference_p50_ms": percentile(inference, 0.5), "inference_p95_ms": percentile(inference, 0.95),
              "http_p50_ms": percentile(roundtrip, 0.5), "http_p95_ms": percentile(roundtrip, 0.95)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
