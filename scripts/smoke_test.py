"""Smoke-test a running API and optionally generate traffic for the dashboards.

Usage:
    python scripts/smoke_test.py --url http://localhost:8000
    python scripts/smoke_test.py --url http://localhost --load 300   # traffic for Grafana
Exits non-zero if any check fails (used by CI after `docker run`).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

SAMPLE = json.loads((Path(__file__).resolve().parent / "sample_input.json").read_text())


def random_patient(rng: random.Random) -> dict:
    return {
        "age": rng.randint(30, 77), "sex": rng.randint(0, 1), "cp": rng.randint(1, 4),
        "trestbps": rng.randint(95, 190), "chol": rng.randint(130, 400),
        "fbs": rng.randint(0, 1), "restecg": rng.randint(0, 2),
        "thalach": rng.randint(80, 200), "exang": rng.randint(0, 1),
        "oldpeak": round(rng.uniform(0, 5), 1), "slope": rng.randint(1, 3),
        "ca": rng.randint(0, 3), "thal": rng.choice([3, 6, 7]),
    }


def wait_until_healthy(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/health", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise SystemExit(f"API at {url} not healthy after {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--load", type=int, default=0, help="extra random requests to send")
    parser.add_argument("--wait", type=int, default=60)
    args = parser.parse_args()
    url = args.url.rstrip("/")

    wait_until_healthy(url, args.wait)
    failures = 0

    for record in SAMPLE:
        resp = requests.post(f"{url}/predict", json=record, timeout=10)
        required = {"prediction", "confidence", "risk_band"}
        ok = resp.status_code == 200 and required <= resp.json().keys()
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] /predict -> {resp.status_code} {resp.text}")

    bad = dict(SAMPLE[0], cp=9)
    resp = requests.post(f"{url}/predict", json=bad, timeout=10)
    ok = resp.status_code == 422
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] invalid input rejected -> {resp.status_code}")

    resp = requests.get(f"{url}/metrics", timeout=10)
    ok = resp.ok and "model_predictions_total" in resp.text
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] /metrics exposes prediction counters")

    rng = random.Random(7)
    for i in range(args.load):
        payload = random_patient(rng) if rng.random() > 0.05 else dict(SAMPLE[0], age=-1)
        requests.post(f"{url}/predict", json=payload, timeout=10)
        if i % 50 == 0:
            print(f"  load: sent {i}/{args.load}")
        time.sleep(0.05)

    print("SMOKE TEST", "FAILED" if failures else "PASSED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
