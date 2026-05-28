#!/usr/bin/env python3
"""Download a float32 .eim model from Edge Impulse Studio API.

Usage:
    export EI_API_KEY="ei_abc123..."   # Your project API key from EI Studio > Dashboard > Keys
    python scripts/download_float32_model.py

This will:
  1. Trigger a float32 Linux x86_64 runner build for project 371734
  2. Poll until the build job completes
  3. Download the .eim file to models/modelfile.eim (backing up the old one)
"""

import os
import sys
import time
import shutil
import zipfile
import tempfile
import requests

PROJECT_ID = 371734
API_BASE = "https://studio.edgeimpulse.com/v1"
DEPLOY_TARGET = "runner-linux-x86_64"  # Linux x86_64 runner
MODEL_TYPE = "float32"
ENGINE = "tflite"  # NOT tflite-eon — EON Compiler is incompatible with Linux runner's full TFLite
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modelfile.eim")


def main():
    api_key = os.environ.get("EI_API_KEY")
    if not api_key:
        print("ERROR: Set EI_API_KEY environment variable first.")
        print("  export EI_API_KEY='ei_...'")
        print("  Find it at: https://studio.edgeimpulse.com > project 371734 > Dashboard > Keys")
        sys.exit(1)

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }

    output = os.path.abspath(OUTPUT_PATH)

    # Step 1: Trigger build
    print(f"Building {MODEL_TYPE} model for {DEPLOY_TARGET}...")
    build_url = f"{API_BASE}/api/{PROJECT_ID}/jobs/build-ondevice-model?type={DEPLOY_TARGET}"
    build_body = {
        "engine": ENGINE,
        "modelType": MODEL_TYPE,
    }
    resp = requests.post(build_url, json=build_body, headers=headers, timeout=30)
    if resp.status_code != 200:
        print(f"Build request failed ({resp.status_code}): {resp.text}")
        sys.exit(1)

    data = resp.json()
    if not data.get("success"):
        print(f"Build request failed: {data.get('error', 'unknown error')}")
        sys.exit(1)

    job_id = data["id"]
    deploy_version = data["deploymentVersion"]
    print(f"  Job ID: {job_id}")
    print(f"  Deployment version: {deploy_version}")

    # Step 2: Poll for job completion
    print("Waiting for build to complete...")
    job_url = f"{API_BASE}/api/{PROJECT_ID}/jobs/{job_id}/status"
    for attempt in range(120):  # up to ~10 minutes
        time.sleep(5)
        resp = requests.get(job_url, headers=headers, timeout=30)
        if resp.status_code == 200:
            status = resp.json()
            job_obj = status.get("job", {})
            finished = job_obj.get("finished", False)
            finished_successful = job_obj.get("finishedSuccessful", False)
            if finished:
                if finished_successful:
                    print("  Build completed successfully!")
                    break
                else:
                    print(f"  Build FAILED.")
                    sys.exit(1)
        if attempt % 6 == 0:
            print(f"  Still building... ({attempt * 5}s)")

    else:
        print("  Build timed out after 10 minutes.")
        sys.exit(1)

    # Step 3: Download the deployment
    print("Downloading deployment...")
    dl_url = f"{API_BASE}/api/{PROJECT_ID}/deployment/history/{deploy_version}/download"
    resp = requests.get(dl_url, headers=headers, stream=True, timeout=120)
    if resp.status_code != 200:
        print(f"Download failed ({resp.status_code}): {resp.text[:200]}")
        sys.exit(1)

    # Save to temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name

    # The download is a ZIP containing the .eim file
    # Try to extract .eim from it, or if it's the .eim directly, just copy
    try:
        if zipfile.is_zipfile(tmp_path):
            print("  Extracting .eim from ZIP...")
            with zipfile.ZipFile(tmp_path, "r") as zf:
                eim_files = [f for f in zf.namelist() if f.endswith(".eim")]
                if eim_files:
                    eim_name = eim_files[0]
                    # Backup old model
                    if os.path.exists(output):
                        backup = output + ".int8.bak"
                        print(f"  Backing up old model to {backup}")
                        shutil.copy2(output, backup)
                    # Extract
                    with zf.open(eim_name) as src, open(output, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    os.chmod(output, 0o755)
                    print(f"  Saved to {output}")
                else:
                    print(f"  No .eim file found in ZIP. Contents: {zf.namelist()}")
                    sys.exit(1)
        else:
            # Might be a raw binary
            print("  Download is not a ZIP, treating as raw .eim...")
            if os.path.exists(output):
                backup = output + ".int8.bak"
                print(f"  Backing up old model to {backup}")
                shutil.copy2(output, backup)
            shutil.move(tmp_path, output)
            os.chmod(output, 0o755)
            print(f"  Saved to {output}")
            tmp_path = None  # don't delete
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    print("\nDone! Float32 model downloaded.")
    print("Run the detection test to verify:")
    print("  python -c \"from apps.detector.inference import detect_instruments; print(detect_instruments('data/frames/frame_missing_scissors3.png'))\"")


if __name__ == "__main__":
    main()
