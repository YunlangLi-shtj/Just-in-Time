#!/usr/bin/env python3
"""Download FLUX diffusers checkpoints from the ModelScope mirror.

huggingface.co is unreachable from this server and both BFL repos are gated
(no HF token available), while ModelScope mirrors them without authentication.
Only the diffusers-format component files are fetched; the duplicated
single-file checkpoints and preview images in the repo are skipped.

Usage:
    python tools/download_modelscope.py --model both
    python tools/download_modelscope.py --model flux1 --workers 8
"""

import argparse
import concurrent.futures as futures
import os
import sys
import threading
import time

import requests

API_URL = "https://www.modelscope.cn/api/v1/models/{repo_id}/repo/files"
RESOLVE_URL = "https://www.modelscope.cn/models/{repo_id}/resolve/master/{path}"

DEFAULT_DEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)

REPOS = {
    "flux1": {
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "target": "FLUX.1-dev",
        "prefixes": (
            "model_index.json",
            "scheduler/",
            "text_encoder/",
            "text_encoder_2/",
            "tokenizer/",
            "tokenizer_2/",
            "transformer/",
            "vae/",
        ),
    },
    "flux2": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "target": "FLUX.2-klein-base-9B",
        "prefixes": (
            "model_index.json",
            "scheduler/",
            "text_encoder/",
            "tokenizer/",
            "transformer/",
            "vae/",
        ),
    },
}

_print_lock = threading.Lock()


def log(message):
    with _print_lock:
        print(message, flush=True)


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024


def is_wanted(path, prefixes):
    return any(path == prefix or path.startswith(prefix) for prefix in prefixes)


def list_repo_files(repo_id):
    session = requests.Session()
    response = session.get(
        API_URL.format(repo_id=repo_id),
        params={"Revision": "master", "Root": "", "Recursive": "True"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("Code") != 200:
        raise RuntimeError(f"ModelScope API error for {repo_id}: {payload!r}")
    return payload["Data"]["Files"]


def download_file(repo_id, path, size, dest_dir, retries=3):
    url = RESOLVE_URL.format(repo_id=repo_id, path=path)
    final_path = os.path.join(dest_dir, path)
    part_path = final_path + ".part"
    os.makedirs(os.path.dirname(final_path) or dest_dir, exist_ok=True)

    if os.path.exists(final_path):
        if os.path.getsize(final_path) == size:
            return path, size, 0.0
        os.remove(final_path)

    if os.path.exists(part_path) and os.path.getsize(part_path) == size:
        os.replace(part_path, final_path)
        return path, size, 0.0

    for attempt in range(1, retries + 1):
        try:
            resuming = os.path.exists(part_path) and os.path.getsize(part_path) > 0
            if resuming and os.path.getsize(part_path) > size:
                os.remove(part_path)
                resuming = False
            initial_size = os.path.getsize(part_path) if resuming else 0

            headers = {}
            if resuming:
                headers["Range"] = f"bytes={initial_size}-"

            with requests.Session() as session:
                with session.get(
                    url, headers=headers, stream=True, timeout=(30, 180)
                ) as response:
                    if resuming and response.status_code == 206:
                        mode = "ab"
                    elif response.status_code == 200:
                        mode = "wb"
                    else:
                        response.raise_for_status()
                        raise RuntimeError(
                            f"unexpected status {response.status_code} for {path}"
                        )

                    started = time.time()
                    with open(part_path, mode) as handle:
                        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                            if chunk:
                                handle.write(chunk)

            downloaded_size = os.path.getsize(part_path)
            if downloaded_size != size:
                raise IOError(
                    f"size mismatch for {path}: got {downloaded_size}, expected {size}"
                )
            os.replace(part_path, final_path)
            elapsed = max(time.time() - started, 1e-6)
            return path, size, max(size - initial_size, 0) / elapsed
        except Exception as error:  # noqa: BLE001 - retry any transient failure
            if attempt == retries:
                raise
            log(f"[retry {attempt}/{retries - 1}] {path}: {error}")
            time.sleep(2 * attempt)

    raise RuntimeError(f"failed to download {path}")


def download_model(key, dest_root, workers):
    spec = REPOS[key]
    repo_id = spec["repo_id"]
    dest_dir = os.path.join(dest_root, spec["target"])

    log(f"\n=== {key}: {repo_id} -> {dest_dir} ===")
    entries = list_repo_files(repo_id)
    files = [
        entry
        for entry in entries
        if entry.get("Type") == "blob" and is_wanted(entry["Path"], spec["prefixes"])
    ]
    files.sort(key=lambda entry: entry["Path"])
    if not files:
        raise RuntimeError(f"no matching files returned for {repo_id}")

    total_bytes = sum(entry["Size"] for entry in files)
    log(f"{len(files)} files, {human_size(total_bytes)} total")

    started = time.time()
    completed_bytes = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(
                download_file, repo_id, entry["Path"], entry["Size"], dest_dir
            ): entry
            for entry in files
        }
        for future in futures.as_completed(pending):
            path, size, speed = future.result()
            completed_bytes += size
            elapsed = max(time.time() - started, 1e-6)
            rate = "already present" if speed == 0.0 else f"{human_size(speed)}/s"
            log(
                f"[{human_size(completed_bytes)}/{human_size(total_bytes)}] "
                f"{path} ({human_size(size)}, {rate}, "
                f"avg {human_size(completed_bytes / elapsed)}/s)"
            )

    missing = []
    for entry in files:
        target = os.path.join(dest_dir, entry["Path"])
        if not os.path.isfile(target) or os.path.getsize(target) != entry["Size"]:
            missing.append(entry["Path"])
    if missing:
        raise RuntimeError(f"verification failed for {len(missing)} files: {missing}")

    log(
        f"=== {key} finished: {len(files)} files, {human_size(total_bytes)}, "
        f"{time.time() - started:.1f}s ==="
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("flux1", "flux2", "both"),
        default="both",
        help="which checkpoint(s) to download",
    )
    parser.add_argument(
        "--dest", default=DEFAULT_DEST, help=f"destination root (default: {DEFAULT_DEST})"
    )
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads")
    args = parser.parse_args()

    keys = ("flux1", "flux2") if args.model == "both" else (args.model,)
    os.makedirs(args.dest, exist_ok=True)
    for key in keys:
        download_model(key, args.dest, args.workers)
    log("\nAll requested checkpoints downloaded and verified.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
