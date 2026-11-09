"""Download KMAR-50K from Mendeley Data (both parts) with sha256 verification.

Layout produced:
    data/raw/kmar50k/{train,test}/{artifact,clean}/<original filename>
    data/raw/meta/  (API listings, folder maps, cohort CSVs)

Resumable: files already present with the correct sha256 are skipped.
Usage:
    python scripts/download_kmar.py [--workers 8] [--dry-run]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
META = RAW / "meta"
OUT = RAW / "kmar50k"
API = "https://data.mendeley.com/public-api/datasets"
DATASETS = {"xw7mrg7ntg": 6, "95w9f5tzz8": 6}  # part1, part2 (version)

# Mendeley folder name -> (split, kind)
FOLDER_MAP = {
    "ArtifactData_part1": ("train", "artifact"),
    "GroundTruthData_part1": ("train", "clean"),
    "ArtifactData_part2": ("train", "artifact"),
    "GroundTruthData_part2": ("train", "clean"),
    "1_GroundTruthData": ("test", "clean"),
    "1_ArtifactData": ("test", "artifact"),
}


HEADERS = {"User-Agent": "curl/7.81.0", "Accept": "*/*"}


def open_url(url, timeout):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def get_json(url):
    with open_url(url, 60) as r:
        return json.load(r)


def sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def build_manifest():
    META.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ds, ver in DATASETS.items():
        info = get_json(f"{API}/{ds}")  # latest version; includes the file list
        assert info["version"] == ver, f"{ds}: expected v{ver}, got v{info['version']}"
        folders = get_json(f"{API}/{ds}/folders/{ver}")
        (META / f"ds_{ds}.json").write_text(json.dumps(info))
        (META / f"folders_{ds}.json").write_text(json.dumps(folders))
        fname = {f["id"]: f["name"] for f in folders}
        for f in info["files"]:
            cd = f["content_details"]
            folder = fname.get(f.get("folder_id"), "root")
            if folder == "root":
                dest = META / f["filename"]
            elif folder in FOLDER_MAP:
                split, kind = FOLDER_MAP[folder]
                dest = OUT / split / kind / f["filename"]
            else:
                raise RuntimeError(f"Unknown folder {folder!r} in {ds}")
            jobs.append(dict(url=cd["download_url"], sha=cd["sha256_hash"],
                             size=cd["size"], dest=str(dest), dataset=ds, folder=folder))
    (META / "manifest.json").write_text(json.dumps(jobs, indent=1))
    return jobs


def fetch(job, retries=4):
    dest = Path(job["dest"])
    if dest.exists() and dest.stat().st_size == job["size"] and sha256(dest) == job["sha"]:
        return "skip", job
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            with open_url(job["url"], 120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if sha256(tmp) != job["sha"]:
                raise IOError("sha256 mismatch")
            os.replace(tmp, dest)
            return "ok", job
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return f"FAIL: {err}", job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_manifest()
    total = sum(j["size"] for j in jobs)
    from collections import Counter
    print(f"{len(jobs)} files, {total / 1e9:.2f} GB")
    print(Counter((j["folder"]) for j in jobs))
    if args.dry_run:
        return

    done = 0
    stats = Counter()
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            status, job = fut.result()
            stats[status.split(":")[0]] += 1
            done += job["size"]
            if status.startswith("FAIL"):
                failed.append((job["dest"], status))
            if i % 50 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"[{i}/{len(jobs)}] {done / 1e9:.2f}/{total / 1e9:.2f} GB "
                      f"{el / 60:.1f} min {dict(stats)}", flush=True)
    if failed:
        (META / "failed.json").write_text(json.dumps(failed, indent=1))
        print(f"{len(failed)} failures -> {META / 'failed.json'}; rerun to retry")
        sys.exit(1)
    print("All files present and verified.")


if __name__ == "__main__":
    main()
