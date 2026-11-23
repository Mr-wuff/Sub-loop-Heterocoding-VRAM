#!/usr/bin/env python3
"""Upload reviewed artifacts to an already created Hugging Face dataset repo.

The default mode only audits a local staging directory. To publish, first review
licenses/provenance, authenticate to HF with a write-capable token, and add --upload.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import sys
import zipfile

REPO = "wuff-mann/Sub-loop-Heterocoding-VRAM"

def audit(root: pathlib.Path) -> int:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("artifacts", [])
    if not entries:
        raise RuntimeError("No artifact records in manifest.json")
    for e in entries:
        p = root / e["path"]
        if not p.is_file():
            raise FileNotFoundError(p)
        data = p.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != e["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {p}")
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failed: {p}: {bad}")
        print(f"PASS {e['path']} [{len(data)} bytes]")
    return len(entries)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging", type=pathlib.Path, help="Unpacked HF candidate root with manifest.json")
    ap.add_argument("--upload", action="store_true", help="Write to Hugging Face Hub (requires prior write-capable authentication)")
    args = ap.parse_args()
    n = audit(args.staging)
    print(f"Verified {n} research archives.")
    if not args.upload:
        print("Dry run only. To upload, use --upload after reviewing licenses and authentication.")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print("Uploading as", who.get("name"))
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(args.staging), commit_message="data: add SHV research evidence archives")
    print(f"Uploaded https://huggingface.co/datasets/{REPO}")

if __name__ == "__main__":
    main()
