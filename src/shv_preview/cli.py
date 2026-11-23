"""Integrity-only utility. This package is not a production VRAM runtime."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.resources as resources
from pathlib import Path


def load_evidence():
    loc = resources.files('shv_preview').joinpath('data/evidence.csv')
    with loc.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def verify_archive(zip_path: str, sha256_path: str) -> bool:
    path=Path(zip_path)
    wanted=Path(sha256_path).read_text(encoding='utf-8').strip().split()[0].lower()
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != wanted:
        print(f'FAIL SHA-256: expected {wanted}, computed {actual}')
        return False
    import zipfile
    with zipfile.ZipFile(path) as z:
        failed=z.testzip()
    if failed:
        print(f'FAIL ZIP CRC: {failed}')
        return False
    print(f'OK SHA-256 + ZIP CRC: {path.name} ({actual})')
    return True


def main(argv=None):
    p=argparse.ArgumentParser(prog='shv-evidence', description='Inspect SHV research claims and result archives (not a training engine).')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='Show scoped evidence records')
    verify=sub.add_parser('verify',help='Check SHA-256 and ZIP CRC of a downloaded result bundle')
    verify.add_argument('zip_file');verify.add_argument('sha256_file')
    args=p.parse_args(argv)
    if args.command=='list':
        for row in load_evidence():
            print(f"{row['round']:10} | {row['measured_result'][:31]:31} | {row['claim']}")
        return 0
    if args.command=='verify':
        return 0 if verify_archive(args.zip_file,args.sha256_file) else 1
    return 2
