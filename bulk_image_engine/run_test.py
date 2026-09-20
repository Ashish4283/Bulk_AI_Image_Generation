"""
One-command verification.

    python run_test.py

Renders the 5 prompts in sample_prompts.txt, asserts 1.jpg .. 5.jpg exist and
decode, and writes test_batch.zip. Exits non-zero if anything is wrong, so it
can be used as a smoke test in CI or on a freshly built GPU instance.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from engine import BulkImageEngine, load_prompts, is_valid_image, zip_batch

HERE = Path(__file__).resolve().parent
EXPECTED = 5


def main() -> int:
    offload = "--offload" in sys.argv
    prompts = load_prompts(HERE / "sample_prompts.txt")
    out_dir = HERE / "output" / "test_batch"
    zip_path = HERE / "output" / "test_batch.zip"

    print(f"[test] {len(prompts)} prompts, offload={offload}")
    if len(prompts) != EXPECTED:
        print(f"[FAIL] expected {EXPECTED} prompts, found {len(prompts)}")
        return 1

    engine = BulkImageEngine(offload=offload)
    summary = engine.run_batch(prompts, out_dir)
    print(f"[test] {summary}")

    problems: list[str] = []

    # 1. every expected file exists, decodes, and is numbered correctly
    for i in range(1, EXPECTED + 1):
        f = out_dir / f"{i}.jpg"
        if not is_valid_image(f):
            problems.append(f"missing or unreadable: {f.name}")

    # 2. no extra images crept in
    strays = sorted(
        p.name for p in out_dir.glob("*.jpg")
        if p.stem not in {str(i) for i in range(1, EXPECTED + 1)}
    )
    if strays:
        problems.append(f"unexpected files: {strays}")

    # 3. the archive contains exactly 1.jpg .. 5.jpg
    result = zip_batch(out_dir, zip_path, prompts)
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist(), key=lambda n: int(Path(n).stem))
    expected_names = [f"{i}.jpg" for i in range(1, EXPECTED + 1)]
    if names != expected_names:
        problems.append(f"zip contents {names} != {expected_names}")

    if problems:
        print("\n[FAIL]")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"\n[PASS] 1.jpg .. {EXPECTED}.jpg verified")
    print(f"[PASS] {zip_path.name} ({result['size_mb']} MB, {result['count']} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
