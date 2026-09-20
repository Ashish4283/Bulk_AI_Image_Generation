"""
One-command verification of the full delivery shape, in miniature.

    python run_test.py                # add --offload on a low-VRAM GPU

Test mode mirrors production exactly, just smaller:

    10 prompts x 3 batches = 30 images

    output/batch_1/1.jpg .. 10.jpg  -> batch_1.zip
    output/batch_2/1.jpg .. 10.jpg  -> batch_2.zip
    output/batch_3/1.jpg .. 10.jpg  -> batch_3.zip
                                    -> all_batches.zip

Production is the same command with a 700-prompt file and 15 batches.

Checks performed:
  * each batch folder contains exactly 1.jpg .. 10.jpg, all decodable
  * the same prompt differs between batches (real variations, not copies)
  * each batch zip holds exactly its 10 images, in order
  * the shipping bundle holds all 3 batch archives

Exits non-zero if anything is wrong, so it works as a smoke test on a
freshly built GPU instance.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from engine import (
    BulkImageEngine, load_prompts, is_valid_image, zip_batch,
    bundle_zips, seed_for,
)

HERE = Path(__file__).resolve().parent
BATCHES = 3


def main() -> int:
    offload = "--offload" in sys.argv
    prompts = load_prompts(HERE / "sample_prompts.txt")
    n = len(prompts)
    out_root = HERE / "output"

    print(f"[test] {n} prompts x {BATCHES} batches = {n * BATCHES} images, "
          f"offload={offload}")

    engine = BulkImageEngine(offload=offload)
    problems: list[str] = []
    zips: list[Path] = []

    for batch_no in range(1, BATCHES + 1):
        out_dir = out_root / f"batch_{batch_no}"
        print(f"\n=== batch {batch_no}/{BATCHES} -> {out_dir} ===")
        summary = engine.run_batch(prompts, out_dir, batch_no=batch_no)
        print(f"[test] batch {batch_no}: {summary}")

        # every expected file present, decodable, correctly numbered
        for i in range(1, n + 1):
            if not is_valid_image(out_dir / f"{i}.jpg"):
                problems.append(f"batch {batch_no}: missing/unreadable {i}.jpg")

        strays = sorted(
            p.name for p in out_dir.glob("*.jpg")
            if p.stem not in {str(i) for i in range(1, n + 1)}
        )
        if strays:
            problems.append(f"batch {batch_no}: unexpected files {strays}")

        zip_path = out_root / f"batch_{batch_no}.zip"
        result = zip_batch(out_dir, zip_path, prompts)
        zips.append(zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(zf.namelist(), key=lambda x: int(Path(x).stem))
        expected = [f"{i}.jpg" for i in range(1, n + 1)]
        if names != expected:
            problems.append(f"batch {batch_no}: zip has {names}, expected {expected}")
        print(f"[test] {zip_path.name}: {result['count']} images, "
              f"{result['size_mb']} MB")

    # batches must be genuine variations, not identical renders
    for i in (1, n):
        seeds = {seed_for(b, i) for b in range(1, BATCHES + 1)}
        if len(seeds) != BATCHES:
            problems.append(f"image {i}: seeds repeat across batches ({seeds})")
        sizes = {
            (out_root / f"batch_{b}" / f"{i}.jpg").stat().st_size
            for b in range(1, BATCHES + 1)
            if (out_root / f"batch_{b}" / f"{i}.jpg").is_file()
        }
        if len(sizes) == 1:
            problems.append(
                f"image {i}: identical file size in every batch - "
                "batches may not be varying"
            )

    bundle_path = out_root / "all_batches.zip"
    bundle = bundle_zips(zips, bundle_path)
    with zipfile.ZipFile(bundle_path) as zf:
        bundled = sorted(zf.namelist())
    expected_bundle = sorted(f"batch_{b}.zip" for b in range(1, BATCHES + 1))
    if bundled != expected_bundle:
        problems.append(f"bundle has {bundled}, expected {expected_bundle}")

    if problems:
        print("\n[FAIL]")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"\n[PASS] {BATCHES} batches x {n} images, each numbered 1..{n}")
    print(f"[PASS] {bundle['count']} batch archives bundled into "
          f"{bundle_path.name} ({bundle['size_mb']} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
