"""
bulk_image_engine.engine
========================

Batch image generation with SDXL-Lightning (4-step UNet on SDXL base 1.0).

Self-contained: this module reads only from its own folder and writes only
into the output directory it is given. It imports nothing from any parent
project.

CLI
---
    python engine.py --prompts prompts.txt --batch 1
    python engine.py --prompts prompts.txt --batch 1 --offload      # low VRAM
    python engine.py --prompts prompts.txt --batch-size 700 --all   # every batch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
LIGHTNING_CKPT = "sdxl_lightning_4step_unet.safetensors"

NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0
DEFAULT_SIZE = 1024
DEFAULT_BATCH_SIZE = 700
DEFAULT_BATCHES = 15          # 15 x 700 = 10,500 images

# Per-image seed = BASE_SEED + (batch - 1) * SEED_STRIDE + index.
# Every image is reproducible, and the same prompt gets a different seed in
# each batch, so batch 2 is a genuine variation rather than a duplicate.
# The stride exceeds any realistic prompt count, so seed ranges never collide.
BASE_SEED = 1_000_000
SEED_STRIDE = 1_000_000


def seed_for(batch_no: int, index: int) -> int:
    """Deterministic seed for one image of one batch."""
    return BASE_SEED + (batch_no - 1) * SEED_STRIDE + index


# --------------------------------------------------------------------------
# Prompt loading
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Prompt:
    """A prompt plus the index its image is named after."""
    index: int      # 1-based; becomes "<index>.jpg"
    line_no: int    # 1-based line in the source file
    text: str


def load_prompts(path: Path, renumber: bool = False) -> list[Prompt]:
    """
    Read prompts, one per line.

    Blank lines and stray carriage returns (CRLF files authored on Windows)
    never break the mapping between a prompt and its image number.

    renumber=False (default): a prompt's image number IS its line number, so
      line 12 always produces 12.jpg even if line 5 was blank. Deleting a
      prompt later cannot silently renumber every image after it.
    renumber=True: blank lines are closed up, producing an unbroken
      1..N sequence.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    # newline="" keeps \r visible so we can strip it rather than embed it.
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        raw_lines = fh.read().splitlines()

    prompts: list[Prompt] = []
    for line_no, raw in enumerate(raw_lines, start=1):
        text = raw.replace("\r", "").replace(" ", " ").strip()
        if not text:
            continue
        index = len(prompts) + 1 if renumber else line_no
        prompts.append(Prompt(index=index, line_no=line_no, text=text))

    if not prompts:
        raise ValueError(f"No usable prompts found in {path}")
    return prompts


def batch_slice(prompts: list[Prompt], batch_no: int, batch_size: int) -> list[Prompt]:
    """Return the prompts belonging to 1-based batch_no."""
    if batch_no < 1:
        raise ValueError("batch numbers start at 1")
    start = (batch_no - 1) * batch_size
    return prompts[start:start + batch_size]


def batch_count(prompts: list[Prompt], batch_size: int) -> int:
    return (len(prompts) + batch_size - 1) // batch_size


# --------------------------------------------------------------------------
# Output validation (drives resume)
# --------------------------------------------------------------------------

def is_valid_image(path: Path, min_bytes: int = 128) -> bool:
    """
    True only if the file exists and fully decodes to a non-empty image.

    A half-written JPEG from an interrupted run exists on disk but does not
    decode. Checking existence alone would skip it forever and ship a corrupt
    image to the client, so every candidate is opened and decoded.

    The byte floor only skips obviously empty files; decoding is the real
    test, since a small but valid image must still pass.
    """
    try:
        if not path.is_file() or path.stat().st_size < min_bytes:
            return False
        from PIL import Image
        with Image.open(path) as im:
            im.verify()                      # structural check
        with Image.open(path) as im:
            im.load()                        # full decode
            return im.width > 0 and im.height > 0
    except Exception:
        return False


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class BulkImageEngine:
    """Lazily loads SDXL-Lightning and renders prompts to numbered JPEGs."""

    def __init__(
        self,
        offload: bool = False,
        size: int = DEFAULT_SIZE,
        quality: int = 95,
        base_model: str = BASE_MODEL,
        lightning_repo: str = LIGHTNING_REPO,
        lightning_ckpt: str = LIGHTNING_CKPT,
    ) -> None:
        self.offload = offload
        self.size = size
        self.quality = quality
        self.base_model = base_model
        self.lightning_repo = lightning_repo
        self.lightning_ckpt = lightning_ckpt
        self._pipe = None

    # -- model ------------------------------------------------------------

    def load(self) -> None:
        if self._pipe is not None:
            return

        import torch
        from diffusers import (
            StableDiffusionXLPipeline,
            UNet2DConditionModel,
            EulerDiscreteScheduler,
        )
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not found. SDXL-Lightning needs an NVIDIA GPU.\n"
                "  - locally: install the CUDA build of torch (see README)\n"
                "  - cloud:   run on a GPU instance (g5.xlarge / RTX 4090)"
            )

        device = "cuda"
        dtype = torch.float16
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[init] loading UNet from {self.lightning_repo}")

        unet = UNet2DConditionModel.from_config(
            self.base_model, subfolder="unet"
        ).to(device, dtype)
        unet.load_state_dict(
            load_file(
                hf_hub_download(self.lightning_repo, self.lightning_ckpt),
                device=device,
            )
        )

        print(f"[init] loading pipeline from {self.base_model}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.base_model, unet=unet, torch_dtype=dtype, variant="fp16"
        )

        # SDXL-Lightning requires trailing timestep spacing; without it the
        # 4-step schedule produces washed-out images.
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()

        if self.offload:
            # Keeps peak VRAM low enough for an 8 GB card such as an RTX 3050.
            # Do not also call .to("cuda") — offload manages placement itself.
            pipe.enable_model_cpu_offload()
            print("[init] model CPU offload ON (low-VRAM mode, slower)")
        else:
            pipe.to(device)

        self._pipe = pipe
        print("[init] ready")

    # -- rendering --------------------------------------------------------

    def render(self, prompt: str, seed: int):
        import torch
        self.load()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return self._pipe(
            prompt=prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            width=self.size,
            height=self.size,
            generator=generator,
        ).images[0]

    def run_batch(
        self,
        prompts: list[Prompt],
        out_dir: Path,
        resume: bool = True,
        batch_no: int = 1,
    ) -> dict:
        """Render every prompt into out_dir as <index>.jpg. Returns a summary."""
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.jsonl"

        total = len(prompts)
        rendered = skipped = failed = 0
        failures: list[dict] = []
        t_start = time.time()

        with manifest_path.open("a", encoding="utf-8") as manifest:
            for position, p in enumerate(prompts, start=1):
                target = out_dir / f"{p.index}.jpg"

                if resume and is_valid_image(target):
                    skipped += 1
                    print(f"[{position}/{total}] Skipped (exists): {target.name}")
                    continue

                seed = seed_for(batch_no, p.index)
                snippet = p.text[:60] + ("..." if len(p.text) > 60 else "")
                t0 = time.time()
                try:
                    image = self.render(p.text, seed)
                    # Write to a temp name first: an interrupted write then
                    # leaves no half-file that resume could mistake for done.
                    tmp = target.with_suffix(".jpg.part")
                    image.convert("RGB").save(
                        tmp, "JPEG", quality=self.quality, optimize=True
                    )
                    os.replace(tmp, target)
                    elapsed = time.time() - t0
                    rendered += 1
                    print(f'[{position}/{total}] Rendered: "{snippet}" - {elapsed:.1f}s')
                    manifest.write(json.dumps({
                        **asdict(p), "batch": batch_no, "file": target.name,
                        "seed": seed, "seconds": round(elapsed, 2),
                    }) + "\n")
                    manifest.flush()
                except KeyboardInterrupt:
                    print("\n[abort] interrupted - rerun the same command to resume")
                    raise
                except Exception as exc:                     # noqa: BLE001
                    failed += 1
                    failures.append({"index": p.index, "error": str(exc)})
                    print(f'[{position}/{total}] FAILED: "{snippet}" - {exc}')

        return {
            "total": total, "rendered": rendered, "skipped": skipped,
            "failed": failed, "failures": failures,
            "seconds": round(time.time() - t_start, 1),
        }


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------

def zip_batch(out_dir: Path, zip_path: Path, expected: list[Prompt]) -> dict:
    """Zip the batch's images in numeric order. Reports anything missing."""
    present, missing = [], []
    for p in expected:
        f = out_dir / f"{p.index}.jpg"
        (present if is_valid_image(f) else missing).append(p.index)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for index in present:
            zf.write(out_dir / f"{index}.jpg", arcname=f"{index}.jpg")

    return {
        "zip": str(zip_path), "count": len(present), "missing": missing,
        "size_mb": round(zip_path.stat().st_size / 1_048_576, 1),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def bundle_zips(zip_paths: list[Path], bundle_path: Path) -> dict:
    """Wrap the per-batch archives in one shipping archive (stored, not re-deflated)."""
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_STORED) as zf:
        for z in zip_paths:
            zf.write(z, arcname=z.name)
    return {
        "zip": str(bundle_path), "count": len(zip_paths),
        "size_mb": round(bundle_path.stat().st_size / 1_048_576, 1),
    }


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Bulk SDXL-Lightning image generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes
  variants (default)  Every batch renders ALL prompts with a different seed.
                      700 prompts x 15 batches = 10,500 images. Each folder is
                      numbered 1..700 independently:
                          output/batch_1/1.jpg .. 700.jpg   -> batch_1.zip
                          output/batch_2/1.jpg .. 700.jpg   -> batch_2.zip
  split               One long prompt file is divided across batches, so each
                      prompt is rendered once. Numbering continues across
                      folders (batch_2 starts at 701.jpg).
""")
    ap.add_argument("--prompts", type=Path, default=here / "sample_prompts.txt")
    ap.add_argument("--out", type=Path, default=here / "output")
    ap.add_argument("--mode", choices=("variants", "split"), default="variants")
    ap.add_argument("--batch", type=int, default=1, help="1-based batch number")
    ap.add_argument("--batches", type=int, default=DEFAULT_BATCHES,
                    help=f"variants mode: how many batches with --all (default {DEFAULT_BATCHES})")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="split mode: prompts per batch")
    ap.add_argument("--all", action="store_true", help="run every batch in order")
    ap.add_argument("--offload", action="store_true", help="low-VRAM CPU offload")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--renumber", action="store_true",
                    help="close gaps from blank lines (default: image number = line number)")
    ap.add_argument("--no-resume", action="store_true", help="re-render existing images")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--bundle", action="store_true",
                    help="also wrap every batch zip into one shipping archive")
    args = ap.parse_args(argv)

    prompts = load_prompts(args.prompts, renumber=args.renumber)

    if args.mode == "variants":
        n_batches = args.batches
        print(f"[load] {len(prompts)} prompts from {args.prompts.name} | mode=variants "
              f"-> {n_batches} batch(es) x {len(prompts)} = "
              f"{n_batches * len(prompts)} images")
    else:
        n_batches = batch_count(prompts, args.batch_size)
        print(f"[load] {len(prompts)} prompts from {args.prompts.name} | mode=split "
              f"-> {n_batches} batch(es) of up to {args.batch_size}")

    todo = range(1, n_batches + 1) if args.all else [args.batch]
    engine = BulkImageEngine(offload=args.offload, size=args.size, quality=args.quality)
    exit_code = 0
    made_zips: list[Path] = []

    for batch_no in todo:
        if batch_no < 1 or batch_no > n_batches:
            print(f"[batch {batch_no}] outside 1..{n_batches} - skipped")
            continue

        # variants: the whole prompt list, a fresh seed per batch.
        # split:    this batch's slice of the prompt list.
        subset = prompts if args.mode == "variants" else batch_slice(
            prompts, batch_no, args.batch_size)
        if not subset:
            print(f"[batch {batch_no}] no prompts in range - nothing to do")
            continue

        out_dir = args.out / f"batch_{batch_no}"
        print(f"\n=== batch {batch_no}/{n_batches}: {len(subset)} prompts "
              f"({subset[0].index}..{subset[-1].index}) -> {out_dir} ===")

        summary = engine.run_batch(subset, out_dir,
                                   resume=not args.no_resume, batch_no=batch_no)
        print(f"[batch {batch_no}] rendered={summary['rendered']} "
              f"skipped={summary['skipped']} failed={summary['failed']} "
              f"in {summary['seconds']}s")

        if summary["failed"]:
            exit_code = 1

        if not args.no_zip:
            zip_path = args.out / f"batch_{batch_no}.zip"
            result = zip_batch(out_dir, zip_path, subset)
            made_zips.append(zip_path)
            print(f"[batch {batch_no}] {result['zip']} "
                  f"({result['count']} images, {result['size_mb']} MB)")
            if result["missing"]:
                print(f"[batch {batch_no}] WARNING missing indices: {result['missing']}")
                exit_code = 1

    if args.bundle and made_zips:
        b = bundle_zips(made_zips, args.out / "all_batches.zip")
        print(f"\n[bundle] {b['zip']} ({b['count']} archives, {b['size_mb']} MB)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
