# bulk_image_engine

Batch image generation with **SDXL-Lightning 4-step** on SDXL base 1.0.

Takes a plain-text file of prompts (one per line), renders each to a numbered
JPEG (`1.jpg`, `2.jpg`, …), and packages each batch as `batch_<n>.zip`.

Self-contained: everything lives in this folder. It imports nothing from, and
writes nothing outside, this directory.

---

## Quick check (5 prompts)

```bash
python run_test.py            # add --offload on a low-VRAM GPU
```

Renders the 5 sample prompts, verifies `1.jpg`..`5.jpg` decode, and writes
`output/test_batch.zip`. Exits non-zero if anything is missing or corrupt.

## Full runs

```bash
# one batch of 700
python engine.py --prompts prompts.txt --batch 1

# every batch in the file, back to back
python engine.py --prompts prompts.txt --all --batch-size 700

# resume after an interruption — same command, finished images are skipped
python engine.py --prompts prompts.txt --batch 1
```

| Flag | Purpose |
|---|---|
| `--prompts` | Input file (default `sample_prompts.txt`) |
| `--batch N` | Render 1-based batch N |
| `--batch-size` | Prompts per batch (default 700) |
| `--all` | Every batch in order |
| `--offload` | CPU offload for low-VRAM GPUs |
| `--size` | Square edge in px (default 1024) |
| `--quality` | JPEG quality (default 95) |
| `--renumber` | Close gaps left by blank lines |
| `--no-resume` | Re-render images that already exist |
| `--no-zip` | Skip packaging |

Output layout:

```
output/
  batch_1/            1.jpg, 2.jpg, …, manifest.jsonl
  batch_1.zip
```

`manifest.jsonl` records index, source line, seed and render time per image —
so any image can be traced back to its prompt and reproduced exactly.

---

## Install

### Local (Windows, NVIDIA GPU)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python run_test.py --offload
```

> Install torch from the CUDA index **first**. Plain `pip install torch` on
> Windows installs the CPU-only build, which cannot run this pipeline.

**RTX 3050 (8 GB):** always use `--offload`. Expect roughly 8–20 s per image —
fine for verifying correctness, not for volume. A 4 GB 3050 will likely still
run out of memory at 1024 px; use `--size 768` to check the pipeline works.

### Cloud GPU (AWS g5.xlarge / RTX 4090)

```bash
sudo apt update && sudo apt install -y python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python run_test.py                       # no --offload: full speed
python engine.py --prompts prompts.txt --all --batch-size 700
```

Use a **Deep Learning AMI** so the NVIDIA driver and CUDA are preinstalled.

Keep runs alive across SSH drops:

```bash
tmux new -s render
# Ctrl-B then D to detach, `tmux attach -t render` to return
```

First run downloads ~7 GB of weights to `~/.cache/huggingface`. Put that cache
on the instance's own NVMe, not an EBS root volume, if you rebuild often.

---

## Throughput

| GPU | Approx. per image @1024px | 700 images | 10,000 images |
|---|---|---|---|
| RTX 4090 | 0.6–1.0 s | 7–12 min | 2–3 h |
| A10G (g5.xlarge) | 1.5–2.5 s | 18–30 min | 4–7 h |
| RTX 3050, `--offload` | 8–20 s | 1.5–4 h | not practical |

10,000 images/day is comfortable on a single A10G or 4090. Measure with
`run_test.py` on the actual instance before committing to a schedule.

---

## Design notes

**Numbering.** By default an image's number is its **line number** in the
prompt file, so line 12 always yields `12.jpg`. A blank line leaves a gap
rather than silently shifting every later image. Pass `--renumber` for an
unbroken 1..N sequence. Confirm which the client wants — if their prompt file
has no blank lines, the two are identical.

**Resume.** Every existing image is opened and decoded before being skipped.
A truncated JPEG from a killed process fails that check and is re-rendered,
so an interrupted run never ships a corrupt file. Images are written to
`.jpg.part` and atomically renamed, so a partial write cannot be mistaken for
a finished image.

**Reproducibility.** Seed = `1_000_000 + index`, so re-running any index
reproduces the same image on the same GPU and library versions.

**Scheduler.** SDXL-Lightning needs
`EulerDiscreteScheduler(timestep_spacing="trailing")` with
`guidance_scale=0.0`. Non-zero guidance or default spacing produces washed-out
results at 4 steps.

---

## Licensing

SDXL 1.0 is released under CreativeML OpenRAIL++-M and SDXL-Lightning under
OpenRAIL++, both of which permit commercial use subject to their use
restrictions. Confirm the client's intended use complies, and agree who is
responsible for content review — this pipeline runs no safety filter.
