# RUNBOOK — Bulk Image Engine

Everything needed to run this project, and the standard operating procedure
for a production day.

- **Code:** `C:\Projects\bulk-image-bid\bulk_image_engine\`
- **Repo:** https://github.com/Ashish4283/Bulk_AI_Image_Generation
- **Deliberately separate** from the ClinicalDataViewer medical project. Nothing
  here reads from or writes to it.

---

## 1. What this does

Prompt file in → numbered images out → one zip per batch.

```
prompts.csv  (700 prompts)
      │
      ▼
  engine.py  ── SDXL-Lightning 4-step, fp16, CUDA
      │
      ▼
output/batch_1/  1.jpg … 700.jpg  →  batch_1.zip
output/batch_2/  1.jpg … 700.jpg  →  batch_2.zip
      …                              all_batches.zip
```

**Two modes:**

| Mode | Behaviour | Use when |
|---|---|---|
| `variants` (default) | Every batch renders **all** prompts with a different seed. 700 × 15 = 10,500 images. Each folder numbered from 1. | The client sends one prompt file and wants N variations |
| `split` | Prompts divided across batches; each rendered once. Numbering continues across folders. | The client sends 10,000 distinct prompts |

---

## 2. Start here (local machine)

### First time only

```powershell
cd C:\Projects\bulk-image-bid\bulk_image_engine
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r requirements-ui.txt      # optional web UI
```

> Install torch from the CUDA index **first**. Plain `pip install torch` on
> Windows gives the CPU build, which cannot run this pipeline.

### Every session

```powershell
cd C:\Projects\bulk-image-bid\bulk_image_engine
```

If a command is "not recognized" after any install, refresh PATH:

```powershell
$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
```

### Verify the GPU before a long run

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Must print a GPU table and `True`. If not, see §7.

---

## 3. Running a job

### Smoke test — 10 prompts × 3 batches = 30 images

```powershell
python run_test.py                 # add --offload on a low-VRAM GPU
```

Asserts each folder holds `1.jpg`..`10.jpg`, that the same prompt differs
between batches, that each zip is complete, and that `all_batches.zip` holds
all three archives. Non-zero exit on any failure.

### Production

```powershell
# 15 batches of the full prompt file, zipped, plus a shipping bundle
python engine.py --prompts prompts.csv --all --batches 15 --bundle

# one batch only
python engine.py --prompts prompts.csv --batch 3

# low-VRAM GPU (under ~8 GB)
python engine.py --prompts prompts.csv --all --offload --size 768
```

### Web UI

```powershell
python app.py           # http://127.0.0.1:8000
```

Start/Stop buttons, live progress and ETA, thumbnail gallery, download links.
On a cloud GPU reach it through an SSH tunnel, never an open port:

```bash
ssh -L 8000:localhost:8000 ubuntu@<instance-ip>
```

### Stopping and resuming — always safe

```powershell
New-Item STOP           # finishes the current image, then exits (code 2)
# or press Ctrl+C
```

Resume by rerunning the **same command**. Finished images are decoded before
being skipped, so a truncated file from a crash is re-rendered rather than
delivered. Images are written to `.jpg.part` and renamed only when complete.

---

## 4. AWS GPU — open and close

> **Status: written but NOT yet tested.** The account's GPU quota is still 0
> (case under EC2 service-team review). Nothing below can run until it is
> approved. Check with:
> `aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --profile ashish-admin --region us-east-1`

### Sign in first

```powershell
aws sso login --profile ashish-admin
```

Tokens last ~8 hours. "Token has expired" just means run this again.

### The safe pattern: self-terminating instance

The instance is told to **terminate itself when the job finishes**. Combined
with `--instance-initiated-shutdown-behavior terminate`, a finished or failed
run cannot leave a GPU billing overnight.

```powershell
.\scripts\gpu.ps1 launch -Batches 15 -Bucket my-output-bucket   # open
.\scripts\gpu.ps1 status                                        # watch
.\scripts\gpu.ps1 terminate                                     # close, manually
```

`launch` does: find a GPU AMI → create key pair and security group if missing
→ start `g5.xlarge` → install deps → clone the repo → render → upload zips to
S3 → self-terminate.

### Manual control

```powershell
aws ec2 stop-instances      --instance-ids i-xxxx --profile ashish-admin
aws ec2 start-instances     --instance-ids i-xxxx --profile ashish-admin
aws ec2 terminate-instances --instance-ids i-xxxx --profile ashish-admin
```

| Action | Keeps disk? | Still billing? |
|---|---|---|
| **stop** | Yes, including the 7 GB model cache | EBS only, pennies/day |
| **terminate** | No, everything deleted | Nothing |

**Stop** between runs on the same day. **Terminate** when finished.

### Cost discipline — non-negotiable

| | |
|---|---|
| g5.xlarge on-demand | ~$1.01/hr |
| g5.xlarge spot | ~$0.35/hr |
| 700 images (~25 min) | ~$0.40 |
| 10,500 images (~5 h) | ~$6 |
| **Idle instance forgotten overnight** | **~$24** |

Always finish a session with:

```powershell
aws ec2 describe-instances --profile ashish-admin --region us-east-1 `
  --filters "Name=instance-state-name,Values=running" `
  --query "Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]" --output table
```

Empty output = nothing billing. The zero-spend budget alert emails you, but
checking is the habit that matters.

---

## 5. Delivering to the client

1. Confirm counts before sending — the run reports any missing index, and a
   short batch is never zipped silently.
2. A day's output is roughly **3 GB** at 1024px JPEG. Use Google Drive,
   Dropbox or an S3 pre-signed link; never email.
3. Send `batch_N.zip` per batch, or `all_batches.zip` for a whole day.
4. For a quality preview, a contact sheet beats a zip — one image, instantly
   viewable, shows the full range.

**Known limitation to disclose up front:** SDXL does not render legible text.
Signage and lettering come out as approximate shapes. Raise it before the
client finds it.

---

## 6. Standard operating procedure — a production day

| # | Step | Command / check |
|---|---|---|
| 1 | Receive the prompt file | `.txt` or `.csv`, confirm line count |
| 2 | Confirm mode | `variants` or `split` — ask if unsure |
| 3 | Smoke test 10 prompts | `python engine.py --prompts client.csv --batch 1 --stop-after 10` |
| 4 | Eyeball the output | Quality, ordering, aspect ratio |
| 5 | Open the GPU | `.\scripts\gpu.ps1 launch -Batches 15` |
| 6 | Monitor | UI, or `.\scripts\gpu.ps1 status` |
| 7 | Verify | Every batch reports `failed=0`, no missing indices |
| 8 | Package and deliver | `batch_*.zip` + contact sheet |
| 9 | **Close the GPU** | `.\scripts\gpu.ps1 terminate`, then confirm nothing is running |
| 10 | Record | Images rendered, GPU hours, cost, issues |

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'torch'` | Deps not installed | §2 first-time install |
| `cuda available: False` | GPU disabled or driver state | §7.1 below |
| `[fatal] model could not be loaded` | Same as above — fails once, not per prompt | Fix the GPU, rerun |
| CUDA out of memory | VRAM too small | Add `--offload`, and `--size 768` |
| `VcpuLimitExceeded` | GPU quota still 0 | Wait for the AWS case |
| `Token has expired` | SSO token >8 h old | `aws sso login --profile ashish-admin` |
| `aws`/`terraform` not recognized | PATH stale in this terminal | Refresh PATH (§2) |
| Run stopped instantly | Stale `STOP` file | It is auto-cleared at startup; delete if odd |
| Output folder empty | Model never loaded | Check the log for `[fatal]` |

### 7.1 GPU not visible (ASUS laptops)

Order matters:

1. **Armoury Crate → Devices → System Settings → GPU Performance → GPU Mode → Standard.**
   *Eco Mode removes the dGPU from Windows entirely.*
2. **Restart** — use Start → Power → **Restart**, not Shut down. Shut down
   does a Fast Startup hibernate-resume and does not reinitialise drivers.
3. Verify: `Get-CimInstance Win32_VideoController` should list the NVIDIA
   adapter with `ConfigManagerErrorCode 0`.

Error codes: `43` = device stopped, needs a true restart. dGPU **absent** =
Eco Mode. `nvidia-smi` "insufficient permissions" is the symptom of both — it
is *not* an admin problem.

---

## 8. Reusable pattern for the next project

What worked here, worth repeating:

1. **Isolate completely.** New folder outside every existing repo, its own git
   repo, its own dependencies. Never add a client project inside another.
2. **Test the logic before the expensive part.** Prompt parsing, batching and
   packaging were all verified on CPU before a GPU existed — and that testing
   caught a real bug.
3. **Then test on real hardware anyway.** Running on an actual GPU caught two
   more bugs that no amount of CPU testing could: VRAM exhaustion before
   offload engaged, and a removed diffusers API.
4. **Fail fast and loudly.** Load the model once up front. One clear error
   beats 700 identical ones on a metered GPU.
5. **Verify, don't assume.** Decode images before skipping them; check counts
   before zipping; confirm teardown with a second command rather than trusting
   "done".
6. **Find the ambiguity early.** "10,000 images from 700 prompts" had two
   readings worth a lot of wasted work. Asking beat guessing.
7. **Prove it with output.** 30 real images and a contact sheet were worth
   more in the bid than any description of the architecture.
