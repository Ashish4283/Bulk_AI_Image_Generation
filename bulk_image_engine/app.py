"""
Web UI for the bulk image engine.

    pip install -r requirements-ui.txt
    python app.py                      # then open http://127.0.0.1:8000

The UI is a thin wrapper: it launches `engine.py` as a subprocess and reads
its output. There is one engine, one set of rules, and the UI cannot drift
from what the CLI does.

Binds to 127.0.0.1 by default. On a cloud GPU, reach it over an SSH tunnel
rather than opening a port:

    ssh -L 8000:localhost:8000 ubuntu@<instance-ip>
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from collections import deque
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel
import uvicorn

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
STOP_FILE = HERE / "STOP"

app = FastAPI(title="Bulk Image Engine")

# --- progress line formats emitted by engine.py -----------------------------
RE_IMAGE = re.compile(r"^\[(\d+)/(\d+)\]\s+(Rendered|Skipped|FAILED)")
RE_BATCH = re.compile(r"^=== batch (\d+)/(\d+):")
RE_ZIP = re.compile(r"^\[batch (\d+)\].*\.zip")


class Job:
    """Tracks the single running render job."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.log: deque[str] = deque(maxlen=400)
        self.batch = 0
        self.batches = 0
        self.done = 0
        self.total = 0
        self.rendered = 0
        self.failed = 0
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.exit_code: int | None = None
        self.lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def reset(self) -> None:
        self.log.clear()
        self.batch = self.batches = self.done = self.total = 0
        self.rendered = self.failed = 0
        self.started_at = time.time()
        self.finished_at = None
        self.exit_code = None

    def _absorb(self, line: str) -> None:
        with self.lock:
            self.log.append(line)
            if m := RE_BATCH.match(line):
                self.batch, self.batches = int(m.group(1)), int(m.group(2))
                self.done = self.total = 0
            elif m := RE_IMAGE.match(line):
                self.done, self.total = int(m.group(1)), int(m.group(2))
                if m.group(3) == "Rendered":
                    self.rendered += 1
                elif m.group(3) == "FAILED":
                    self.failed += 1

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            self._absorb(raw.rstrip("\n"))
        self.proc.wait()
        self.exit_code = self.proc.returncode
        self.finished_at = time.time()
        self._absorb(f"--- process exited with code {self.exit_code} ---")

    def start(self, argv: list[str]) -> None:
        if self.running:
            raise HTTPException(409, "a job is already running")
        if STOP_FILE.exists():
            STOP_FILE.unlink()
        self.reset()
        self._absorb("$ python engine.py " + " ".join(argv))
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(HERE / "engine.py"), *argv],
            cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def stop(self) -> None:
        # Cooperative: the engine finishes the current image, then exits, so
        # nothing is left half-written. Killing the process would be faster
        # and is exactly what we do not want.
        STOP_FILE.write_text("")
        self._absorb("--- stop requested; finishing current image ---")


JOB = Job()


class StartRequest(BaseModel):
    prompts: str = "sample_prompts.csv"
    mode: str = "variants"
    batches: int = 3
    size: int = 1024
    quality: int = 95
    offload: bool = False
    bundle: bool = True
    stop_after: int | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/prompt-files")
def prompt_files() -> dict:
    # Exclude requirements*.txt — they are dependency lists, not prompts.
    files = sorted(
        p.name for p in HERE.iterdir()
        if p.suffix.lower() in (".txt", ".csv")
        and not p.name.lower().startswith("requirements")
    )
    return {"files": files}


@app.post("/api/start")
def start(req: StartRequest) -> dict:
    src = HERE / req.prompts
    if not src.is_file():
        raise HTTPException(400, f"prompt file not found: {req.prompts}")

    argv = [
        "--prompts", str(src), "--mode", req.mode, "--all",
        "--batches", str(req.batches), "--size", str(req.size),
        "--quality", str(req.quality), "--stop-file", str(STOP_FILE),
    ]
    if req.offload:
        argv.append("--offload")
    if req.bundle:
        argv.append("--bundle")
    if req.stop_after:
        argv += ["--stop-after", str(req.stop_after)]

    JOB.start(argv)
    return {"ok": True}


@app.post("/api/stop")
def stop() -> dict:
    if not JOB.running:
        raise HTTPException(409, "nothing is running")
    JOB.stop()
    return {"ok": True}


@app.get("/api/status")
def status() -> dict:
    with JOB.lock:
        elapsed = (JOB.finished_at or time.time()) - (JOB.started_at or time.time())
        rate = JOB.rendered / elapsed if elapsed > 0 and JOB.rendered else 0.0
        remaining = max(JOB.total - JOB.done, 0)
        batches_left = max(JOB.batches - JOB.batch, 0)
        eta = ((remaining + batches_left * JOB.total) / rate) if rate else None
        return {
            "running": JOB.running,
            "batch": JOB.batch, "batches": JOB.batches,
            "done": JOB.done, "total": JOB.total,
            "rendered": JOB.rendered, "failed": JOB.failed,
            "elapsed": round(elapsed, 1),
            "per_image": round(1 / rate, 2) if rate else None,
            "eta": round(eta) if eta else None,
            "exit_code": JOB.exit_code,
            "stopping": STOP_FILE.exists(),
            "log": list(JOB.log)[-120:],
        }


@app.get("/api/gallery")
def gallery(limit: int = 12) -> dict:
    """Newest images across all batch folders."""
    files = sorted(
        (p for p in OUT.glob("batch_*/*.jpg")),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )[:limit]
    return {"images": [f"{p.parent.name}/{p.name}" for p in files]}


@app.get("/api/thumb/{batch}/{name}")
def thumb(batch: str, name: str) -> Response:
    path = (OUT / batch / name).resolve()
    if OUT.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "not found")
    from PIL import Image
    with Image.open(path) as im:
        im.thumbnail((320, 320))
        buf = BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=80)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.get("/api/zips")
def zips() -> dict:
    if not OUT.is_dir():
        return {"zips": []}
    return {"zips": [
        {"name": z.name, "size_mb": round(z.stat().st_size / 1_048_576, 1)}
        for z in sorted(OUT.glob("*.zip"))
    ]}


@app.get("/api/download/{name}")
def download(name: str) -> FileResponse:
    path = (OUT / name).resolve()
    if OUT.resolve() != path.parent or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, filename=path.name, media_type="application/zip")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"UI at http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
