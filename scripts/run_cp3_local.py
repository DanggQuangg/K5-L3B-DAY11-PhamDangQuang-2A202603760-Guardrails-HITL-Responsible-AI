"""Start the portable llama.cpp server and run Checkpoint 3 against it."""
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.openai_runtime import LOCAL_BLUE_BASE_URL, LOCAL_BLUE_MODEL


def main() -> int:
    model_dir = ROOT / ".local-model"
    binary = model_dir / "runtime" / "llama-server.exe"
    model = model_dir / "LFM3-1.7B-Q4_K_M.gguf"
    for path in (binary, model):
        if not path.is_file():
            raise SystemExit(f"Missing {path.relative_to(ROOT)}. See README.md.")

    endpoint = urlsplit(LOCAL_BLUE_BASE_URL)
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex((endpoint.hostname, endpoint.port)) == 0:
            raise SystemExit(f"Local model port {endpoint.port} is already in use.")

    health_url = f"{endpoint.scheme}://{endpoint.netloc}/health"
    log_path = model_dir / "llama-server.log"
    print(f"Starting {LOCAL_BLUE_MODEL} at {endpoint.netloc}...", flush=True)
    with log_path.open("wb") as log, subprocess.Popen(
        [
            str(binary), "--model", str(model), "--alias", LOCAL_BLUE_MODEL,
            "--host", endpoint.hostname, "--port", str(endpoint.port),
            "--ctx-size", "4096", "--parallel", "1", "--reasoning", "off",
            "--no-webui",
        ],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ) as server:
        try:
            deadline = monotonic() + 90
            while server.poll() is None and monotonic() < deadline:
                try:
                    with urlopen(health_url, timeout=2):
                        break
                except (URLError, TimeoutError):
                    sleep(1)
            else:
                raise SystemExit(f"Local model server failed to start. See {log_path}.")

            return subprocess.run(
                [sys.executable, "-X", "utf8", str(ROOT / "src" / "main.py"),
                 "--part", "3", "--local"],
                cwd=ROOT,
                check=False,
            ).returncode
        finally:
            if server.poll() is None:
                server.terminate()


if __name__ == "__main__":
    sys.exit(main())
