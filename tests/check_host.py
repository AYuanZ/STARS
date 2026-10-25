#!/usr/bin/env python3
"""Host integration checks. Does not claim GPU execution coverage."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=10, **kwargs)


def wait_for(predicate, process):
    deadline = time.monotonic() + 5
    while not predicate():
        if process.poll() is not None:
            raise AssertionError(f"scheduler exited with {process.returncode}")
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for scheduler")
        time.sleep(0.02)


def main():
    cli = str(ROOT / "build/starsctl")
    run(cli, "--help")
    run("/bin/true", env={**os.environ, "LD_PRELOAD": str(ROOT / "build/libstars.so")})
    with tempfile.TemporaryDirectory(prefix="stars-check-") as directory:
        directory = Path(directory)
        env = {**os.environ, "STARS_SOCKET_DIR": str(directory / "socket")}
        log_path = directory / "scheduler.log"
        with log_path.open("w") as log:
            process = subprocess.Popen([str(ROOT / "build/stars-scheduler")], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_for(lambda: (directory / "socket/scheduler.sock").exists(), process)
                for arguments, expected in [
                    (("--anti-thrash=off",), "Scheduler turned OFF"),
                    (("--anti-thrash=on",), "Scheduler turned ON"),
                    (("--set-tq=2",), "New TQ = 2"),
                ]:
                    run(cli, *arguments, env=env)
                    wait_for(lambda: expected in log_path.read_text(), process)
                print("PASS: scheduler startup and CLI wire protocol (off/on/time quantum)")
            except Exception:
                print(log_path.read_text())
                raise
            finally:
                process.terminate()
                process.wait(timeout=5)
        bad_env = {**env, "STARS_SOCKET_DIR": "/" + "x" * 200}
        result = subprocess.run([cli, "--set-tq=2"], env=bad_env,
                                capture_output=True, timeout=5)
        assert result.returncode != 0, "overlong socket paths must fail"

    spec = importlib.util.spec_from_file_location("baseline", ROOT / "src/profiling/baseline.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    for mode in ("NV", "MPSE"):
        config = ROOT / f"configs/{mode}.json"
        loaded = baseline.load_config(config)
        assert loaded["mode"] == mode
        run("python3", str(ROOT / "src/profiling/baseline.py"), "--config", str(config), "--dry-run")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.json"
        loaded["clients"] = 0
        path.write_text(json.dumps(loaded))
        try:
            baseline.load_config(path)
        except ValueError:
            pass
        else:
            raise AssertionError("zero-client configuration must fail")
    print("PASS: preload loading, path bounds, NV/MPSE configuration validation")
    print("GPU execution requires a working NVIDIA driver; it is not covered by make check.")


if __name__ == "__main__":
    main()
