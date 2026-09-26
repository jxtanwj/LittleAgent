"""Run every backend check in one go and summarise the result.

Four layers, because each proves something the others cannot:

  * pytest            - offline, no API key, a fraction of a second. Proves the
                        code does what the shapes in the tests say it does.
  * live pytest       - one real model call. Proves the code works against the
                        actual provider, which no fake can.
  * stream_probe      - talks to a running server over real HTTP. Proves the
                        streaming survives the transport, and shows the timing:
                        a buffering layer is invisible to every test above.
  * dump_chunks       - prints the real chunk shapes. Proves the assumptions the
                        offline tests are built on are still true. A fake model
                        cannot do this: it emits finished messages, never the
                        tool_call_chunk fragments a real provider sends.

Every check is optional and reports why it was skipped rather than failing the
run, so this stays usable on a machine with no API key and no server.

Usage:
    python -m scripts.run_all
    python -m scripts.run_all --offline-only
    python -m scripts.run_all --no-server      # skip starting uvicorn
    python -m scripts.run_all --no-color       # plain text

Colour is plain ANSI, so it needs no dependency. It turns itself off when the
output is redirected to a file or a pipe, or when NO_COLOR is set.

Exits non-zero if any check that ran failed.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
HOST = "127.0.0.1"
PORT = 8000

_ANSI = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove colour codes, so lengths are not measured including them."""
    return _ANSI.sub("", text)


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------


class Palette:
    """ANSI colour helpers that do nothing when colour is disabled.

    Every method tolerates being handed already-coloured text, which happens
    when a detail string is highlighted before it is wrapped.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def bright_red(self, text: str) -> str:
        return self._wrap("91", text)

    def status(self, state: str, text: str) -> str:
        return {"passed": self.green, "failed": self.red, "skipped": self.yellow}[state](text)


def colour_enabled(override: bool | None) -> bool:
    """Decide whether to emit colour.

    Explicit flag wins. Otherwise NO_COLOR (any value) disables, and a redirected
    stdout disables, since escape codes would end up in the file.
    """
    if override is not None:
        return override
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


# The palette actually used at runtime. Without colour every method is a no-op,
# so the call sites below need no branches.
C = Palette(False)


@dataclass
class Result:
    name: str
    status: str  # "passed" | "failed" | "skipped"
    detail: str = ""
    seconds: float = 0.0
    output: list[str] = field(default_factory=list)


def run(cmd: list[str], timeout: int) -> tuple[int, list[str], float]:
    """Run a command with stdout inherited, collecting its output lines too.

    stderr is merged into stdout so a failure's reason is visible in one place.
    """
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=BACKEND,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    lines = (proc.stdout or "").splitlines()
    for line in lines:
        print(f"    {line}")
    return proc.returncode, lines, time.perf_counter() - started


def tail(lines: list[str], count: int = 4) -> str:
    """The last few meaningful lines, for the summary.

    Colour codes are stripped first: a subprocess may emit them, and this string
    gets truncated, which would otherwise leave a half-written escape sequence
    that bleeds colour into the rest of the output.
    """
    interesting = [strip_ansi(l).strip() for l in lines if strip_ansi(l).strip()]
    return " | ".join(interesting[-count:])[:200]


def port_is_open(host: str = HOST, port: int = PORT) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def start_server() -> subprocess.Popen | None:
    """Launch uvicorn detached from this process group.

    stdout and stderr are discarded: the server would otherwise interleave its
    log with the check output, and nothing here needs to read it. A start that
    fails shows up as the port never opening.
    """
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "littleagent.api.app:app",
             "--host", HOST, "--port", str(PORT), "--log-level", "warning"],
            cwd=BACKEND,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"    could not launch uvicorn: {exc}")
        return None


def wait_for_port(seconds: float = 20.0) -> bool:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if port_is_open():
            return True
        time.sleep(0.25)
    return False


def check_offline() -> Result:
    code, lines, secs = run([sys.executable, "-m", "pytest", "-q", "--no-header"], timeout=300)
    status = "passed" if code == 0 else "failed"
    return Result("pytest (offline, no API key needed)", status, tail(lines, 2), secs, lines)


def check_live() -> Result:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return Result(
            "pytest -m live (real API)",
            "skipped",
            "DEEPSEEK_API_KEY is not set in this shell",
        )
    code, lines, secs = run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-m", "live"], timeout=300
    )
    status = "passed" if code == 0 else "failed"
    return Result("pytest -m live (real API)", status, tail(lines, 2), secs, lines)


def check_chunk_shapes() -> Result:
    """The shape dump needs a real key; without one there is nothing to learn."""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return Result(
            "dump_chunks (real chunk shapes)",
            "skipped",
            "DEEPSEEK_API_KEY is not set in this shell",
        )
    code, lines, secs = run(
        [sys.executable, "-m", "scripts.dump_chunks", "--max-blocks", "6"], timeout=180
    )
    status = "passed" if code == 0 else "failed"
    shapes = [l.strip() for l in lines if ">>>" in l]
    return Result(
        "dump_chunks (real chunk shapes)",
        status,
        " ; ".join(shapes) if shapes else tail(lines, 2),
        secs,
        lines,
    )


def check_streaming(no_server: bool) -> Result:
    name = "stream_probe (real HTTP streaming)"
    if no_server:
        return Result(name, "skipped", "--no-server was given")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        return Result(name, "skipped", "DEEPSEEK_API_KEY is not set, the agent would error")

    started_here = None
    if not port_is_open():
        print(C.dim(f"    nothing on {HOST}:{PORT}, starting uvicorn"))
        started_here = start_server()
        if started_here is None or not wait_for_port():
            if started_here is not None:
                started_here.terminate()
            return Result(name, "skipped", f"could not start a server on {HOST}:{PORT}")
        print(C.dim(f"    server is up on {HOST}:{PORT}"))
    else:
        print(C.dim(f"    reusing the server already listening on {HOST}:{PORT}"))

    try:
        code, lines, secs = run(
            [sys.executable, "-m", "scripts.stream_probe", "--thread-id", "run-all"],
            timeout=180,
        )
    finally:
        if started_here is not None:
            started_here.terminate()
            try:
                started_here.wait(timeout=10)
                print(C.dim("    stopped the server started by this run"))
            except subprocess.TimeoutExpired:
                started_here.kill()

    status = "passed" if code == 0 else "failed"
    counts = [l.strip() for l in lines if l.strip().startswith("events")]
    verdicts = [l.strip() for l in lines if ">>>" in l]
    detail = " ; ".join([*counts, *verdicts]) or tail(lines, 2)
    return Result(name, status, detail, secs, lines)


def main() -> int:
    global C

    # Model output can contain any Unicode, including emoji, which a legacy
    # Windows code page cannot encode. Without this the run dies mid-check on a
    # character instead of reporting a result.
    for handle in (sys.stdout, sys.stderr):
        if hasattr(handle, "reconfigure"):
            handle.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-only", action="store_true",
                        help="run only the fast offline suite")
    parser.add_argument("--no-server", action="store_true",
                        help="never start uvicorn; only probe one already running")
    parser.add_argument("--color", dest="color", action="store_true", default=None,
                        help="force colour even when piping")
    parser.add_argument("--no-color", dest="color", action="store_false",
                        help="disable colour")
    args = parser.parse_args()

    C = Palette(colour_enabled(args.color))

    print(C.bold(f"backend: {BACKEND}"))
    print(C.dim(f"python : {sys.executable}"))
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        print(f"api key: {C.green(f'set ({len(key)} chars)')}")
    else:
        print(f"api key: {C.yellow('NOT set')} {C.dim('- live checks will be skipped')}")
    print()

    # Only the checks that will actually run are counted, so the labels stay
    # sequential when --offline-only drops the middle two.
    planned = [("offline test suite", check_offline)]
    if not args.offline_only:
        planned += [
            ("live API test", check_live),
            ("chunk shapes", check_chunk_shapes),
        ]
    planned.append(("streaming over HTTP", lambda: check_streaming(args.no_server)))

    total = len(planned)
    results: list[Result] = []
    for index, (title, fn) in enumerate(planned, start=1):
        print(C.cyan(C.bold(f"[{index}/{total}] {title}")))
        try:
            result = fn()
        except subprocess.TimeoutExpired:
            result = Result(title, "failed", "timed out")
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the rest
            result = Result(title, "failed", f"{type(exc).__name__}: {exc}")
        results.append(result)

        marker = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}[result.status]
        suffix = C.dim(f" ({result.seconds:.1f}s)") if result.seconds else ""
        detail = C.status(result.status, result.detail) if result.status != "passed" else result.detail
        print(f"  -> {C.status(result.status, marker)}{suffix} {detail}")
        print()

    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]

    print(C.dim("=" * 68))
    summary = (
        f"{C.green(str(len(passed)) + ' passed')}, "
        f"{C.red(str(len(failed)) + ' failed') if failed else '0 failed'}, "
        f"{C.yellow(str(len(skipped)) + ' skipped') if skipped else '0 skipped'}"
    )
    print(C.bold(summary))
    for result in failed:
        print(f"  {C.red('FAILED ')} {C.bold(result.name)}")
        print(f"          {result.detail}")
    for result in skipped:
        print(f"  {C.yellow('SKIPPED')} {result.name}")
        print(C.dim(f"          {result.detail}"))
    if not failed:
        print()
        print(C.dim("A green offline run only means the code matches the shapes the tests"))
        print(C.dim("assume. The live, probe and dump checks are what confirm those shapes"))
        print(C.dim("still match reality."))
    print(C.dim("=" * 68))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
