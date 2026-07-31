"""
tools/capture_ui.py -- drive the Streamlit app in headless Chromium.

Does two jobs that both need a real browser:

  A. FIGURES. Streamlit renders dataframes to a <canvas>, so a screenshot is
     the only way to actually see them -- no DOM inspection can substitute.
     Produces the images used by docs/.

  B. THREADING PROOF ON THE REAL PATH. The pyo3 interpreter is `unsendable`
     and Streamlit gives every rerun a FRESH ScriptRunner thread, so the
     failure mode is sequential reruns, not concurrency. Synthetic thread pools
     cannot reproduce it. This exercises:
       - 10 consecutive Run clicks (10 reruns, 10 potential threads)
       - page.reload()            -> a brand-new Streamlit session
       - a second browser context -> two sessions live at once
     asserting the exact energy is bit-identical every time and that the server
     log contains no `sent to another thread` panic.

  C. COLD-START TIMING measured through the browser (goto -> Section 1 painted),
     not inferred from an in-app timer.

Runs its own server on a dedicated port so it cannot collide with a manually
started app.

Usage::

    python tools/capture_ui.py
    python tools/capture_ui.py --keep-server   # leave it running afterwards
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = PROJECT_ROOT / ".python" / "python.exe"
PORT = 8599
BASE_URL = f"http://localhost:{PORT}"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
SERVER_LOG = LOGS_DIR / "streamlit_capture.log"
REPORT = LOGS_DIR / "ui_capture_report.json"

VIEWPORT = {"width": 1440, "height": 6000}

#: The operational pyo3 panic we must never see.
PANIC_PATTERN = re.compile(r"unsendable, but sent to another thread", re.I)
#: The benign teardown message (fires at process exit, affects no result).
TEARDOWN_PATTERN = re.compile(r"unsendable, but is being dropped", re.I)

PASSED = True


def check(name: str, ok: bool, detail: str) -> None:
    global PASSED
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        PASSED = False


# --------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------

def start_server() -> subprocess.Popen:
    """Launch Streamlit on the dedicated port, logging to SERVER_LOG."""
    LOGS_DIR.mkdir(exist_ok=True)
    log = open(SERVER_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(PYTHON), "-m", "streamlit", "run", "ui/app.py",
         "--server.port", str(PORT),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false",
         "--server.fileWatcherType", "none"],
        cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Streamlit exited early (code {proc.returncode}); see {SERVER_LOG}"
            )
        try:
            with urllib.request.urlopen(f"{BASE_URL}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    print(f"  server ready on {BASE_URL}")
                    return proc
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("Streamlit did not become healthy within 90 s")


def read_server_log() -> str:
    return SERVER_LOG.read_text(encoding="utf-8", errors="replace") if SERVER_LOG.exists() else ""


#: Directories whose appearance in a traceback means OUR code failed.
PROJECT_MARKERS = ("ui\\app.py", "ui/app.py", "quantum", "classical", "training", "tools")

#: Library-level failures that are dangerous even with NO project frames.
#:
#: Frame-based classification alone has a hole: the pyo3 `unsendable` panic
#: originates entirely inside qdk and never touches project source, so a purely
#: structural rule would file the exact failure this project spent a phase
#: eliminating as harmless transport noise. These signatures override the frame
#: check in both directions.
DANGEROUS_SIGNATURES = (
    "unsendable, but sent to another thread",   # the pyo3 panic
    "PanicException",                           # any Rust panic surfaced to Python
    "pyo3_runtime",
    "SIGSEGV",
    "Fatal Python error",
)

#: Known-benign, harness-induced transport teardown. Narrow on purpose.
BENIGN_TRANSPORT = (
    "_call_connection_lost",
    "WinError 10054",
    "ConnectionResetError",
)


def classify_tracebacks(log: str) -> tuple[list[str], list[str]]:
    """Split server-log tracebacks into application failures and transport noise.

    Not every traceback in the log is a defect. This harness opens and closes
    several browser contexts (five for the cold-start samples, plus a second
    session), and each close tears down a websocket. On Windows, asyncio's
    proactor event loop logs the resulting ``ConnectionResetError``
    ``[WinError 10054]`` from ``_call_connection_lost`` -- caused by the harness,
    not the app.

    Classification, in priority order:

    1. Any ``DANGEROUS_SIGNATURES`` match is an APPLICATION failure regardless
       of which frames appear. This exists specifically so the pyo3 panic, whose
       frames live entirely inside qdk, can never be filed as noise.
    2. Otherwise a traceback referencing project source is an application
       failure.
    3. Otherwise, only if it matches a known benign transport signature, it is
       noise.
    4. Anything else is treated as an application failure -- unknown tracebacks
       fail the gate rather than being waved through.

    Returns:
        ``(application_tracebacks, transport_tracebacks)``.
    """
    blocks = log.split("Traceback (most recent call last)")[1:]
    app: list[str] = []
    transport: list[str] = []
    for block in blocks:
        # A traceback ends where the next log timestamp begins; 40 lines is a
        # generous bound that never runs into the following record.
        body = "\n".join(block.splitlines()[:40]).strip()

        if any(sig in body for sig in DANGEROUS_SIGNATURES):
            app.append(body)
        elif any(marker in body for marker in PROJECT_MARKERS):
            app.append(body)
        elif any(sig in body for sig in BENIGN_TRANSPORT):
            transport.append(body)
        else:
            # Default to failing: an unrecognised traceback is not evidence of
            # health.
            app.append(body)
    return app, transport


# --------------------------------------------------------------------------
# Page helpers
# --------------------------------------------------------------------------

def metrics(page) -> dict[str, str]:
    """Map every rendered st.metric label to its displayed value."""
    out: dict[str, str] = {}
    blocks = page.locator('[data-testid="stMetric"]')
    for i in range(blocks.count()):
        block = blocks.nth(i)
        try:
            label = block.locator('[data-testid="stMetricLabel"]').inner_text(timeout=3000)
            value = block.locator('[data-testid="stMetricValue"]').inner_text(timeout=3000)
        except Exception:
            continue
        out[label.strip()] = value.strip()
    return out


def click_run(page, settle_ms: int = 900) -> dict[str, str]:
    """Press Run Quantum Optimization and wait for the metrics to render."""
    page.get_by_role("button", name="Run Quantum Optimization").click()
    page.wait_for_selector('[data-testid="stMetric"]', timeout=60_000)
    # A rerun replaces the DOM; give Streamlit a moment to finish painting so
    # we never read a half-updated metric row.
    page.wait_for_timeout(settle_ms)
    return metrics(page)


def energy_of(m: dict[str, str]) -> str | None:
    for label, value in m.items():
        if "Circuit energy" in label:
            return value
    return None


def heading_offsets(page) -> list[dict]:
    """Absolute y-position and height of every h2, in page coordinates.

    Streamlit does NOT wrap each section in its own container: an ``h2`` is a
    direct child of the single outermost ``stVerticalBlock``, so
    ``ancestor::stVerticalBlock`` selects the whole page. Sections are therefore
    delimited by where the NEXT heading starts.

    The viewport is deliberately taller than the document, so nothing scrolls
    and ``getBoundingClientRect().top`` is already a page coordinate.
    """
    return page.evaluate(
        """() => {
            const scroller = document.scrollingElement || document.documentElement;
            const dy = scroller.scrollTop || 0;
            return [...document.querySelectorAll('h2')].map(h => ({
                text: h.innerText.trim(),
                y: h.getBoundingClientRect().top + dy,
            }));
        }"""
    )


def freeze_animations(page) -> None:
    """Disable Streamlit's entrance transitions before screenshotting.

    Streamlit fades elements in on mount. Capturing mid-transition yields
    washed-out, partially transparent figures, so transitions are switched off
    and the fade wrappers forced to full opacity. Scoped to Streamlit's own
    containers rather than ``*`` so nothing intentionally hidden is revealed.
    """
    page.add_style_tag(content="""
        *, *::before, *::after {
            transition: none !important;
            animation: none !important;
        }
        [data-testid="stElementContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="stMetric"] { opacity: 1 !important; }
    """)
    page.wait_for_timeout(300)


def content_bottom(page) -> float:
    """Absolute y of the lowest rendered element, ignoring trailing padding.

    ``scrollHeight`` includes Streamlit's bottom padding, which for the LAST
    section would append a screenful of blank space to the figure.
    """
    return page.evaluate(
        """() => {
            const dy = (document.scrollingElement || document.documentElement).scrollTop || 0;
            const els = [...document.querySelectorAll('[data-testid="stElementContainer"]')];
            if (!els.length) return (document.scrollingElement
                                     || document.documentElement).scrollHeight;
            return Math.max(...els.map(e => e.getBoundingClientRect().bottom + dy));
        }"""
    )


def shoot_section(page, heading: str, path: Path) -> bool:
    """Screenshot from ``heading`` down to just above the following h2.

    The clip rectangle is derived from element geometry, never from hardcoded
    pixel offsets, so it survives content changes.
    """
    page.evaluate(
        "(document.scrollingElement || document.documentElement).scrollTop = 0"
    )
    page.wait_for_timeout(150)

    offsets = heading_offsets(page)
    idx = next((i for i, h in enumerate(offsets) if heading in h["text"]), None)
    if idx is None:
        return False

    top = max(offsets[idx]["y"] - 24, 0)
    if idx + 1 < len(offsets):
        bottom = offsets[idx + 1]["y"] - 12
    else:
        bottom = content_bottom(page) + 16
    width = page.evaluate("document.documentElement.clientWidth")

    height = max(bottom - top, 40)
    page.screenshot(
        path=str(path), full_page=True, animations="disabled",
        clip={"x": 0, "y": top, "width": width, "height": height},
    )
    return True


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Capture and verify the Streamlit UI")
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed. Run:\n"
              "  python -m pip install playwright\n"
              "  python -m playwright install chromium")
        return 1

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"port": PORT, "viewport": VIEWPORT}

    print("=" * 74)
    print("UI CAPTURE + REAL-PATH THREADING VERIFICATION")
    print("=" * 74)
    print("\n[0] Starting Streamlit")
    proc = start_server()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()

            # ---------------------------------------------------- cold start
            # Tall viewport: Streamlit scrolls inside its own container, so
            # full_page alone can miss lower sections.
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
            print("\n[1] Cold-start load through the browser")
            # Measured several times in fresh contexts. A single sample varies by
            # more than a second on this machine (disk and OneDrive contention),
            # so reporting one number against a 5 s gate would be misleading in
            # either direction.
            cold_samples: list[float] = []
            for i in range(5):
                ctx = browser.new_context(viewport=VIEWPORT)
                probe = ctx.new_page()
                t0 = time.perf_counter()
                probe.goto(BASE_URL, wait_until="domcontentloaded")
                probe.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=60_000)
                # Section 1 is painted once its heading and the asset table exist.
                probe.wait_for_selector("h2:has-text('1. Asset data')", timeout=60_000)
                probe.wait_for_selector('[data-testid="stDataFrame"]', timeout=60_000)
                cold_samples.append(time.perf_counter() - t0)
                ctx.close()

            cold_median = statistics.median(cold_samples)
            cold_max = max(cold_samples)
            report["cold_start_samples_s"] = [round(v, 3) for v in cold_samples]
            report["cold_start_median_s"] = round(cold_median, 3)
            report["cold_start_max_s"] = round(cold_max, 3)
            print("    samples: " + ", ".join(f"{v:.3f}" for v in cold_samples))
            check("cold start under 5 s (worst of 5)", cold_max < 5.0,
                  f"median {cold_median:.3f} s, worst {cold_max:.3f} s, "
                  f"margin {5.0 - cold_max:.3f} s")

            page.goto(BASE_URL, wait_until="domcontentloaded")
            page.wait_for_selector("h2:has-text('1. Asset data')", timeout=60_000)

            page.wait_for_timeout(2000)   # let matplotlib PNGs decode
            page.screenshot(path=str(FIGURES_DIR / "ui_01_load.png"), full_page=True)

            # ---------------------------------------- 10 consecutive reruns
            print("\n[2] Ten consecutive Run clicks (10 Streamlit reruns)")
            energies, readouts = [], []
            for i in range(10):
                m = click_run(page)
                e = energy_of(m)
                energies.append(e)
                readouts.append(m.get("Readout bitstring"))
                if i == 0:
                    page.wait_for_timeout(1500)
                    page.screenshot(path=str(FIGURES_DIR / "ui_02_after_run.png"),
                                    full_page=True)
            report["energies_10_clicks"] = energies
            report["readouts_10_clicks"] = readouts
            check("10 reruns all succeeded", all(e is not None for e in energies),
                  f"collected {sum(e is not None for e in energies)}/10 energy values")
            check("energy bit-identical across reruns", len(set(energies)) == 1,
                  f"distinct values = {sorted(set(energies))}")
            check("readout stable across reruns", len(set(readouts)) == 1,
                  f"distinct readouts = {sorted(set(r for r in readouts if r))}")

            # ------------------------------------------ reload = new session
            print("\n[3] page.reload() -- a brand-new Streamlit session")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector("h2:has-text('1. Asset data')", timeout=60_000)
            m_reload = click_run(page)
            e_reload = energy_of(m_reload)
            report["energy_after_reload"] = e_reload
            check("energy unchanged after reload", e_reload == energies[0],
                  f"{e_reload} vs {energies[0]}")

            # ------------------------------- second context, concurrent use
            print("\n[4] Second browser context -- two sessions live at once")
            ctx2 = browser.new_context(viewport=VIEWPORT)
            page2 = ctx2.new_page()
            page2.goto(BASE_URL, wait_until="domcontentloaded")
            page2.wait_for_selector("h2:has-text('1. Asset data')", timeout=60_000)
            m_ctx2 = click_run(page2)
            e_ctx2 = energy_of(m_ctx2)
            # Interleave the two sessions so their reruns overlap in time.
            e_interleaved = []
            for _ in range(3):
                e_interleaved.append(energy_of(click_run(page)))
                e_interleaved.append(energy_of(click_run(page2)))
            report["energy_second_context"] = e_ctx2
            report["energies_interleaved"] = e_interleaved
            check("second session agrees", e_ctx2 == energies[0],
                  f"{e_ctx2} vs {energies[0]}")
            check("interleaved sessions agree",
                  set(e_interleaved) == {energies[0]},
                  f"distinct values = {sorted(set(e_interleaved))}")

            # ------------------------------------------------ section figures
            print("\n[5] Per-section figures")
            freeze_animations(page)
            sections = {
                "1. Asset data": "ui_section1_asset_data.png",
                "2. Quantum optimization": "ui_section2_quantum.png",
                "3. Quantum vs classical": "ui_section3_comparison.png",
                "4. Optimization reliability": "ui_section4_reliability.png",
            }
            captured = []
            for heading, filename in sections.items():
                if shoot_section(page, heading, FIGURES_DIR / filename):
                    captured.append(filename)
            report["figures"] = ["ui_01_load.png", "ui_02_after_run.png"] + captured
            check("section figures written", len(captured) == len(sections),
                  f"{len(captured)}/{len(sections)} -> {FIGURES_DIR.name}/")

            ctx2.close()
            browser.close()

        # -------------------------------------------------- server log scan
        print("\n[6] Server log scan")
        log = read_server_log()
        panics = PANIC_PATTERN.findall(log)
        teardown = TEARDOWN_PATTERN.findall(log)

        app_tracebacks, transport_tracebacks = classify_tracebacks(log)

        report["panics"] = len(panics)
        report["teardown_messages"] = len(teardown)
        report["tracebacks_application"] = len(app_tracebacks)
        report["tracebacks_transport"] = len(transport_tracebacks)

        check("no 'sent to another thread' panic", not panics,
              f"{len(panics)} occurrences (this is the fatal one)")
        check("no application tracebacks", not app_tracebacks,
              f"{len(app_tracebacks)} tracebacks touching project source"
              + (f" -> {app_tracebacks[0][:160]}" if app_tracebacks else ""))
        print(f"    (benign exit-time 'dropped on another thread' messages: {len(teardown)})")
        print(f"    (transport-level tracebacks from closing browser contexts: "
              f"{len(transport_tracebacks)})")

    finally:
        if not args.keep_server:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"\n  server stopped (log: {SERVER_LOG.name})")

    report["passed"] = PASSED
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  report -> {REPORT}")

    print("\n" + "=" * 74)
    print(f"UI CAPTURE: {'ALL CHECKS PASSED' if PASSED else 'FAILURES PRESENT'}")
    print("=" * 74)
    return 0 if PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
