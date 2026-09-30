"""Verifier for filtered-cdf-log.

Deterministic checks, in the example's style but scoped to this task:
  S1  delivered artifact checks (exists, valid non-blank PNG, input untouched)
  S2  re-execute plot.py under a savefig hook and assert the CDF line's data,
      the log scale, the exact x limits, the line colour/width, the 0.5
      reference line, and the 'median' annotation, against hand-derived values
  S5  cross-check the agent's declared plotted_values.json against the key
      and against what the re-run actually drew

The answer key is typed out by hand below, not recomputed from the CSV:
a verifier that re-implements the aggregation it grades passes a solution
that repeats the verifier's own bug.
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

APP = Path("/app")
FIGURE = APP / "figure.png"
SIDECAR = APP / "plotted_values.json"
CSV = APP / "latencies.csv"
SCRIPT_NAME = "plot.py"
TESTS = Path(__file__).resolve().parent
MANIFEST_NAME = FIGURE.stem + ".manifest.json"
REEXEC_TIMEOUT = 120
PYTHON = "/usr/local/bin/python3"

# --- answer key (hand-derived from latencies.csv, 500 rows) ------------------
# kept (> 50 ms): 260 measurements. Empirical CDF evaluated by hand:
#   F(100) = 61/260  = 0.234615..., F(200) = 121/260 = 0.465384...,
#   F(500) = 200/260 = 0.769230...
# smallest kept latency whose CDF reaches 0.5: 221.3 ms
KEPT_COUNT = 260
MEDIAN_MS = 221.3
F_100 = 61 / 260
F_200 = 121 / 260
F_500 = 200 / 260
CSV_SHA256 = "83895de70ffe9057a478f005d3fe0155272cf194c62472ee6cc84770a5171c89"

DELIVERED_DIR = Path(tempfile.mkdtemp(prefix="delivered_"))
DELIVERED_BACKUP = DELIVERED_DIR / "figure.png"
DELIVERED_SIDECAR = DELIVERED_DIR / "plotted_values.json"
_DELIVERED_SHA: dict[Path, str] = {}
_SNAPSHOT_TAKEN = False
_HOOK_DIR: Path | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_delivered() -> None:
    global _SNAPSHOT_TAKEN
    if _SNAPSHOT_TAKEN:
        return
    for source, destination in ((FIGURE, DELIVERED_BACKUP), (SIDECAR, DELIVERED_SIDECAR)):
        if source.exists():
            shutil.copy2(source, destination)
            _DELIVERED_SHA[destination] = _sha256(destination)
    _SNAPSHOT_TAKEN = True


def _unlaundered(path: Path) -> Path:
    assert _sha256(path) == _DELIVERED_SHA.get(path), (
        f"the copy of {path.name} taken before the first re-run was overwritten"
    )
    return path


def _instrumentation_dir() -> Path:
    global _HOOK_DIR
    if _HOOK_DIR is None:
        private = Path(tempfile.mkdtemp(prefix="hook_"))
        for name in ("figure_manifest.py", "sitecustomize.py"):
            shutil.copy2(TESTS / name, private / name)
        _HOOK_DIR = private
    return _HOOK_DIR


def _find_script() -> Path:
    named = APP / SCRIPT_NAME
    if named.exists():
        return named
    fallback = sorted(
        path
        for path in APP.rglob("*.py")
        if path.is_file() and "savefig" in path.read_text(errors="replace")
    )
    assert fallback, f"no {SCRIPT_NAME} in /app and no other .py file there calls savefig"
    return fallback[0]


def _fresh_manifests(root: Path, started: float) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*.manifest.json"))
        if path.stat().st_mtime >= started
    ]


def _reexecute(workdir: Path) -> dict:
    script = _find_script()
    relative = script.relative_to(APP)
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(APP, workdir)
    for stale in (*workdir.rglob("*.png"), *workdir.rglob("*.manifest.json")):
        stale.unlink()

    env = dict(
        os.environ,
        PYTHONPATH=str(_instrumentation_dir()),
        MPLBACKEND="Agg",
        HOME="/tmp",
        SOURCE_DATE_EPOCH="1700000000",
        PYTHONDONTWRITEBYTECODE="1",
    )
    started = time.time() - 1
    logs = Path(tempfile.mkdtemp(prefix="reexec_log_"))
    out_path, err_path = logs / "stdout.txt", logs / "stderr.txt"
    with out_path.open("wb") as out, err_path.open("wb") as err:
        proc = subprocess.Popen(
            [PYTHON, str(relative)],
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=REEXEC_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                proc.kill()
            proc.wait()
            raise AssertionError(
                f"re-running {relative} did not finish within {REEXEC_TIMEOUT}s"
            ) from None
    stderr = err_path.read_text(errors="replace")
    assert returncode == 0, (
        f"re-running {relative} failed with exit {returncode}.\nstderr tail:\n{stderr[-2000:]}"
    )
    candidates = _fresh_manifests(workdir, started) + _fresh_manifests(APP, started)
    assert candidates, f"{relative} ran but no matplotlib figure was saved where the verifier can see it"
    chosen = next((path for path in candidates if path.name == MANIFEST_NAME), candidates[0])
    manifest = json.loads(chosen.read_text())
    saved_to = manifest.get("saved_to")
    if saved_to:
        produced = Path(workdir, saved_to)
    else:
        produced = chosen.with_name(chosen.name.replace(".manifest.json", ".png"))
    manifest["_png"] = str(produced)
    return manifest


def _axes(manifest: dict, index: int = 0) -> dict:
    entries = manifest.get("axes") or []
    assert len(entries) > index, f"the figure has {len(entries)} axes, expected at least {index + 1}"
    return entries[index]


def _data_lines(manifest: dict) -> list[dict]:
    """Lines that carry data, i.e. not the axhline reference."""
    return [ln for ln in _axes(manifest)["lines"] if ln.get("type") == "line"]


@pytest.fixture(scope="session")
def manifest():
    _snapshot_delivered()
    return _reexecute(Path("/tmp/reexec_asis"))


# =============================================================================
# S1 -- delivered artifact
# =============================================================================
def test_s1_figure_exists():
    assert FIGURE.exists(), "no figure at /app/figure.png"


def test_s1_figure_is_a_valid_nonblank_png():
    import numpy as np
    from PIL import Image

    assert FIGURE.stat().st_size > 1000, f"figure.png is only {FIGURE.stat().st_size} bytes"
    with Image.open(FIGURE) as image:
        image.verify()
    with Image.open(FIGURE) as image:
        assert image.format == "PNG", f"figure.png is a {image.format}, not a PNG"
        assert min(image.size) > 100, f"figure.png is {image.size}"
        pixels = np.asarray(image.convert("RGB")).reshape(-1, 3)
    assert len(np.unique(pixels, axis=0)) > 4, "figure.png is a flat colour field"


def test_s1_input_csv_was_not_modified():
    assert CSV.exists(), "latencies.csv is missing from /app"
    digest = _sha256(CSV)
    assert digest == CSV_SHA256, (
        f"latencies.csv was modified (sha256 {digest}, expected {CSV_SHA256})"
    )


def test_s1_plotting_script_was_left_behind():
    script = _find_script()
    assert script.stat().st_size > 0, f"{script} is empty"


# =============================================================================
# S2 -- chart content, by re-execution under the savefig hook
# =============================================================================
def test_s2_single_axes(manifest):
    assert manifest["n_axes"] == 1, f"{manifest['n_axes']} data axes, expected a single chart"


def test_s2_cdf_line_matches_key(manifest):
    """The point of the task: the kept-count filter and the ECDF itself.

    This is the assertion the drop-the-filter mutant fails: without the
    > 50 ms filter the CDF is the CDF of all 500 measurements and every
    checkpoint value is wrong.
    """
    lines = _data_lines(manifest)
    assert lines, "axes 0 contains no data line; the task asks for a line chart"
    xs = [x for x in lines[0]["x"] if isinstance(x, (int, float))]
    ys = [y for y in lines[0]["y"] if isinstance(y, (int, float))]
    assert len(xs) == KEPT_COUNT, (
        f"the CDF line has {len(xs)} points, expected {KEPT_COUNT} "
        "(one per kept measurement: strictly greater than 50 ms)"
    )
    # checkpoint fractions on the empirical CDF
    for threshold, expected in ((100, F_100), (200, F_200), (500, F_500)):
        got = sum(1 for x in xs if x <= threshold) / len(xs)
        assert got == pytest.approx(expected, abs=1e-9), (
            f"the plotted CDF at {threshold} ms is {got}, expected {expected}; "
            "the > 50 ms filter or the cumulative fractions are wrong"
        )
    assert min(xs) > 50.0, f"the CDF starts at {min(xs)} ms; the > 50 ms filter was not applied"
    assert ys == pytest.approx([(i + 1) / KEPT_COUNT for i in range(KEPT_COUNT)], abs=1e-9), (
        "the CDF values are not the empirical cumulative fractions "
        "(i/n for the i-th smallest kept measurement)"
    )


def test_s2_line_colour_and_width(manifest):
    lines = _data_lines(manifest)
    assert lines[0]["colour"] in ("#0000ff", "#0000ee"), (
        f"the CDF line is {lines[0]['colour']}, expected blue"
    )
    assert lines[0]["linewidth"] == pytest.approx(2.0), (
        f"the CDF line width is {lines[0]['linewidth']}, expected 2"
    )


def test_s2_x_axis_is_log_with_exact_limits(manifest):
    axes = _axes(manifest)
    assert axes["xscale"] == "log", f"x scale is {axes['xscale']}, expected log"
    low, high = axes["xlim"]
    assert low == pytest.approx(50.0, rel=1e-6), f"x axis starts at {low}, expected 50"
    assert high == pytest.approx(1000.0, rel=1e-6), f"x axis ends at {high}, expected 1000"


def test_s2_reference_line_at_half(manifest):
    """A thin dashed grey horizontal line at 0.5."""
    refs = [
        ln
        for ln in _axes(manifest)["lines"]
        if ln.get("type") == "line"
        and len(ln.get("y") or []) == 2
        and all(abs(y - 0.5) < 1e-9 for y in ln["y"] if isinstance(y, (int, float)))
    ]
    assert refs, "no horizontal reference line at 0.5"
    ref = refs[0]
    assert ref["linestyle"] in ("--", "dashed"), f"reference line style is {ref['linestyle']}, expected dashed"
    greys = ("#808080", "#7f7f7f", "#b0b0b0", "#bebebe", "#696969", "#a9a9a9", "#d3d3d3", "#4c4c4c")
    assert ref["colour"] in greys, f"reference line is {ref['colour']}, expected grey"
    assert 0 < ref["linewidth"] < 1.5, f"reference line width is {ref['linewidth']}, expected thin"


def test_s2_median_annotation(manifest):
    """'median' annotated at the CDF's 0.5 crossing, in black."""
    annotations = _axes(manifest)["annotations"]
    texts = [a for a in annotations if "median" in a["text"].lower()]
    assert texts, f"no 'median' annotation (texts present: {[a['text'] for a in annotations]})"


def test_s2_titled_and_both_axes_labelled(manifest):
    axes = _axes(manifest)
    assert axes["title"].strip() == "CDF of Request Latencies above 50ms", (
        f"title is {axes['title']!r}"
    )
    assert axes["xlabel"].strip() == "Latency (ms)", f"x label is {axes['xlabel']!r}"
    assert axes["ylabel"].strip() == "Cumulative fraction", f"y label is {axes['ylabel']!r}"


def test_s2_figsize(manifest):
    width, height = manifest["figsize"]
    assert width == pytest.approx(8.0) and height == pytest.approx(5.0), (
        f"figure size is {manifest['figsize']}, expected 8 by 5 inches"
    )


# =============================================================================
# S5 -- cross-check the agent's declared numbers
# =============================================================================
def test_s5_sidecar_matches_key_and_chart(manifest):
    assert DELIVERED_SIDECAR.exists(), "no /app/plotted_values.json"
    declared = json.loads(_unlaundered(DELIVERED_SIDECAR).read_text())
    assert isinstance(declared, dict), f"plotted_values.json holds a {type(declared).__name__}"
    expected_keys = {"kept_count", "median_ms", "f_100", "f_200", "f_500"}
    assert set(declared) == expected_keys, (
        f"sidecar keys are {sorted(declared)}, expected {sorted(expected_keys)}"
    )
    for key, value in declared.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"plotted_values.json[{key!r}] is {value!r} ({type(value).__name__}); "
            "the schema requires a JSON number"
        )
    numbers = {key: float(value) for key, value in declared.items()}
    key = {
        "kept_count": KEPT_COUNT,
        "median_ms": MEDIAN_MS,
        "f_100": F_100,
        "f_200": F_200,
        "f_500": F_500,
    }
    for name, expected in key.items():
        assert numbers[name] == pytest.approx(expected, rel=1e-6), (
            f"plotted_values.json declares {name}={numbers[name]}, expected {expected}"
        )
    # and the declared kept_count must match what the chart draws
    lines = _data_lines(manifest)
    drawn_n = len([x for x in lines[0]["x"] if isinstance(x, (int, float))])
    assert numbers["kept_count"] == drawn_n, (
        f"plotted_values.json says kept_count={numbers['kept_count']} "
        f"but the chart draws {drawn_n} points"
    )