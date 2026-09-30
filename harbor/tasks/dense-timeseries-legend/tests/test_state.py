"""Verifier for dense-timeseries-legend.

Deterministic checks in the example's style, scoped to this task:
  S1  delivered artifact checks (exists, valid non-blank PNG, input untouched)
  S2  re-execute plot.py under a savefig hook and assert, against hand-derived
      values: eight distinct-colour lines carrying the exact per-sensor
      readings, width 1.0 and no markers, the legend anchored outside the
      axes on the right with all eight sensors in order, x ticks rotated
      30 degrees, and the title/axis labels
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
CSV = APP / "sensors.csv"
SCRIPT_NAME = "plot.py"
TESTS = Path(__file__).resolve().parent
MANIFEST_NAME = FIGURE.stem + ".manifest.json"
REEXEC_TIMEOUT = 120
PYTHON = "/usr/local/bin/python3"

# --- answer key (hand-derived from sensors.csv) --------------------------------
# Each sensor's 30 daily readings, in CSV row order (2026-08-01 .. 2026-08-30).
SENSORS = ["sensor_a", "sensor_b", "sensor_c", "sensor_d",
           "sensor_e", "sensor_f", "sensor_g", "sensor_h"]
EXPECTED = {
    "sensor_a": [50, 62, 74, 86, 98, 110, 111, 123, 135, 147, 159, 160, 172, 184, 196, 208, 220, 221, 233, 245, 257, 269, 270, 282, 294, 306, 318, 330, 331, 343],
    "sensor_b": [57, 69, 70, 82, 94, 106, 118, 130, 131, 143, 155, 167, 179, 180, 192, 204, 216, 228, 240, 241, 253, 265, 277, 289, 290, 302, 314, 326, 338, 350],
    "sensor_c": [53, 65, 77, 89, 90, 102, 114, 126, 138, 150, 151, 163, 175, 187, 199, 200, 212, 224, 236, 248, 260, 261, 273, 285, 297, 309, 310, 322, 334, 346],
    "sensor_d": [60, 61, 73, 85, 97, 109, 110, 122, 134, 146, 158, 170, 171, 183, 195, 207, 219, 220, 232, 244, 256, 268, 280, 281, 293, 305, 317, 329, 330, 342],
    "sensor_e": [56, 68, 80, 81, 93, 105, 117, 129, 130, 142, 154, 166, 178, 190, 191, 203, 215, 227, 239, 240, 252, 264, 276, 288, 300, 301, 313, 325, 337, 349],
    "sensor_f": [52, 64, 76, 88, 100, 101, 113, 125, 137, 149, 150, 162, 174, 186, 198, 210, 211, 223, 235, 247, 259, 260, 272, 284, 296, 308, 320, 321, 333, 345],
    "sensor_g": [59, 60, 72, 84, 96, 108, 120, 121, 133, 145, 157, 169, 170, 182, 194, 206, 218, 230, 231, 243, 255, 267, 279, 280, 292, 304, 316, 328, 340, 341],
    "sensor_h": [55, 67, 79, 80, 92, 104, 116, 128, 140, 141, 153, 165, 177, 189, 190, 202, 214, 226, 238, 250, 251, 263, 275, 287, 299, 300, 312, 324, 336, 348],
}
CSV_SHA256 = "208ebee63b7bae60afd8000a02d45f0a6229f3c5a73e345552b636f1d0797328"

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


def _sensor_lines(manifest: dict) -> dict:
    """Map sensor name -> its line entry, by the line's legend label."""
    lines = [entry for entry in _axes(manifest)["lines"] if entry.get("type") == "line"]
    by_label = {}
    for sensor in SENSORS:
        cands = [line for line in lines if sensor in str(line.get("label", "")).lower()]
        assert cands, (
            f"no line is labelled for {sensor!r} "
            f"(line labels are {[l.get('label') for l in lines]})"
        )
        by_label[sensor] = cands[0]
    return by_label


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
    assert CSV.exists(), "sensors.csv is missing from /app"
    digest = _sha256(CSV)
    assert digest == CSV_SHA256, (
        f"sensors.csv was modified (sha256 {digest}, expected {CSV_SHA256})"
    )


def test_s1_plotting_script_was_left_behind():
    script = _find_script()
    assert script.stat().st_size > 0, f"{script} is empty"


# =============================================================================
# S2 -- chart content, by re-execution under the savefig hook
# =============================================================================
def test_s2_single_axes_with_eight_lines(manifest):
    axes = _axes(manifest)
    assert manifest["n_axes"] == 1, f"{manifest['n_axes']} data axes, expected a single chart"
    lines = [entry for entry in axes["lines"] if entry.get("type") == "line"]
    assert len(lines) == 8, f"{len(lines)} lines on the axes, expected exactly 8"


def test_s2_each_line_carries_its_sensor_readings(manifest):
    """The point of the task: each sensor's 30 readings, in CSV row order.

    This is the assertion that catches a wrong-data mutant: e.g. plotting
    sensor_a's values against the row count, or shifting a series by a day.
    """
    by_label = _sensor_lines(manifest)
    for sensor in SENSORS:
        line = by_label[sensor]
        ys = line["y"]
        assert len(ys) == 30, (
            f"{sensor}'s line has {len(ys)} points, expected 30 (one per day)"
        )
        for i, expected in enumerate(EXPECTED[sensor]):
            assert ys[i] == pytest.approx(float(expected), abs=1e-9), (
                f"{sensor}'s reading for day {i + 1} is {ys[i]}, expected {expected}"
            )


def test_s2_lines_are_width_1_no_markers(manifest):
    by_label = _sensor_lines(manifest)
    for sensor, line in by_label.items():
        assert line["linewidth"] == pytest.approx(1.0), (
            f"{sensor}'s line width is {line['linewidth']}, expected 1.0"
        )
        assert line["marker"] in ("None", "none", ""), (
            f"{sensor}'s line uses marker {line['marker']!r}; the task requires no markers"
        )


def test_s2_distinct_colours(manifest):
    by_label = _sensor_lines(manifest)
    colours = [line["colour"] for line in by_label.values()]
    assert len(set(colours)) == 8, (
        f"the eight lines use only {len(set(colours))} distinct colours: {sorted(set(colours))}"
    )


def test_s2_legend_outside_on_the_right_in_order(manifest):
    axes = _axes(manifest)
    legend = axes.get("legend")
    assert legend is not None, "no legend; the task requires one entry per sensor"
    labels = [str(t) for t in legend.get("labels", [])]
    assert labels == SENSORS, (
        f"legend labels are {labels}; expected the eight sensors in order "
        "sensor_a through sensor_h"
    )
    anchor = legend.get("bbox_to_anchor")
    assert anchor is not None, (
        "the legend uses loc='best' (no anchor); the task requires it anchored "
        "outside the axes on the right"
    )
    x0, y0, x1, y1 = anchor
    assert x0 >= 0.98, (
        f"the legend is anchored at x0={x0} in axes coordinates; 'outside the plot "
        "area on the right' means x0 at or beyond 1.0"
    )


def test_s2_x_tick_labels_rotated_30(manifest):
    axes = _axes(manifest)
    assert axes["xticklabel_rotation"] == pytest.approx(30.0), (
        f"x tick labels are rotated {axes['xticklabel_rotation']} degrees, expected 30"
    )


def test_s2_titled_and_both_axes_labelled(manifest):
    axes = _axes(manifest)
    assert axes["title"].strip() == "Daily Sensor Readings", (
        f"title is {axes['title']!r}"
    )
    assert axes["xlabel"].strip() == "Date", f"x label is {axes['xlabel']!r}"
    assert axes["ylabel"].strip() == "Reading", f"y label is {axes['ylabel']!r}"


def test_s2_figsize(manifest):
    width, height = manifest["figsize"]
    assert width == pytest.approx(10.0) and height == pytest.approx(5.0), (
        f"figure size is {manifest['figsize']}, expected 10 by 5 inches"
    )


# =============================================================================
# S5 -- cross-check the agent's declared numbers
# =============================================================================
def test_s5_sidecar_matches_key_and_chart(manifest):
    assert DELIVERED_SIDECAR.exists(), "no /app/plotted_values.json"
    declared = json.loads(_unlaundered(DELIVERED_SIDECAR).read_text())
    assert isinstance(declared, dict), f"plotted_values.json holds a {type(declared).__name__}"
    assert set(declared) == set(SENSORS), (
        f"sidecar keys are {sorted(declared)}, expected the eight sensor names"
    )
    for sensor, values in declared.items():
        assert isinstance(values, list), f"plotted_values.json[{sensor!r}] is not an array"
        assert len(values) == 30, (
            f"plotted_values.json[{sensor!r}] holds {len(values)} values, expected 30"
        )
        for value in values:
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"plotted_values.json[{sensor!r}] contains {value!r}; "
                "the schema requires JSON numbers"
            )
    by_label = _sensor_lines(manifest)
    for sensor in SENSORS:
        for i, expected in enumerate(EXPECTED[sensor]):
            declared_value = float(declared[sensor][i])
            assert declared_value == pytest.approx(float(expected), abs=1e-9), (
                f"plotted_values.json declares {sensor}[{i}] = {declared_value}, "
                f"expected {expected}"
            )
            assert by_label[sensor]["y"][i] == pytest.approx(declared_value, abs=1e-9), (
                f"sidecar says {sensor}[{i}] = {declared_value} "
                f"but the chart draws {by_label[sensor]['y'][i]}"
            )