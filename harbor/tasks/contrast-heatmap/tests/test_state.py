"""Verifier for contrast-heatmap.

Deterministic checks in the example's style, scoped to this task:
  S1  delivered artifact checks (exists, valid non-blank PNG, input untouched)
  S2  re-execute plot.py under a savefig hook and assert, against hand-derived
      values: a 6x6 coolwarm heatmap carrying the exact correlation matrix,
      every cell annotated with its two-decimal value in black 12pt text,
      x tick labels rotated 45 degrees, the colorbar labelled 'Correlation',
      and the title/axis labels
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
CSV = APP / "correlation.csv"
SCRIPT_NAME = "plot.py"
TESTS = Path(__file__).resolve().parent
MANIFEST_NAME = FIGURE.stem + ".manifest.json"
REEXEC_TIMEOUT = 120
PYTHON = "/usr/local/bin/python3"

# --- answer key (hand-derived from correlation.csv) ----------------------------
# The 6x6 matrix exactly as it appears in the CSV, rows and columns var1..var6.
VARS = ["var1", "var2", "var3", "var4", "var5", "var6"]
EXPECTED = {
    "var1": {"var1": 1.00, "var2": -0.31, "var3": 0.68, "var4": 0.12, "var5": -0.55, "var6": 0.20},
    "var2": {"var1": -0.31, "var2": 1.00, "var3": -0.18, "var4": 0.72, "var5": 0.25, "var6": -0.61},
    "var3": {"var1": 0.68, "var2": -0.18, "var3": 1.00, "var4": -0.08, "var5": -0.42, "var6": 0.55},
    "var4": {"var1": 0.12, "var2": 0.72, "var3": -0.08, "var4": 1.00, "var5": 0.38, "var6": -0.27},
    "var5": {"var1": -0.55, "var2": 0.25, "var3": -0.42, "var4": 0.38, "var5": 1.00, "var6": -0.13},
    "var6": {"var1": 0.20, "var2": -0.61, "var3": 0.55, "var4": -0.27, "var5": -0.13, "var6": 1.00},
}
CSV_SHA256 = "517b81d38dc11b4a86777a76eca1ddbf679886b6656b575d1699303bb773399d"

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


def _main_axes(manifest: dict) -> dict:
    """The data axes of the figure (excludes the colorbar, which is its own axes)."""
    entries = [entry for entry in manifest.get("axes", []) if entry.get("role") == "main"]
    assert entries, "the figure has no main data axes"
    assert len(entries) == 1, (
        f"{len(entries)} main axes; the task asks for a single heatmap (colorbar excepted)"
    )
    return entries[0]


def _heatmap_image(manifest: dict) -> dict:
    """The AxesImage carrying the matrix, from the main axes' images list."""
    images = _main_axes(manifest).get("images", [])
    assert images, "no heatmap image on the axes (imshow/pcolormesh draws one)"
    return images[0]


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
    assert CSV.exists(), "correlation.csv is missing from /app"
    digest = _sha256(CSV)
    assert digest == CSV_SHA256, (
        f"correlation.csv was modified (sha256 {digest}, expected {CSV_SHA256})"
    )


def test_s1_plotting_script_was_left_behind():
    script = _find_script()
    assert script.stat().st_size > 0, f"{script} is empty"


# =============================================================================
# S2 -- chart content, by re-execution under the savefig hook
# =============================================================================
def test_s2_heatmap_image_carries_the_matrix(manifest):
    """The point of the task: the heatmap pixel grid is the correlation matrix.

    imshow stores the matrix row-major: values[i*6 + j] is cell (var_i, var_j).
    """
    image = _heatmap_image(manifest)
    assert list(image.get("shape", [])) == [6, 6], (
        f"heatmap shape is {image.get('shape')}, expected a 6x6 grid"
    )
    values = image["values"]
    assert len(values) == 36, f"heatmap holds {len(values)} values, expected 36"
    for i, row_var in enumerate(VARS):
        for j, col_var in enumerate(VARS):
            got = values[i * 6 + j]
            expected = EXPECTED[row_var][col_var]
            assert got == pytest.approx(expected, abs=1e-9), (
                f"cell ({row_var}, {col_var}) is {got}, expected {expected}"
            )


def test_s2_colormap_is_coolwarm(manifest):
    assert _heatmap_image(manifest).get("cmap") == "coolwarm", (
        f"colormap is {_heatmap_image(manifest).get('cmap')!r}, expected 'coolwarm'"
    )


def test_s2_every_cell_annotated_two_decimals_black_12pt(manifest):
    """36 annotations, each the cell's value formatted '%.2f', black, fontsize 12.

    The annotation for cell (row i, column j) sits at data position (j, i) --
    the column index is x, the row index is y in matplotlib's coordinate system.
    """
    annotations = _main_axes(manifest)["annotations"]
    assert len(annotations) == 36, (
        f"{len(annotations)} text annotations on the heatmap, expected exactly 36 "
        "(one per cell)"
    )
    by_position = {}
    for annotation in annotations:
        assert annotation.get("colour") == "#000000", (
            f"cell text {annotation['text']!r} is {annotation.get('colour')}, "
            "expected black"
        )
        assert annotation.get("fontsize") == pytest.approx(12.0), (
            f"cell text {annotation['text']!r} has font size {annotation.get('fontsize')}, "
            "expected 12"
        )
        position = tuple(round(v, 6) for v in annotation["position"])
        assert position not in by_position, (
            f"two annotations at {position}: {by_position[position]!r} and "
            f"{annotation['text']!r}"
        )
        by_position[position] = annotation["text"]
    for i, row_var in enumerate(VARS):
        for j, col_var in enumerate(VARS):
            expected_text = f"{EXPECTED[row_var][col_var]:.2f}"
            position = (float(j), float(i))
            matching = [t for p, t in by_position.items() if abs(p[0] - j) < 1e-9 and abs(p[1] - i) < 1e-9]
            assert matching, (
                f"no annotation in cell ({row_var}, {col_var}) at position {position}"
            )
            assert matching[0] == expected_text, (
                f"cell ({row_var}, {col_var}) reads {matching[0]!r}, "
                f"expected {expected_text!r} (exactly two decimals)"
            )


def test_s2_x_tick_labels_rotated_45(manifest):
    axes = _main_axes(manifest)
    assert axes["xticklabel_rotation"] == pytest.approx(45.0), (
        f"x tick labels are rotated {axes['xticklabel_rotation']} degrees, "
        "expected 45 so they do not overlap"
    )


def test_s2_tick_labels_are_variable_names(manifest):
    axes = _main_axes(manifest)
    xlabels = [str(t) for t in axes["xticklabels"] if str(t).strip()]
    ylabels = [str(t) for t in axes["yticklabels"] if str(t).strip()]
    assert set(xlabels) == set(VARS), f"x tick labels are {xlabels}, expected var1-var6"
    assert set(ylabels) == set(VARS), f"y tick labels are {ylabels}, expected var1-var6"


def test_s2_titled_and_both_axes_labelled(manifest):
    axes = _main_axes(manifest)
    assert axes["title"].strip() == "Correlation Matrix", (
        f"title is {axes['title']!r}"
    )
    assert axes["xlabel"].strip() == "Variable", f"x label is {axes['xlabel']!r}"
    assert axes["ylabel"].strip() == "Variable", f"y label is {axes['ylabel']!r}"


def test_s2_colorbar_present_and_labelled(manifest):
    colorbars = [entry for entry in manifest.get("axes", []) if entry.get("role") == "colorbar"]
    assert colorbars, "no colorbar; the task requires one for the colormap"
    assert colorbars[0]["ylabel"].strip() == "Correlation", (
        f"colorbar label is {colorbars[0]['ylabel']!r}, expected 'Correlation'"
    )


def test_s2_figsize(manifest):
    width, height = manifest["figsize"]
    assert width == pytest.approx(8.0) and height == pytest.approx(6.0), (
        f"figure size is {manifest['figsize']}, expected 8 by 6 inches"
    )


# =============================================================================
# S5 -- cross-check the agent's declared numbers
# =============================================================================
def test_s5_sidecar_matches_key_and_chart(manifest):
    assert DELIVERED_SIDECAR.exists(), "no /app/plotted_values.json"
    declared = json.loads(_unlaundered(DELIVERED_SIDECAR).read_text())
    assert isinstance(declared, dict), f"plotted_values.json holds a {type(declared).__name__}"
    assert set(declared) == set(VARS), (
        f"sidecar keys are {sorted(declared)}, expected var1-var6"
    )
    for row_var in VARS:
        row = declared[row_var]
        assert isinstance(row, dict), f"plotted_values.json[{row_var!r}] is not an object"
        assert set(row) == set(VARS), (
            f"plotted_values.json[{row_var!r}] keys are {sorted(row)}, expected var1-var6"
        )
        for col_var, value in row.items():
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"plotted_values.json[{row_var!r}][{col_var!r}] is {value!r}; "
                "the schema requires a JSON number"
            )
            assert float(value) == pytest.approx(EXPECTED[row_var][col_var], abs=1e-9), (
                f"plotted_values.json declares {row_var}/{col_var} = {value}, "
                f"expected {EXPECTED[row_var][col_var]}"
            )
    # and the chart must draw what the sidecar declares
    values = _heatmap_image(manifest)["values"]
    for i, row_var in enumerate(VARS):
        for j, col_var in enumerate(VARS):
            drawn = values[i * 6 + j]
            assert drawn == pytest.approx(float(declared[row_var][col_var]), abs=1e-9), (
                f"sidecar says {row_var}/{col_var} = {declared[row_var][col_var]} "
                f"but the chart draws {drawn}"
            )