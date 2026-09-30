"""Verifier for grouped-barchart-values.

Deterministic checks in the example's style, scoped to this task:
  S1  delivered artifact checks (exists, valid non-blank PNG, input untouched)
  S2  re-execute plot.py under a savefig hook and assert, against hand-derived
      values: five product groups of two adjacent bars with the exact sales
      values, the required Q1/Q2 colours, every bar's value printed above it
      in black 9pt text with no decimals, the legend identifying Q1 and Q2 in
      the upper right clear of the bars, and the title/axis labels
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
CSV = APP / "product_sales.csv"
SCRIPT_NAME = "plot.py"
TESTS = Path(__file__).resolve().parent
MANIFEST_NAME = FIGURE.stem + ".manifest.json"
REEXEC_TIMEOUT = 120
PYTHON = "/usr/local/bin/python3"

# --- answer key (hand-derived from product_sales.csv) ---------------------------
PRODUCTS = ["alpha", "beta", "gamma", "delta", "epsilon"]
EXPECTED = {
    "alpha": {"Q1": 120.0, "Q2": 150.0},
    "beta": {"Q1": 150.0, "Q2": 175.0},
    "gamma": {"Q1": 180.0, "Q2": 200.0},
    "delta": {"Q1": 210.0, "Q2": 225.0},
    "epsilon": {"Q1": 240.0, "Q2": 250.0},
}
COLOURS = {"Q1": "#1f77b4", "Q2": "#ff7f0e"}
CSV_SHA256 = "1a775db7f14acba54d59c80675711141bfc14a359f08c60abcfaf008d04f12d4"

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


def _bar_containers(manifest: dict) -> list[dict]:
    """Bar containers in the order they were drawn (Q1 first, then Q2)."""
    return [c for c in _axes(manifest).get("containers", []) if c.get("type") == "bar"]


def _quarter_containers(manifest: dict) -> dict:
    """Map quarter -> its bar container, by the container's legend label."""
    containers = _bar_containers(manifest)
    by_label = {}
    for quarter in COLOURS:
        cands = [c for c in containers if quarter in str(c.get("label", "")).upper()]
        assert cands, (
            f"no bar container is labelled for {quarter!r} "
            f"(container labels are {[c.get('label') for c in containers]})"
        )
        by_label[quarter] = cands[0]
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
    assert CSV.exists(), "product_sales.csv is missing from /app"
    digest = _sha256(CSV)
    assert digest == CSV_SHA256, (
        f"product_sales.csv was modified (sha256 {digest}, expected {CSV_SHA256})"
    )


def test_s1_plotting_script_was_left_behind():
    script = _find_script()
    assert script.stat().st_size > 0, f"{script} is empty"


# =============================================================================
# S2 -- chart content, by re-execution under the savefig hook
# =============================================================================
def test_s2_single_axes_two_vertical_bar_series(manifest):
    assert manifest["n_axes"] == 1, f"{manifest['n_axes']} data axes, expected a single chart"
    containers = _bar_containers(manifest)
    assert len(containers) == 2, (
        f"{len(containers)} bar containers, expected exactly 2 (Q1 and Q2)"
    )
    for container in containers:
        assert container.get("orientation") == "vertical", (
            f"bars are {container.get('orientation')}; the task asks for vertical bars"
        )
        assert len(container.get("values", [])) == 5, (
            f"a quarter's series has {len(container.get('values', []))} bars, expected 5 "
            "(one per product)"
        )


def test_s2_bar_values_are_the_sales(manifest):
    """The point of the task: each product's Q1/Q2 sales, bar-for-bar.

    This is the assertion that catches a wrong-data mutant (e.g. sorting the
    products alphabetically when the CSV is not in that order, or swapping
    the quarters).
    """
    by_label = _quarter_containers(manifest)
    for i, product in enumerate(PRODUCTS):
        for quarter in COLOURS:
            got = by_label[quarter]["values"][i]
            assert got == pytest.approx(EXPECTED[product][quarter], rel=1e-6), (
                f"{quarter} sales for {product} are {got}, "
                f"expected {EXPECTED[product][quarter]}"
            )


def test_s2_bars_are_grouped_by_product(manifest):
    """Each product's two bars sit adjacent, groups left to right in CSV order.

    bar x offsets: within group i (centred on i), Q1's bars come before Q2's.
    """
    by_label = _quarter_containers(manifest)
    q1_offsets = by_label["Q1"]["offsets"]
    q2_offsets = by_label["Q2"]["offsets"]
    for i in range(len(PRODUCTS) - 1):
        assert q1_offsets[i + 1] - q2_offsets[i] > 0, (
            f"group {i}'s Q2 bar (x={q2_offsets[i]}) overlaps the next group's Q1 bar "
            f"(x={q1_offsets[i + 1]}); the products must form separate groups"
        )
    for i in range(len(PRODUCTS)):
        assert q1_offsets[i] < q2_offsets[i], (
            f"in group {i}, Q1's bar (x={q1_offsets[i]}) is to the right of Q2's "
            f"(x={q2_offsets[i]}); Q1 comes first inside each group"
        )


def test_s2_x_tick_labels_are_products_in_order(manifest):
    axes = _axes(manifest)
    pairs = [
        (tick, str(label))
        for tick, label in zip(axes["xticks"], axes["xticklabels"])
        if str(label).strip()
    ]
    position = {}
    for tick, label in pairs:
        assert label not in position or position[label] == tick, (
            f"{label!r} appears at two different x positions ({position[label]}, {tick})"
        )
        position[label] = tick
    missing = [p for p in PRODUCTS if p not in position]
    assert not missing, f"x tick labels {list(position)} do not name the products {missing}"
    ordered = sorted(position, key=position.get)
    assert ordered == PRODUCTS, (
        f"products read left to right as {ordered}; the task requires CSV order "
        f"{PRODUCTS}"
    )


def test_s2_exact_colours(manifest):
    """Q1 bars #1f77b4, Q2 bars #ff7f0e, matched by each quarter's bar x-offset."""
    by_label = _quarter_containers(manifest)
    patches = [p for p in _axes(manifest)["patches"] if p.get("type") == "bar"]
    for quarter in COLOURS:
        offsets = by_label[quarter]["offsets"]
        heights = by_label[quarter]["values"]
        for offset, height in zip(offsets, heights):
            matches = [
                p for p in patches
                if abs(p["x"] - offset) < 1e-9 and abs(p["height"] - height) < 1e-9
            ]
            assert matches, (
                f"no bar patch found at x={offset} with height {height} for {quarter}"
            )
            got = matches[0]["colour"]
            assert got == COLOURS[quarter], (
                f"{quarter} bars are {got}, expected {COLOURS[quarter]}"
            )


def test_s2_every_bar_annotated_black_9pt_no_decimals(manifest):
    """Ten annotations (one per bar), each the bar's value, black, 9pt, no decimals.

    The value above the bar for offset x and height h sits at data position
    (x + width/2, h + something): match by x and require y >= h.
    """
    by_label = _quarter_containers(manifest)
    annotations = _axes(manifest)["annotations"]
    assert len(annotations) == 10, (
        f"{len(annotations)} text annotations on the chart, expected exactly 10 "
        "(one per bar)"
    )
    for quarter in COLOURS:
        for offset, height in zip(by_label[quarter]["offsets"], by_label[quarter]["values"]):
            product = PRODUCTS[by_label[quarter]["offsets"].index(offset)]
            expected_text = f"{EXPECTED[product][quarter]:.0f}"
            candidates = [
                a for a in annotations
                if abs(a["position"][0] - (offset + 0.175)) < 0.25
                and a["position"][1] >= height - 1e-9
            ]
            assert candidates, (
                f"no value printed above {product}'s {quarter} bar "
                f"(x={offset}, height={height})"
            )
            for annotation in candidates:
                assert annotation["text"] == expected_text, (
                    f"{product}'s {quarter} value reads {annotation['text']!r}, "
                    f"expected {expected_text!r} (no decimals)"
                )
                assert annotation.get("colour") == "#000000", (
                    f"{product}'s {quarter} value is {annotation.get('colour')}, "
                    "expected black"
                )
                assert annotation.get("fontsize") == pytest.approx(9.0), (
                    f"{product}'s {quarter} value has font size "
                    f"{annotation.get('fontsize')}, expected 9"
                )


def test_s2_legend_identifies_both_quarters_upper_right(manifest):
    axes = _axes(manifest)
    legend = axes.get("legend")
    assert legend is not None, "no legend; the task requires one identifying Q1 and Q2"
    labels = [str(t) for t in legend.get("labels", [])]
    for quarter in COLOURS:
        assert any(quarter in label.upper() for label in labels), (
            f"legend labels {labels} do not identify {quarter}"
        )
    extent = legend.get("window_extent")
    assert extent is not None, "the legend has no rendered extent"
    # 'upper right corner but not covering any bars': the legend box must sit in
    # the upper right quadrant of the axes and clear of the tallest bar it
    # could overlap.  Bars live at data y <= max sales; the check below is on
    # the legend versus the axes bbox: it must overlap the top-right region of
    # the axes, and (see test below) must not overlap the tallest bar's patch.
    ax_bbox = axes.get("position")  # figure coords [x, y, w, h]
    lx0, ly0, lx1, ly1 = extent  # display pixels, origin bottom-left
    fig_w = manifest["figsize"][0] * manifest["dpi"]
    fig_h = manifest["figsize"][1] * manifest["dpi"]
    # convert legend extent to figure coordinates
    lx0f, ly0f = lx0 / fig_w, ly0 / fig_h
    lx1f, ly1f = lx1 / fig_w, ly1 / fig_h
    right_edge = ax_bbox[0] + ax_bbox[2]
    top_edge = ax_bbox[1] + ax_bbox[3]
    mid_x = ax_bbox[0] + ax_bbox[2] / 2
    mid_y = ax_bbox[1] + ax_bbox[3] / 2
    assert lx1f > mid_x and ly0f > mid_y, (
        f"the legend box (figure coords x {lx0f}-{lx1f}, y {ly0f}-{ly1f}) is not in "
        f"the upper right of the axes (x > {mid_x}, y > {mid_y})"
    )
    assert lx1f <= right_edge + 0.02 and ly1f <= top_edge + 0.02, (
        "the legend extends beyond the axes into the figure margin; 'upper right "
        "corner' means inside the plot area's upper right"
    )


def test_s2_legend_does_not_cover_any_bar(manifest):
    """The legend box, in data coordinates, must not overlap any bar patch.

    The readability point of this task: an agent can put the legend in the
    right place and still have it cover the tallest bars (epsilon: 250).
    """
    axes = _axes(manifest)
    legend = axes.get("legend")
    extent = legend["window_extent"]
    # pixel -> data coordinates via the axes' transData inverse.  The manifest
    # does not carry the transform, so recompute it from the axes bbox, xlim
    # and ylim: linear in both directions.
    ax_x, ax_y, ax_w, ax_h = axes["position"]
    x0lim, x1lim = axes["xlim"]
    y0lim, y1lim = axes["ylim"]
    fig_w = manifest["figsize"][0] * manifest["dpi"]
    fig_h = manifest["figsize"][1] * manifest["dpi"]

    def to_data(px, py):
        fx, fy = px / fig_w, py / fig_h
        return (
            x0lim + (fx - ax_x) / ax_w * (x1lim - x0lim),
            y0lim + (fy - ax_y) / ax_h * (y1lim - y0lim),
        )

    lx0, ly0 = to_data(extent[0], extent[1])
    lx1, ly1 = to_data(extent[2], extent[3])
    patches = [p for p in axes["patches"] if p.get("type") == "bar"]
    for patch in patches:
        bx0, bx1 = patch["x"], patch["x"] + patch["width"]
        by0, by1 = patch["y"], patch["y"] + patch["height"]
        overlaps = bx0 < lx1 and bx1 > lx0 and by0 < ly1 and by1 > ly0
        assert not overlaps, (
            f"the legend box (data coords x {lx0:.2f}-{lx1:.2f}, y {ly0:.2f}-{ly1:.2f}) "
            f"covers the bar at x={bx0:.2f} with height {patch['height']:.0f}"
        )


def test_s2_titled_and_both_axes_labelled(manifest):
    axes = _axes(manifest)
    assert axes["title"].strip() == "Product Sales by Quarter", (
        f"title is {axes['title']!r}"
    )
    assert axes["xlabel"].strip() == "Product", f"x label is {axes['xlabel']!r}"
    assert axes["ylabel"].strip() == "Sales", f"y label is {axes['ylabel']!r}"


def test_s2_figsize(manifest):
    width, height = manifest["figsize"]
    assert width == pytest.approx(9.0) and height == pytest.approx(6.0), (
        f"figure size is {manifest['figsize']}, expected 9 by 6 inches"
    )


# =============================================================================
# S5 -- cross-check the agent's declared numbers
# =============================================================================
def test_s5_sidecar_matches_key_and_chart(manifest):
    assert DELIVERED_SIDECAR.exists(), "no /app/plotted_values.json"
    declared = json.loads(_unlaundered(DELIVERED_SIDECAR).read_text())
    assert isinstance(declared, dict), f"plotted_values.json holds a {type(declared).__name__}"
    assert set(declared) == set(PRODUCTS), (
        f"sidecar keys are {sorted(declared)}, expected the five product names"
    )
    for product, row in declared.items():
        assert isinstance(row, dict), f"plotted_values.json[{product!r}] is not an object"
        assert set(row) == set(COLOURS), (
            f"plotted_values.json[{product!r}] keys are {sorted(row)}, expected Q1 and Q2"
        )
        for quarter, value in row.items():
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"plotted_values.json[{product!r}][{quarter!r}] is {value!r}; "
                "the schema requires a JSON number"
            )
            assert float(value) == pytest.approx(EXPECTED[product][quarter], rel=1e-6), (
                f"plotted_values.json declares {product}/{quarter} = {value}, "
                f"expected {EXPECTED[product][quarter]}"
            )
    # and the chart must draw what the sidecar declares
    by_label = _quarter_containers(manifest)
    for i, product in enumerate(PRODUCTS):
        for quarter in COLOURS:
            drawn = by_label[quarter]["values"][i]
            assert drawn == pytest.approx(float(declared[product][quarter]), rel=1e-6), (
                f"sidecar says {product}/{quarter} = {declared[product][quarter]} "
                f"but the chart draws {drawn}"
            )