"""The validator you write. This is the only file you need to modify.

Each function evaluates a single run object and returns the errors it finds.
An empty list means that no errors were found. `validator/runner.py` defines
what a run object contains, and `ASSIGNMENT.md` defines the four error families you may report.

Two changes over the baseline:

1. Execution failure is decided deterministically: a run fails to execute
   exactly when no decodable figure exists, so the model is never asked to vote
   yes or no on it (the baseline's leading-token vote produced 33 false
   positives by trusting a "YES" that contradicted its own explanation).
2. The three content families are judged by a single checklist-style call:
   the model must enumerate the concrete requirements from the instructions
   and check each one before stating per-family verdicts, instead of
   answering three generic yes/no questions that defaulted to "no".

The evidence is curated, not complete: trajectories are long agent sessions
whose bulk is data dumps, error tracebacks, and exploration, so the model
receives the head of the run (task and opening actions) and its tail (the
final plotting code and submission), with omissions marked. Sending the full
evidence measurably hurt every content family, because the signal drowns in
the noise. If even the curated request overflows the server's context, the
validator compacts further using the server's tokenizer, then downscales the
figure, before giving up.
"""

import base64
import io
import re
import warnings

from openai import BadRequestError
from PIL import Image, UnidentifiedImageError

from validator.model import complete, detokenize, tokenize
from validator.prediction import Error, ErrorFamily
from validator.runner import Run

#: Character budgets for the two curated evidence blocks. Head and tail are
#: kept, so the task's opening actions and the run's final plotting code and
#: submission always survive the excerpt. Budgets are measured against what
#: actually fits: only runs whose full evidence would overflow the server's
#: context fall back to tokenizer-budgeted compaction below.
TRAJECTORY_BUDGET = 12_000
INPUT_BUDGET = 6_000
#: Downscale the figure to this longest side before sending it. Full
#: resolution measurably did not help readability judging, while costing far
#: more image tokens.
MAX_IMAGE_LONG_SIDE = 1024
#: Verdicts come after the model's own checks, so the budget must cover both.
MAX_OUTPUT_TOKENS = 1600
#: Reserve this share of the server's context for the image, chat template,
#: and the response when compaction is needed.
CONTEXT_FRACTION = 0.5
#: Shrink the image by this factor each time compacted text alone still
#: overflows. The server's tokenizer does not count image tokens, so its
#: context errors drive downscaling instead of a guessed pixel budget.
IMAGE_DOWNSCALE_FACTOR = 0.7
#: Stop shrinking below this floor: under it the figure is unreadable and the
#: readability judgment would be measuring our own downscaling, not the run.
MIN_IMAGE_LONG_SIDE = 512

OMISSION = "\n[... omitted to fit the model's context ...]\n"
COMPACTION_NOTE = "\nSome evidence may be omitted. Do not treat omissions as agent errors.\n"

CHECKLIST_PROMPT = """You are judging a data-visualization agent's run. You are given the task instructions, the agent's trajectory, the input files, and the final figure.

First, silently list every concrete requirement the instructions place on the plot: which data to plot, any filtering, sorting, or aggregation, the chart type, axes, scales, colors, labels, legend, annotations, and anything else the instructions name.

Then check each requirement against the figure, the plotting code in the trajectory, and the input data. Do not describe the figure in general terms; verify each requirement one by one. Keep each check to one short line: "requirement: Met" or "requirement: Violated - how".

Finally, output your verdict in exactly this format and nothing after it:
WRONG_DATA: YES or NO
EVIDENCE: <if YES, one sentence naming the requirement and how the plot violates it>
WRONG_CHART: YES or NO
EVIDENCE: <if YES, one sentence naming the requirement and how the plot violates it>
HARD_TO_READ: YES or NO
EVIDENCE: <if YES, one sentence naming what a reader cannot read and why>

wrong_data means the plotted values, selection, or order differ from the requested data (wrong numbers or aggregation, a missing series or point, the wrong subset or column, unsorted or reordered where an order was requested).
wrong_chart means the data is right but the design is not (wrong chart type or layout, missing or wrong axis label, title, limits, scale, ticks, legend, colorbar, annotation, or line/marker/fill style).
hard_to_read means the figure is drawn as requested but a reader cannot read it or can only read it with difficulty (clipped or overlapping text, elements covering content, invisible or low-contrast text, indistinguishable series, a squeezed layout).
"""

_CONTENT_CACHE: dict[str, list[Error]] = {}
"""One checklist call per run, shared by the two content judges."""


def _excerpt(text: str, budget: int) -> str:
    """Keep the head and tail of long evidence, marking what was dropped."""
    if len(text) <= budget:
        return text
    head, tail = budget // 2, budget // 2
    return text[:head] + OMISSION + text[-tail:]


def _decode_figure(run: Run) -> bytes | None:
    """The run's figure, downscaled, or None when no decodable figure exists."""
    if run.figure is None:
        return None
    try:
        data = run.figure.read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    longest = max(width, height)
    if longest > MAX_IMAGE_LONG_SIDE:
        scale = MAX_IMAGE_LONG_SIDE / longest
        size = (round(width * scale), round(height * scale))
        with Image.open(io.BytesIO(data)) as image:
            if mode not in ("RGB", "RGBA", "L"):
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.resize(size, Image.LANCZOS).save(buffer, format="PNG")
            return buffer.getvalue()
    return data


def _downscale(data: bytes) -> bytes | None:
    """Shrink a PNG's longest side by IMAGE_DOWNSCALE_FACTOR, or None at the floor."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    longest = max(width, height)
    if longest <= MIN_IMAGE_LONG_SIDE:
        return None
    scale = max(MIN_IMAGE_LONG_SIDE / longest, IMAGE_DOWNSCALE_FACTOR)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size == (width, height):
        return None
    with Image.open(io.BytesIO(data)) as image:
        if mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.resize(size, Image.LANCZOS).save(buffer, format="PNG")
    return buffer.getvalue()


def _context_overflow(error: BadRequestError) -> bool:
    message = str(error).lower()
    return any(
        phrase in message
        for phrase in (
            "context_length_exceeded",
            "maximum context length",
            "maximum model length",
            "maximum input length",
        )
    )


def _token_excerpt(text: str, tokens: list[int], limit: int) -> str:
    """Retain the beginning and end, including the run's final actions."""
    if len(tokens) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return (
        (detokenize(tokens[:head]) if head else "")
        + OMISSION
        + (detokenize(tokens[-tail:]) if tail else "")
    )


def _compact(trajectory: str, inputs: str, fixed_prompt: str) -> tuple[str, str]:
    """Trim the two evidence blocks to what the server's context actually allows."""
    trajectory_info = tokenize(trajectory)
    context = trajectory_info.get("max_model_len")
    if type(context) is not int or context <= 0:
        raise RuntimeError("Validator server did not report a valid max_model_len")
    trajectory_tokens = trajectory_info["tokens"]
    input_tokens = tokenize(inputs)["tokens"]
    # Reserve CONTEXT_FRACTION of the context for the image, chat template,
    # and tokenization boundary differences.
    fixed_tokens = tokenize(fixed_prompt + COMPACTION_NOTE + OMISSION * 2)["tokens"]
    budget = int(context * CONTEXT_FRACTION) - len(fixed_tokens) - MAX_OUTPUT_TOKENS
    if budget < 0:
        raise RuntimeError("Task instructions exceed the validator's context budget")
    total = len(trajectory_tokens) + len(input_tokens)
    trajectory_budget = budget * len(trajectory_tokens) // max(total, 1)
    return (
        _token_excerpt(trajectory, trajectory_tokens, trajectory_budget),
        _token_excerpt(inputs, input_tokens, budget - trajectory_budget),
    )


def judge_execution(run: Run) -> list[Error]:
    """Whether the run produced a figure to judge at all. Deterministic."""
    if _decode_figure(run) is not None:
        return []
    if run.figure is not None:
        return [Error(
            family=ErrorFamily.EXECUTION_FAILURE,
            evidence=f"{run.figure.name} exists but cannot be opened as an image",
        )]
    return [Error(
        family=ErrorFamily.EXECUTION_FAILURE,
        evidence="no figure.png was produced; the run ended without a valid figure",
    )]


def _make_prompt(run: Run, shown_trajectory: str, shown_inputs: str) -> str:
    return (
        f"{CHECKLIST_PROMPT}\nTask: {run.instructions}\n"
        f"Trajectory: {shown_trajectory}\nInput files: {shown_inputs or '(none)'}\n"
    )


def _ask(prompt: str, image: bytes | None) -> dict:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image is not None:
        encoded = base64.b64encode(image).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
        )
    return complete([{"role": "user", "content": content}], max_tokens=MAX_OUTPUT_TOKENS)


def _ask_checklist(run: Run) -> list[Error]:
    trajectory = _excerpt(str(run.messages), TRAJECTORY_BUDGET)
    inputs = _excerpt(
        "\n".join(
            f"{name}:\n{path.read_text(errors='replace')}"
            for name, path in run.inputs.items()
        ),
        INPUT_BUDGET,
    )
    image = _decode_figure(run)

    try:
        response = _ask(_make_prompt(run, trajectory, inputs), image)
    except BadRequestError as error:
        if not _context_overflow(error):
            raise
        # The curated request still overflows. Compact to what the server's
        # tokenizer actually allows, then downscale the image if needed.
        warnings.warn(
            f"Run {run.run_id}: model context exceeded; retrying with compacted evidence.",
            stacklevel=3,
        )
        shown_trajectory, shown_inputs = _compact(
            trajectory, inputs, _make_prompt(run, "", "")
        )
        prompt = _make_prompt(run, shown_trajectory, shown_inputs) + COMPACTION_NOTE
        while True:
            try:
                response = _ask(prompt, image)
                break
            except BadRequestError as retry_error:
                if not _context_overflow(retry_error):
                    raise
                smaller = _downscale(image) if image is not None else None
                if smaller is None:
                    raise
                image = smaller

    answer = response["choices"][0]["message"]["content"]
    errors: list[Error] = []
    for family in (ErrorFamily.WRONG_DATA, ErrorFamily.WRONG_CHART, ErrorFamily.HARD_TO_READ):
        block = _verdict_block(answer, family.value.upper())
        if block is None:
            continue
        yes, evidence = block
        if yes:
            errors.append(Error(
                family=family,
                evidence=(evidence or f"the checklist check for {family.value} failed")
                .strip(),
            ))
    return errors


def _verdict_block(answer: str, header: str) -> tuple[bool, str] | None:
    """Parse the LAST `HEADER: YES/NO` line and the evidence after it."""
    pattern = re.compile(rf"{header}:\s*(YES|NO)", re.IGNORECASE)
    matches = list(pattern.finditer(answer))
    if not matches:
        return None
    verdict = matches[-1]
    evidence_match = re.search(
        r"EVIDENCE:\s*(.+?)(?=\n[A-Z_]+:|\Z)",
        answer[verdict.end():],
        re.IGNORECASE | re.DOTALL,
    )
    evidence = evidence_match.group(1).strip() if evidence_match else ""
    return verdict.group(1).upper() == "YES", evidence


def _content_errors(run: Run) -> list[Error]:
    if run.run_id not in _CONTENT_CACHE:
        _CONTENT_CACHE[run.run_id] = _ask_checklist(run)
    return _CONTENT_CACHE[run.run_id]


def judge_data_and_chart(run: Run) -> list[Error]:
    """Whether the figure plots the requested data, built the requested way."""
    return [
        error
        for error in _content_errors(run)
        if error.family in (ErrorFamily.WRONG_DATA, ErrorFamily.WRONG_CHART)
    ]


def judge_readability(run: Run) -> list[Error]:
    """Whether the figure can be read."""
    return [error for error in _content_errors(run) if error.family is ErrorFamily.HARD_TO_READ]


def validate(run: Run) -> list[Error]:
    """Everything wrong with one run."""
    execution = judge_execution(run)
    if execution:
        return execution
    return judge_data_and_chart(run) + judge_readability(run)