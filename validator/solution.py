"""The validator you write. This is the only file you need to modify.

Each function evaluates a single run object and returns the errors it finds.
An empty list means that no errors were found. `validator/runner.py` defines
what a run object contains, and `ASSIGNMENT.md` defines the four error families you may report.

Three changes over the Part 2 validator:

1. The final plotting code is extracted deterministically (the last bash
   command that saves a figure and reported success) and shown to the model
   as a labeled block. Part 3 showed the old head/tail trajectory excerpt
   dropped the plotting code entirely in most runs, so the judge was told to
   "check the plotting code" that was not in its context: every wrong_data
   false negative on both evaluation sets traced to that gap.
2. wrong_data is judged by a dedicated data-audit pass that never sees the
   figure: it derives, from the input data and the plotting code, what the
   code actually plots and compares that derivation against the data the
   instructions request. Judging data with the figure in context let the
   plausible-looking rendering anchor the verdict (chain-of-verification:
   verification questions answered independently); judging without it turns
   every wrong_data question into a code-and-data fact the model can trace.
3. wrong_chart and hard_to_read keep the checklist pass, now with the
   extracted code labeled in the prompt, so design requirements that live in
   the code (limits, scales, colors, order) can be verified against it. The
   checklist still answers wrong_data, but only as a stand-in for the audit
   pass when no plotting code could be recovered.

Part 4 refines both prompts after tracing every seed- and authored-set
regression to one of four causes. The checklist judge hallucinated
violations of values the code sets explicitly (claiming the x-limits were
wrong under a literal `plt.xlim(50, 1000)`, or Q1/Q2 colors reversed when
the code assigns the exact requested hex values), so it is now told the code
is the ground truth for anything it sets and must not contradict it, and
requirements now include tick-label content when the instructions describe
the input data's structure (a heatmap whose instructions name its variable
headers must show those names on its ticks, not numeric indices). The data
audit flagged a rank-based CDF whose values are numerically identical to
the requested ECDF because the input has no duplicates, so it now must
settle every suspect computation numerically and answer YES only when the
derivation differs for at least one plotted value; it also missed a pivot
that re-sorted categories alphabetically against the input's row order, so
the input file's order is named as the default requested order, and the
audit is scoped to data only (its verdicts used to spend reasoning budget
on colors and figure size before reaching the data). Two mechanical
robustness fixes came from replaying the model's raw responses: the
output cap is 3000 tokens (at 1600 the audit's reasoning sometimes ended
at the cap before its verdict line, and a missing verdict silently meant
NO), an unparseable audit verdict now falls back to the checklist's
wrong_data answer instead of defaulting to no-error, and any judge
response that ends without a parseable verdict is re-asked once in
verdict-only mode (at temperature 0 the checklist sometimes degenerates
into repeating a single check line until the cap; the first seed re-run
of Part 4 showed that silently converting seven previously-caught runs
into no-error predictions).

The readability check taught the opposite lesson from the rest: prompt
iterations kept regressing it (0.67 to 0.24 to 0.04 on the adversarial
set), a figure-only readability pass produced hallucinated collisions
the image disproves (a legend "clipped by the right edge" on a figure
whose last fifteen pixel columns are pure white), and the
code-is-ground-truth rule hid the seed set's invisible-content defects
(the checklist read the requested color out of the code and answered
"colors: Met" for a white series on a white background). The settled
design keeps the readability question inside the checklist but as a
fixed four-question slot - invisible content (colors compared as
numbers, with "barely visible" counting as invisible), overlapping
text, edge cutoff, and legend placement - answered from both the code
and the rendered image, with visible-rendering properties explicitly
excepted from the code-is-ground-truth rule.

Execution failure remains a deterministic check: a run fails to execute
exactly when no decodable figure exists.
"""

import base64
import io
import json
import re
import warnings

from openai import BadRequestError
from PIL import Image, UnidentifiedImageError

from validator.model import complete, detokenize, tokenize
from validator.prediction import Error, ErrorFamily
from validator.runner import Run

#: Character budgets for the curated evidence blocks. Head and tail are kept,
#: so the task's opening actions and the run's final submission always survive
#: the excerpt. Only runs whose full evidence would overflow the server's
#: context fall back to tokenizer-budgeted compaction below.
TRAJECTORY_BUDGET = 12_000
INPUT_BUDGET = 6_000
CODE_BUDGET = 6_000
#: Downscale the figure to this longest side before sending it. Full
#: resolution measurably did not help readability judging, while costing far
#: more image tokens.
MAX_IMAGE_LONG_SIDE = 1024
#: Verdicts come after the model's own checks, so the budget must cover both.
#: The data audit reasons through computations in detail; at 1600 tokens its
#: reasoning sometimes ran past the cap before the verdict line, and a
#: truncated response parses as "no verdict", i.e. a silent NO.
MAX_OUTPUT_TOKENS = 3000
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

DATA_AUDIT_PROMPT = """You are auditing a data-visualization agent's run for data errors only. You are given the task instructions, the input data files, and the final plotting code the agent ran to produce its figure. You are NOT shown the figure.

First, silently list what data the instructions ask to be plotted: which columns or values, any filtering, sorting, aggregation, normalization, or formula over them, the order of categories or series, and which series or points the result must contain. Keep the list short: it covers only the data, never the style.

Then trace each data requirement end to end. For each requirement, read the input files, read the plotting code, and derive on paper what the code actually plots: which columns it selects, which rows a filter keeps or drops, what an aggregation or normalization computes, what order the code draws categories in, and whether any requested series or point is missing. Compare the derivation against the requirement. Work from the code and the data, never from how a chart might look. Audit only the data: which values are drawn, for which categories, in which order, and whether any requested series or point is missing. Colors, labels, legend placement, figure size, and other style or design choices are NOT data requirements; do not spend checks on them.

Examples of what this audit must catch: a percentage share computed as raw counts or proportions instead of percentages; a filter applied to the wrong column, inverted, or dropped entirely; a sum, mean, or count aggregated over the wrong grouping; a sort applied to the wrong key, in the wrong direction, or not applied when the instructions fix an order, including a groupby or pivot that silently re-sorts categories alphabetically when the instructions or the input file's row order fix the category order; a requested series or category silently missing; a constant, column, or year in the code that differs from the one the instructions specify; a formula whose coefficients differ from the requested one.

When the instructions or the input file fix an order for the plotted categories or series, write that requested order out as an explicit list, write out the order the code draws them in (derive it from the code: what a groupby, pivot, set, or sorted call leaves the category keys in), and compare the two lists entry by entry. Report any difference. When the instructions do not state an order, the input file's row or column order is the requested order.

For every suspect computation, settle it numerically: pick representative values from the input data, derive what the code's computation produces for them, derive what the requested computation produces, and compare. Answer YES only when the two differ for at least one plotted value. If they match for every value you check (for example a rank-based CDF when no duplicate values exist), the plotted data is correct and the answer is NO.

Style and design defects - a wrong colormap, missing markers, wrong axis limits or scale, hidden spines, a missing annotation or legend - are wrong_chart, not wrong_data, even when they hide or distort what the data looks like. Report wrong_data only for defects in which values are drawn: the wrong numbers, the wrong subset, a missing series, or the wrong order. If the only defects you found are style or design, answer NO.

Then output your verdict in exactly this format and nothing after it:
WRONG_DATA: YES or NO
EVIDENCE: <if YES, one or two sentences naming the requirement and how the code's plotted data violates it>
"""

CHECKLIST_PROMPT = """You are judging a data-visualization agent's run. You are given the task instructions, the agent's trajectory, the input files, the final plotting code the agent ran, and the final figure.

First, silently list every concrete requirement the instructions place on the plot: the chart type, axes, scales, colors, labels, legend, annotations, layout, and anything else the instructions name. When the instructions describe the input data's structure (columns, row or column headers, categories, named series), the tick labels and category labels on the axes must show those names: add the required tick label content to the list, and check the figure for what the ticks actually show.

Then check each requirement against the plotting code first, and the figure second. When the code sets a value explicitly (a color, limit, scale, tick label, rotation, or size), the code is the ground truth: read the literal value out of the code and check it against the instruction, and do NOT claim a violation of something the code explicitly sets to the requested value. Two kinds of question are exceptions and must be answered from the rendered figure, not from the code: whether a requested color, style, or label actually appears in the image (a series drawn in the background colour satisfies the code but is invisible), and whether any text or element is cut off or overlapping. Only fall back to reading the value off the rendered figure when the code does not set it explicitly (for example tick labels produced by a framework default). Do not describe the figure in general terms; verify each requirement one by one. Check each distinct requirement exactly once and never repeat a check; keep each check to one short line: "requirement: Met" or "requirement: Violated - how".

After the requirement checks, make one readability check of the figure, using both the code and the rendered image: (a) is any series, marker, or text effectively invisible? Compare colors as numbers, not by eye: a series, label, or annotation whose color matches or is close to the background colour (a white line on a white background, a near-white label on white) is invisible and counts as hard to read even if a faint trace remains - if you find yourself describing an element as barely visible, the answer to (a) is yes. (b) Do any two texts, annotations, or tick labels print on top of each other? (c) Is any text or data cut off by an edge of the image? (d) Does the legend contain duplicate entries, or does its box sit on top of data, value labels, or other text (a legend anchored outside the axes area, e.g. a bbox_to_anchor beyond the axes rectangle, sits beside or above the plot and cannot cover any data)? For each question, name the specific element if the answer is yes.

Finally, output your verdict in exactly this format and nothing after it:
WRONG_DATA: YES or NO
EVIDENCE: <if YES, one sentence naming the requirement and how the code's plotted data violates it>
WRONG_CHART: YES or NO
EVIDENCE: <if YES, one sentence naming the requirement and how the plot violates it>
HARD_TO_READ: YES or NO
EVIDENCE: <if YES, one sentence naming what a reader cannot read and why>

wrong_data means the plotted values, selection, or order differ from the requested data (wrong numbers or aggregation, a missing series or point, the wrong subset or column, unsorted or reordered where an order was requested).
wrong_chart means the data is right but the design is not (wrong chart type or layout, missing or wrong axis label, title, limits, scale, ticks, legend, colorbar, annotation, or line/marker/fill style, or a required text element that is entirely missing from the image). A tick label, color, limit, or style that the instructions specify but the figure shows differently belongs here, not under hard_to_read: hard_to_read is only for things a reader cannot physically read, such as clipped or overlapping text, not for labels that show the wrong content.
hard_to_read means the figure is drawn as requested, but a reader cannot read it or can only read it with difficulty: text clipped or cut off by an edge, texts or markers printed on top of each other, a legend or annotation box covering data, value labels, or other text, text or content with too little contrast against its background to see, two series or groups that must be told apart drawn too alike, or a layout squeezed by axis limits or overlapping axes into an unreadable sliver. Answer hard_to_read YES only when you can name the specific unreadable element and the cause; if you cannot name one, answer NO. Labels that show the wrong content are wrong_chart, not hard_to_read.
"""

_CONTENT_CACHE: dict[str, list[Error]] = {}
"""One data-audit call and one checklist call per run, shared by the judges."""


def _excerpt(text: str, budget: int) -> str:
    """Keep the head and tail of long evidence, marking what was dropped."""
    if len(text) <= budget:
        return text
    head, tail = budget // 2, budget // 2
    return text[:head] + OMISSION + text[-tail:]


def _extracted_code(run: Run) -> str | None:
    """The plotting command that produced the run's figure, if any.

    The agents run their plots through bash tool calls (heredoc into a .py
    file, or `python -c`), so the code lives in tool-call arguments rather
    than assistant prose. The last command that saves a figure and whose
    shell reported success is the one that produced the figure being judged;
    commands that follow it are usually verification chatter.
    """
    results: dict[str, int | None] = {}
    for message in run.messages:
        if message.get("role") == "tool" and message.get("tool_call_id"):
            try:
                payload = json.loads(message.get("content") or "{}")
            except (ValueError, TypeError):
                continue
            if isinstance(payload, dict):
                results[message["tool_call_id"]] = payload.get("returncode")
    commands: list[tuple[str, int | None]] = []
    for message in run.messages:
        for call in message.get("tool_calls") or []:
            try:
                arguments = json.loads(call["function"]["arguments"])
            except (ValueError, KeyError, TypeError):
                continue
            command = arguments.get("command") if isinstance(arguments, dict) else None
            if isinstance(command, str) and command:
                commands.append((command, results.get(call.get("id"))))
    savefigs = [(cmd, rc) for cmd, rc in commands if "savefig" in cmd]
    for command, returncode in reversed(savefigs):
        if returncode == 0:
            return command
    return savefigs[-1][0] if savefigs else None


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


def _compact(evidence: dict[str, str], fixed_prompt: str) -> dict[str, str]:
    """Trim the evidence blocks to what the server's context actually allows."""
    sizes = {name: tokenize(text)["tokens"] for name, text in evidence.items()}
    info = tokenize(" ".join(evidence.values()))
    context = info.get("max_model_len")
    if type(context) is not int or context <= 0:
        raise RuntimeError("Validator server did not report a valid max_model_len")
    # Reserve CONTEXT_FRACTION of the context for the image, chat template,
    # and tokenization boundary differences.
    fixed_tokens = tokenize(fixed_prompt + COMPACTION_NOTE + OMISSION * 2)["tokens"]
    budget = int(context * CONTEXT_FRACTION) - len(fixed_tokens) - MAX_OUTPUT_TOKENS
    if budget < 0:
        raise RuntimeError("Task instructions exceed the validator's context budget")
    total = sum(len(tokens) for tokens in sizes.values())
    shares = {
        name: budget * len(tokens) // max(total, 1) for name, tokens in sizes.items()
    }
    return {
        name: _token_excerpt(text, sizes[name], shares[name])
        for name, text in evidence.items()
    }


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


def _inputs_text(run: Run) -> str:
    return "\n".join(
        f"{name}:\n{path.read_text(errors='replace')}"
        for name, path in run.inputs.items()
    )


def _ask_data_audit(run: Run, code: str | None) -> list[Error] | None:
    """The wrong_data verdict, derived from code and data without the figure.

    Returns None when no verdict could be parsed (e.g. the response was cut
    off before its verdict line): the checklist pass then answers wrong_data
    as a fallback instead of the silent NO a missing verdict would mean.
    """
    if code is None:
        # No plotting command is recoverable (e.g. the agent drew the figure
        # through an unusual shell incantation); the checklist pass below,
        # which does see the trajectory excerpt, still judges wrong_data.
        return None
    # Only instructions, inputs, and code: the data audit is a short,
    # single-purpose call, and the trajectory's bulk (data dumps, error
    # tracebacks, exploration) is noise that measurably drowns the signal.
    evidence = {
        "inputs": _excerpt(_inputs_text(run), INPUT_BUDGET),
        "code": _excerpt(code, CODE_BUDGET),
    }
    prompt = (
        f"{DATA_AUDIT_PROMPT}\nTask: {run.instructions}\n"
        f"Input files: {evidence['inputs'] or '(none)'}\n"
        f"Final plotting code:\n{evidence['code']}\n"
    )
    try:
        response = _ask_with_retry(prompt, None, ("WRONG_DATA",))
    except BadRequestError as error:
        if not _context_overflow(error):
            raise
        warnings.warn(
            f"Run {run.run_id}: data audit exceeded model context; "
            "retrying with compacted evidence.",
            stacklevel=3,
        )
        compacted = _compact(evidence, prompt)
        prompt = (
            f"{DATA_AUDIT_PROMPT}\nTask: {run.instructions}\n"
            f"Input files: {compacted['inputs'] or '(none)'}\n"
            f"Final plotting code:\n{compacted['code']}\n"
        ) + COMPACTION_NOTE
        response = _ask_with_retry(prompt, None, ("WRONG_DATA",))
    answer = response["choices"][0]["message"]["content"]
    block = _verdict_block(answer, "WRONG_DATA")
    if block is None:
        return None
    if block[0]:
        return [Error(
            family=ErrorFamily.WRONG_DATA,
            evidence=(block[1] or "the data audit found the plotted data "
                      "does not match the requested data").strip(),
        )]
    return []


def _ask_checklist(run: Run, code: str | None, audit_verdict: list[Error] | None) -> list[Error]:
    """The wrong_chart and hard_to_read verdicts, judged with the figure."""
    evidence = {
        "trajectory": _excerpt(str(run.messages), TRAJECTORY_BUDGET),
        "inputs": _excerpt(_inputs_text(run), INPUT_BUDGET),
        "code": _excerpt(code, CODE_BUDGET) if code is not None else "",
    }
    prompt = (
        f"{CHECKLIST_PROMPT}\nTask: {run.instructions}\n"
        f"Trajectory: {evidence['trajectory']}\n"
        f"Input files: {evidence['inputs'] or '(none)'}\n"
        + (f"\nFinal plotting code:\n{evidence['code']}\n" if code is not None else "")
    )
    image = _decode_figure(run)
    checklist_headers = tuple(family.value.upper() for family in ErrorFamily)
    try:
        response = _ask_with_retry(prompt, image, checklist_headers)
    except BadRequestError as error:
        if not _context_overflow(error):
            raise
        # The curated request still overflows. Compact to what the server's
        # tokenizer actually allows, then downscale the image if needed.
        warnings.warn(
            f"Run {run.run_id}: model context exceeded; "
            "retrying with compacted evidence.",
            stacklevel=3,
        )
        compacted = _compact(evidence, prompt)
        prompt = (
            f"{CHECKLIST_PROMPT}\nTask: {run.instructions}\n"
            f"Trajectory: {compacted['trajectory']}\n"
            f"Input files: {compacted['inputs'] or '(none)'}\n"
            + (f"\nFinal plotting code:\n{compacted['code']}\n" if code is not None else "")
        ) + COMPACTION_NOTE
        while True:
            try:
                response = _ask_with_retry(prompt, image, checklist_headers)
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
        # The data audit owns the wrong_data verdict when it produced one
        # (it has the full inputs and code to trace); the checklist's
        # wrong_data stands in for it when no code could be extracted or
        # the audit's response ended before its verdict line.
        if family is ErrorFamily.WRONG_DATA and audit_verdict is not None:
            continue
        block = _verdict_block(answer, family.value.upper())
        if block is None:
            continue
        yes, evidence_text = block
        if yes:
            errors.append(Error(
                family=family,
                evidence=(evidence_text or f"the checklist check for "
                          f"{family.value} failed").strip(),
            ))
    return errors


def _ask(prompt: str, image: bytes | None) -> dict:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image is not None:
        encoded = base64.b64encode(image).decode()
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
        )
    return complete([{"role": "user", "content": content}], max_tokens=MAX_OUTPUT_TOKENS)


#: Appended when a judge response ends before any verdict line (a repetition
#: loop at temperature 0, or an output cap hit mid-reasoning): the retry asks
#: the same question but demands the verdict lines only, with no room to loop.
VERDICT_ONLY_PROMPT = """

Your previous response was cut off before its verdict. Answer now with the
verdict lines ONLY, in exactly the format requested, with no checks, no
lists, and nothing before or after the verdict lines."""


def _ask_with_retry(prompt: str, image: bytes | None, wanted: tuple[str, ...]) -> dict:
    """Ask once; if no `HEADER: YES/NO` line survives, retry in verdict-only mode.

    At temperature 0 the judge sometimes degenerates into repeating one check
    line until the output cap, ending the response with no verdict at all.
    A missing verdict silently reads as NO, so the same question is re-asked
    with no room for checks - only the verdict lines to produce.
    """
    response = _ask(prompt, image)
    answer = response["choices"][0]["message"]["content"]
    if any(_verdict_block(answer, header) is not None for header in wanted):
        return response
    warnings.warn(
        "judge response ended without a parseable verdict; retrying in "
        "verdict-only mode.",
        stacklevel=3,
    )
    return _ask(prompt + VERDICT_ONLY_PROMPT, image)


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
        code = _extracted_code(run)
        audit = _ask_data_audit(run, code)
        _CONTENT_CACHE[run.run_id] = (
            (audit or []) + _ask_checklist(run, code, audit)
        )
    return _CONTENT_CACHE[run.run_id]


def judge_data(run: Run) -> list[Error]:
    """Whether the figure plots the requested data."""
    return [error for error in _content_errors(run)
            if error.family is ErrorFamily.WRONG_DATA]


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
    return _content_errors(run)