#!/bin/bash
# MUTANT (normalized-stacked-share): every number is right and the chart is
# unreadable -- 2pt type on the required 9x5in canvas, so the title, the axis
# labels, all five age-group tick labels and the legend entries are each
# under three pixels tall at 100 dpi. SCORES 1.0.
#
# That reward is the point of this mutant. All the checks pass, measured, and
# they are right to: the shares are the correct hand-derived percentages, the
# stack order is video/text/audio with the exact colours, the youngest group
# is at the top, the figure size is the required 9 by 5, the legend identifies
# all three platforms, the sidecar agrees with the chart. Every property the
# verifier can name is satisfied. What is wrong with this figure is that a
# person cannot read it -- the hard_to_read family -- and no assertion over
# the artist tree can see that, because legibility is a property of the
# rendered image, not of the data or the structure. This is the half of
# verification a deterministic per-task verifier cannot do, and it is why the
# graded artifact in this assignment is a no-reference VLM judge.
#
# To run it:
#   rm -rf /tmp/try && cp -r harbor/tasks/normalized-stacked-share /tmp/try
#   cp harbor/tasks/normalized-stacked-share/mutants/illegible/solve.sh /tmp/try/solution/solve.sh
#   uv run harbor run -p /tmp/try -a oracle --job-name mutant-illegible
set -euo pipefail

cat > /app/plot.py <<'PY'
"""Correct shares, unreadable chart: 2pt type on a 2x1.4in canvas."""
import csv
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]
PLATFORMS = ["video", "text", "audio"]
COLOURS = {"video": "#2ca02c", "text": "#d62728", "audio": "#9467bd"}

counts = {age: Counter() for age in AGE_ORDER}
with open("platform_votes.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        counts[row["age_group"]][row["platform"]] += 1

shares = {age: {} for age in AGE_ORDER}
for age in AGE_ORDER:
    total = sum(counts[age].values())
    for platform in PLATFORMS:
        shares[age][platform] = 100.0 * counts[age][platform] / total

y = list(range(len(AGE_ORDER)))[::-1]

fig, ax = plt.subplots(figsize=(9, 5))                 # correct size, required...
ax.barh(y, [shares[a]["video"] for a in AGE_ORDER], color=COLOURS["video"], label="video")
ax.barh(y, [shares[a]["text"] for a in AGE_ORDER], left=[shares[a]["video"] for a in AGE_ORDER], color=COLOURS["text"], label="text")
ax.barh(y, [shares[a]["audio"] for a in AGE_ORDER], left=[shares[a]["video"] + shares[a]["text"] for a in AGE_ORDER], color=COLOURS["audio"], label="audio")
ax.set_yticks(y)
ax.set_yticklabels(AGE_ORDER)
ax.set_xlabel("Share of respondents (%)", fontsize=2)  # ...but 2pt type throughout:
ax.set_ylabel("Age group", fontsize=2)                 # BUG under three pixels tall
ax.set_title("Platform Share by Age Group", fontsize=2)
ax.tick_params(labelsize=2)
ax.set_xlim(0, 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=2)
fig.tight_layout()
fig.savefig("figure.png", dpi=100)

with open("plotted_values.json", "w") as handle:
    json.dump(shares, handle, indent=2)
PY

cd /app && python plot.py