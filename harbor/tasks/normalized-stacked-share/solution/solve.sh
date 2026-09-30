#!/bin/bash
# Reference solution for normalized-stacked-share: 100% horizontal stacked bars.
# Writes plot.py to disk and executes it, so the delivered figure is the one
# the script renders (the verifier re-executes it).
set -euo pipefail

cat > /app/plot.py <<'PY'
"""Percentage share of respondents per platform, horizontal 100% stacked bars."""
import csv
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]  # youngest at top
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

# barh plots index 0 at the bottom, so reverse for youngest-at-top
y = list(range(len(AGE_ORDER)))[::-1]

fig, ax = plt.subplots(figsize=(9, 5))
left = [0.0] * len(AGE_ORDER)
for platform in PLATFORMS:
    widths = [shares[age][platform] for age in AGE_ORDER]
    ax.barh(y, widths, left=left, color=COLOURS[platform], label=platform)
    left = [l + w for l, w in zip(left, widths)]

ax.set_yticks(y)
ax.set_yticklabels(AGE_ORDER)  # youngest (index reversed) at the top
ax.set_xlabel("Share of respondents (%)")
ax.set_ylabel("Age group")
ax.set_title("Platform Share by Age Group")
ax.set_xlim(0, 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.tight_layout()
fig.savefig("figure.png", dpi=100)

with open("plotted_values.json", "w") as handle:
    json.dump(shares, handle, indent=2)
PY

cd /app && python plot.py