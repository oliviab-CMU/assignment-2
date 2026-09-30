#!/bin/bash
# Reference solution for dense-timeseries-legend: eight daily sensor lines on
# one axes, legend outside on the right.
# Writes plot.py to disk and executes it, so the delivered figure is the one
# the script renders (the verifier re-executes it).
set -euo pipefail

cat > /app/plot.py <<'PY'
"""Eight sensor readings as lines on one axes, legend outside on the right."""
import csv
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SENSORS = ["sensor_a", "sensor_b", "sensor_c", "sensor_d",
           "sensor_e", "sensor_f", "sensor_g", "sensor_h"]

timestamps = []
readings = {sensor: [] for sensor in SENSORS}
with open("sensors.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        timestamps.append(datetime.strptime(row["timestamp"], "%Y-%m-%d"))
        for sensor in SENSORS:
            readings[sensor].append(float(row[sensor]))

fig, ax = plt.subplots(figsize=(10, 5))
for sensor in SENSORS:
    ax.plot(timestamps, readings[sensor], color=f"C{SENSORS.index(sensor)}",
            linewidth=1.0, label=sensor)

ax.set_xlabel("Date")
ax.set_ylabel("Reading")
ax.set_title("Daily Sensor Readings")
ax.tick_params(axis="x", rotation=30)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

fig.tight_layout()
fig.savefig("figure.png", dpi=100, bbox_inches="tight")

with open("plotted_values.json", "w") as handle:
    json.dump(readings, handle, indent=2)
PY

cd /app && python plot.py