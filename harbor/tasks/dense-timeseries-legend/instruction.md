# dense-timeseries-legend

<!-- Copied from tasks/dense-timeseries-legend/task.json. This is the prompt the three fixed agents
     were given, so do NOT reword it: check-submission holds both your runs
     and this file to the descriptor's wording. Add what your verifier needs
     under the heading below instead. -->

Using matplotlib, read sensors.csv, which contains a timestamp column and eight sensor reading columns (sensor_a through sensor_h), one row per day. Plot each sensor as its own line on a single axes, each in a distinct color, with line width 1.0 and no markers. Label the x-axis 'Date' and the y-axis 'Reading'. Give the figure the title 'Daily Sensor Readings'. Place the legend outside the plot area on the right side so it covers no data, showing one entry per sensor in the order sensor_a through sensor_h. Rotate the x tick labels by 30 degrees. Save the figure to figure.png with a figure size of 10 by 5 inches.

## Required outputs

Leave these behind in `/app`, which is also the working directory. `plot.py` must
refer to these files by bare filename rather than by absolute path: the
verifier re-runs the script in a copied workspace, so an absolute output path
would point outside that copy.

Saving extra images is allowed; the answer is graded from the figure saved as
`figure.png`. Give any extra image a different *stem*: a second save named
`figure.<anything>` collides with the graded one.

1. **`/app/plot.py`** - the plotting script. Self-contained and re-runnable:
   `python plot.py` from `/app` must read `sensors.csv` and write `figure.png`
   with no arguments and no manual steps. It must read the readings out of
   `sensors.csv` rather than hard-coding them, and it must render the same
   image every time it runs.
2. **`/app/figure.png`** - the chart, as saved by that script.
3. **`/app/plotted_values.json`** - the readings shown in the chart, one entry
   per sensor. It must contain one JSON object with exactly eight keys,
   `sensor_a` through `sensor_h`, in no particular order. Each value must be a
   JSON array of exactly 30 JSON numbers: that sensor's reading for each of the
   30 days, in the CSV's row order (August 1 through August 30, 2026). No
   strings, no dates in the arrays.

   For example:
   `{"sensor_a": [0.0, 0.0, 0.0], "sensor_b": [0.0, 0.0, 0.0], "sensor_c": [0.0, 0.0, 0.0], "sensor_d": [0.0, 0.0, 0.0], "sensor_e": [0.0, 0.0, 0.0], "sensor_f": [0.0, 0.0, 0.0], "sensor_g": [0.0, 0.0, 0.0], "sensor_h": [0.0, 0.0, 0.0]}`

   (each array here must hold 30 numbers, not the 3 shown)

Do not modify `sensors.csv`.

Only matplotlib, numpy, pandas and Pillow are available, and there is no
network access; everything you need is already installed.