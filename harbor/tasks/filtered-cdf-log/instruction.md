# filtered-cdf-log

<!-- Copied from tasks/filtered-cdf-log/task.json. This is the prompt the three fixed agents
     were given, so do NOT reword it: check-submission holds both your runs
     and this file to the descriptor's wording. Add what your verifier needs
     under the heading below instead. -->

Using matplotlib, read latencies.csv, which contains a single column latency_ms with 500 request latency measurements. First keep only the measurements above 50 ms, then compute the empirical cumulative distribution: for each distinct latency value, the fraction of kept measurements less than or equal to it. Plot this CDF as a single blue line with line width 2. Label the x-axis 'Latency (ms)' and the y-axis 'Cumulative fraction', and give the figure the title 'CDF of Request Latencies above 50ms'. Set the x-axis to a logarithmic scale, with the x-axis limits exactly from 50 to 1000. Draw a thin dashed horizontal reference line at 0.5 in grey, and annotate the point where the CDF crosses 0.5 with the text 'median' in black. Save the figure to figure.png with a figure size of 8 by 5 inches.

## Required outputs

Leave these behind in `/app`, which is also the working directory. `plot.py` must
refer to these files by bare filename rather than by absolute path: the
verifier re-runs the script in a copied workspace, so an absolute output path
would point outside that copy.

Saving extra images is allowed; the answer is graded from the figure saved as
`figure.png`. Give any extra image a different *stem*: a second save named
`figure.<anything>` collides with the graded one.

1. **`/app/plot.py`** - the plotting script. Self-contained and re-runnable:
   `python plot.py` from `/app` must read `latencies.csv` and write
   `figure.png` with no arguments and no manual steps. It must read the
   numbers out of `latencies.csv` rather than hard-coding them, and it must
   render the same image every time it runs.
2. **`/app/figure.png`** - the chart, as saved by that script.
3. **`/app/plotted_values.json`** - summary values of the plotted CDF. It must
   contain one JSON object with exactly these five keys, each a JSON number
   (no strings, no unit suffixes):
   - `kept_count`: how many measurements are strictly greater than 50 ms.
   - `median_ms`: the smallest kept latency whose CDF value reaches 0.5, in ms.
   - `f_100`: the fraction of kept measurements less than or equal to 100 ms.
   - `f_200`: the fraction of kept measurements less than or equal to 200 ms.
   - `f_500`: the fraction of kept measurements less than or equal to 500 ms.

   For example: `{"kept_count": 0, "median_ms": 0.0, "f_100": 0.0, "f_200": 0.0, "f_500": 0.0}`.

Do not modify `latencies.csv`.

Only matplotlib, numpy, pandas and Pillow are available, and there is no
network access; everything you need is already installed.