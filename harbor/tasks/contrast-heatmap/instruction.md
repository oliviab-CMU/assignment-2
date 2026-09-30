# contrast-heatmap

<!-- Copied from tasks/contrast-heatmap/task.json. This is the prompt the three fixed agents
     were given, so do NOT reword it: check-submission holds both your runs
     and this file to the descriptor's wording. Add what your verifier needs
     under the heading below instead. -->

Using matplotlib, read correlation.csv, which holds a 6x6 correlation matrix with row and column headers in the first column and first row. Draw a heatmap of the correlation values using the 'coolwarm' colormap, with the correlation value printed inside every cell in black text with font size 12, formatted to exactly two decimal places. Label the x-axis 'Variable' and the y-axis 'Variable', rotate the x tick labels by 45 degrees so they do not overlap, and give the figure the title 'Correlation Matrix'. Add a colorbar labeled 'Correlation'. Save the figure to figure.png with a figure size of 8 by 6 inches.

## Required outputs

Leave these behind in `/app`, which is also the working directory. `plot.py` must
refer to these files by bare filename rather than by absolute path: the
verifier re-runs the script in a copied workspace, so an absolute output path
would point outside that copy.

Saving extra images is allowed; the answer is graded from the figure saved as
`figure.png`. Give any extra image a different *stem*: a second save named
`figure.<anything>` collides with the graded one.

1. **`/app/plot.py`** - the plotting script. Self-contained and re-runnable:
   `python plot.py` from `/app` must read `correlation.csv` and write
   `figure.png` with no arguments and no manual steps. It must read the
   correlation values out of `correlation.csv` rather than hard-coding them, and
   it must render the same image every time it runs.
2. **`/app/figure.png`** - the chart, as saved by that script.
3. **`/app/plotted_values.json`** - the correlation values shown in the chart.
   It must contain one JSON object with exactly six keys, `var1` through `var6`.
   Each value must itself be a JSON object with exactly six keys `var1` through
   `var6`, each a JSON number giving the correlation between the two variables
   as shown in the heatmap (a number between -1 and 1). No strings.

   For example:
   `{"var1": {"var1": 0.0, "var2": 0.0, "var3": 0.0, "var4": 0.0, "var5": 0.0, "var6": 0.0}, "var2": {"var1": 0.0, "var2": 0.0, "var3": 0.0, "var4": 0.0, "var5": 0.0, "var6": 0.0}, "var3": {"var1": 0.0, "var2": 0.0, "var3": 0.0, "var4": 0.0, "var5": 0.0, "var6": 0.0}, "var4": {"var1": 0.0, "var2": 0.0, "var3": 0.0, "var4": 0.0, "var5": 0.0, "var6": 0.0}, "var5": {"var1": 0.0, "var2": 0.0, "var3": 0.0, "var4": 0.0, "var5": 0.0, "var6": 0.0}, "var6": {"var1": 0.0, "var2": 0.0, "var3": 0.0, "var4": 0.0, "var5": 0.0, "var6": 0.0}}`

Do not modify `correlation.csv`.

Only matplotlib, numpy, pandas and Pillow are available, and there is no
network access; everything you need is already installed.