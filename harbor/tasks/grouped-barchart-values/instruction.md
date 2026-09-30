# grouped-barchart-values

<!-- Copied from tasks/grouped-barchart-values/task.json. This is the prompt the three fixed agents
     were given, so do NOT reword it: check-submission holds both your runs
     and this file to the descriptor's wording. Add what your verifier needs
     under the heading below instead. -->

Using matplotlib, read product_sales.csv, which contains columns product, quarter, and sales. Draw a grouped bar chart of sales by quarter, one group per product with two adjacent bars (one per quarter) inside each group, using the colors '#1f77b4' for Q1 and '#ff7f0e' for Q2. Print the sales value above each bar in black text with font size 9, formatted with no decimals. Label the x-axis 'Product', the y-axis 'Sales', and give the figure the title 'Product Sales by Quarter'. Add a legend identifying Q1 and Q2, placed in the upper right corner but not covering any bars. Save the figure to figure.png with a figure size of 9 by 6 inches.

## Required outputs

Leave these behind in `/app`, which is also the working directory. `plot.py` must
refer to these files by bare filename rather than by absolute path: the
verifier re-runs the script in a copied workspace, so an absolute output path
would point outside that copy.

Saving extra images is allowed; the answer is graded from the figure saved as
`figure.png`. Give any extra image a different *stem*: a second save named
`figure.<anything>` collides with the graded one.

1. **`/app/plot.py`** - the plotting script. Self-contained and re-runnable:
   `python plot.py` from `/app` must read `product_sales.csv` and write
   `figure.png` with no arguments and no manual steps. It must read the sales
   out of `product_sales.csv` rather than hard-coding them, and it must render
   the same image every time it runs.
2. **`/app/figure.png`** - the chart, as saved by that script.
3. **`/app/plotted_values.json`** - the sales values shown in the chart, one
   entry per product. It must contain one JSON object with exactly five keys,
   the products `alpha`, `beta`, `gamma`, `delta`, `epsilon`, in no particular
   order. Each value must itself be a JSON object with exactly two keys `Q1`
   and `Q2`, each a JSON number giving that product's sales for that quarter as
   shown in the chart. No strings.

   For example:
   `{"alpha": {"Q1": 0.0, "Q2": 0.0}, "beta": {"Q1": 0.0, "Q2": 0.0}, "gamma": {"Q1": 0.0, "Q2": 0.0}, "delta": {"Q1": 0.0, "Q2": 0.0}, "epsilon": {"Q1": 0.0, "Q2": 0.0}}`

Do not modify `product_sales.csv`.

Only matplotlib, numpy, pandas and Pillow are available, and there is no
network access; everything you need is already installed.