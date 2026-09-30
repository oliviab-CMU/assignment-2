# normalized-stacked-share

<!-- Copied from tasks/normalized-stacked-share/task.json. This is the prompt the three fixed agents
     were given, so do NOT reword it: check-submission holds both your runs
     and this file to the descriptor's wording. Add what your verifier needs
     under the heading below instead. -->

Using matplotlib, read platform_votes.csv, which contains columns age_group and platform, with one row per respondent. Draw a horizontal stacked bar chart showing, for each age group, the percentage share of respondents per platform (stack segments summing to 100 for each bar), not the raw counts. Order age groups on the y-axis from youngest at the top to oldest at the bottom, and stack platform segments in a fixed order: video, text, audio, with the colors '#2ca02c', '#d62728', '#9467bd' respectively. Label the x-axis 'Share of respondents (%)' and the y-axis 'Age group', and give the figure the title 'Platform Share by Age Group'. Add a legend identifying the three platforms, placed below the plot. Save the figure to figure.png with a figure size of 9 by 5 inches.

## Required outputs

Leave these behind in `/app`, which is also the working directory. `plot.py` must
refer to these files by bare filename rather than by absolute path: the
verifier re-runs the script in a copied workspace, so an absolute output path
would point outside that copy.

Saving extra images is allowed; the answer is graded from the figure saved as
`figure.png`. Give any extra image a different *stem*: a second save named
`figure.<anything>` collides with the graded one.

1. **`/app/plot.py`** - the plotting script. Self-contained and re-runnable:
   `python plot.py` from `/app` must read `platform_votes.csv` and write
   `figure.png` with no arguments and no manual steps. It must read the
   counts out of `platform_votes.csv` rather than hard-coding them, and it
   must render the same image every time it runs.
2. **`/app/figure.png`** - the chart, as saved by that script.
3. **`/app/plotted_values.json`** - the percentage shares shown in the chart.
   It must contain one JSON object with exactly five keys, the age groups
   `18-24`, `25-34`, `35-44`, `45-54`, `55+`. Each value must itself be a
   JSON object with exactly three keys `video`, `text`, `audio`, each a JSON
   number giving that platform's percentage share of the age group (a number
   between 0 and 100; the three must sum to 100). No strings, no raw counts.

   For example:
   `{"18-24": {"video": 0.0, "text": 0.0, "audio": 0.0}, "25-34": {"video": 0.0, "text": 0.0, "audio": 0.0}, "35-44": {"video": 0.0, "text": 0.0, "audio": 0.0}, "45-54": {"video": 0.0, "text": 0.0, "audio": 0.0}, "55+": {"video": 0.0, "text": 0.0, "audio": 0.0}}`

Do not modify `platform_votes.csv`.

Only matplotlib, numpy, pandas and Pillow are available, and there is no
network access; everything you need is already installed.