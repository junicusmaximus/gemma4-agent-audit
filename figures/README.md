# Figures

Every plotted number is in `figure_data.csv`: one row per plotted value with the figure and the source file and the key path inside it and the source hash and the raw value. The source files are the sealed reports under `analyses/` of this package (their paths in the csv use the placeholder `PROJECT_ROOT`). The plotting scripts read only `figure_data.csv`.

## Files

| file | content |
|---|---|
| `figure_data.csv` | the plotted values and their provenance |
| `rev3/fig1.png` and `rev3/fig1.svg` | Fig. 1 contract defects in the kit training data |
| `rev3/fig2a.png` and `rev3/fig2a.svg` | Fig. 2a lever grid |
| `rev3/fig2b.png` and `rev3/fig2b.svg` | Fig. 2b termination by trajectory source |
| `rev3/fig2.png` and `rev3/fig2.svg` | Fig. 2 panels a and b in one file |
| `rev3/card.png` | the writeup card |
| `rev3/make_figures_rev3.py` | draws everything in `rev3/` from `figure_data.csv` |
| `rev2/cover.png` | the cover image (1200 by 675 px), unchanged since the second revision |
| `rev2/make_figures_rev2.py` | draws the cover from the `fig2b` rows of `figure_data.csv` (it redraws the other rev2 figures too) |

To redraw run `python3 make_figures_rev3.py` inside `rev3/`. It needs matplotlib and no server access. The output goes next to the script.

## Fig. 1 contract defects in the kit training data
Left and middle: for the A1 data and the A2 data the share of supervised tokens by category and the share of empty-thought completions and the share of history-dropped chunks. Bottom: the token strip after the tool response marker with the training context against the served context (schematic with token ids and counts).

## Fig. 2a lever grid
Paired effects with 95 % CIs of each single-lever contrast on loop rate and submit rate and repeat after error (M5). The bf16 and QAT panels are in separate bands and are never pooled. Repair counts are in the right column and the untuned references are in the band headers. The contrast M0q minus M0 compares a QAT-trained arm with a bf16-trained arm so it sits in its own band. M5 effects are paired on tasks with an error observation in both arms and the number of tasks per row is in the `note` column of the csv. The repair counts are pooled resolved counts out of 120 (3 decoding seeds and 40 tasks) with the trained arm first.

## Fig. 2b termination by trajectory source
Loop and submit rate per decoding seed (markers) and the three-seed mean (bar) for the untuned base and the self-generated arm (two training seeds) and the teacher subset. The known holdout and the 12-task external panel are shown separately and no base ran on the external panel. Grey lines mark the frozen collapse criteria. On the external panel submit means a parsed submit_patch attempt (36 and 36 and 3 rollouts) and not a non-empty captured patch (33 and 36 and 0). The external repair counts are deliberately not drawn.

## Not included
The script that derived `figure_data.csv` from the reports is not in this package because it reads files that are withheld (the serialization audit report that embeds the organiser's agent system prompt). The csv holds the values it produced.
