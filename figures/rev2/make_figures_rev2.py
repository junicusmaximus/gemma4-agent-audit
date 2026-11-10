#!/usr/bin/env python3
"""Rev2 (style revision): draw Fig. 1, Fig. 2a, Fig. 2b, Fig. 2 and the cover from ../figure_data.csv only.
Output goes next to this script (rev2/). Plotted values are identical to the current files; only text changed.

Run: py -3 -X utf8 make_figures_rev2.py   (needs matplotlib; no server access)
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "figure_data.csv"
DPI = 220

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, VIOLET, GREEN = "#2a78d6", "#eb6834", "#6250d6", "#008300"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "svg.fonttype": "path", "axes.unicode_minus": True,
})


def read_rows():
    with open(DATA, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


ROWS = read_rows()

# display text for labels that come from the CSV (the CSV keys stay unchanged)
LABEL_MAP = {
    "tool-call text": "tool call text",
    "tool-response opener": "tool response opener",
    "mask empty thoughts (M1 − M0)": "mask empty thoughts\n(M1 minus M0)",
    "match serving template (M0mc − M0)": "match serving template\n(M0mc minus M0)",
    "train on QAT weights (M0q − M0)": "train on QAT weights\n(M0q minus M0)",
    "match serving template (M0qmc − M0q)": "match serving template\n(M0qmc minus M0q)",
    "restore dropped history, partly (Hq1mc − M0qmc)": "partly restore dropped history\n(Hq1mc minus M0qmc)",
    "mask empty thoughts, seed 1 (M1qmc − M0qmc)": "mask empty thoughts for seed 1\n(M1qmc minus M0qmc)",
    "mask empty thoughts, seed 2 (M1qmc-s2 − M0qmc)": "mask empty thoughts for seed 2\n(M1qmc-s2 minus M0qmc)",
}


def L(text):
    return LABEL_MAP.get(text, text)


def sel(figure, panel=None, series=None, x_label=None, seed=None):
    out = []
    for r in ROWS:
        if r["figure"] != figure:
            continue
        if panel is not None and r["panel"] != panel:
            continue
        if series is not None and r["series"] != series:
            continue
        if x_label is not None and r["x_label"] != x_label:
            continue
        if seed is not None and r["decoding_seed"] != seed:
            continue
        out.append(r)
    return out


def one(*a, **k):
    rs = sel(*a, **k)
    assert len(rs) == 1, (a, k, len(rs))
    return rs[0]


def fnum(s):
    return float(s)


def clean(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Fig. 1
def draw_fig1(fig):
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.72],
                  left=0.215, right=0.945, top=0.90, bottom=0.04, hspace=0.42, wspace=0.62)
    arms = ["A1 data (arm_ordinary_tokmatched)", "A2 data (arm_short)"]
    short = {arms[0]: "A1 data", arms[1]: "A2 data"}
    colors = {arms[0]: BLUE, arms[1]: ORANGE}

    # (a) supervised-token categories
    ax = fig.add_subplot(gs[0, 0])
    cats = [r["x_label"] for r in sel("fig1", "a_token_categories", arms[0])]
    h = 0.36
    for j, arm in enumerate(arms):
        for i, cat in enumerate(cats):
            v = fnum(one("fig1", "a_token_categories", arm, cat)["value"])
            y = i + (j - 0.5) * h * 1.05
            ax.barh(y, v, height=h, color=colors[arm], zorder=3)
            ax.text(v + 1.2, y, f"{v:.1f}" if v >= 0.1 else "0.0", va="center", ha="left", fontsize=9, color=INK2)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([L(c) for c in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of supervised tokens (%)")
    clean(ax, "x")
    ax.set_title("Supervised token"+chr(10)+"categories", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=6)

    # (b) shares of completions / chunks
    ax = fig.add_subplot(gs[0, 1])
    labs = [("empty-thought completions", "Empty thought\ncompletions"),
            ("history-dropped chunks", "History dropped\nchunks")]
    for j, arm in enumerate(arms):
        for i, (key, _) in enumerate(labs):
            r = one("fig1", "b_shares", arm, key)
            v = fnum(r["value"])
            y = i + (j - 0.5) * h * 1.05
            ax.barh(y, v, height=h, color=colors[arm], zorder=3)
            ax.text(v + 1.2, y, f"{v:.1f}", va="center", ha="left", fontsize=9, color=INK2)
    ax.set_ylim(len(labs) - 0.5 + 0.3, -0.5 - 0.3)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels([lab for _, lab in labs])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of completions or chunks")
    clean(ax, "x")
    ax.set_title("Empty thoughts and"+chr(10)+"dropped history", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=6)
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=7, color=colors[a], label=short[a]) for a in arms]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5, handletextpad=0.3, borderaxespad=0.2)

    # (c) token strip
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.set_title("Token strip after a tool response", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=4)
    cnt = {r["x_label"]: r for r in sel("fig1", "c_token_counts")}
    toks = sel("fig1", "c_token_strip", "training context")
    label_for = {"<tool_response|>": "<tool_response|>", "<|channel>": "<|channel>", "thought": "thought",
                 "newline": "newline"}
    x0, gap = 1.0, 1.2
    widths = [23.0, 15.0, 15.0, 15.0]
    ytrain, yserved, hb = 36.0, 5.0, 12.0

    def box(xb, yb, wb, name, tid, opener):
        ax.add_patch(FancyBboxPatch((xb, yb), wb, hb, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc="#f7d9cc" if opener else "#ecebe6", ec=ORANGE if opener else AXIS, lw=1.2))
        ax.text(xb + wb / 2, yb + hb * 0.66, name, ha="center", va="center", fontsize=9, color=INK)
        ax.text(xb + wb / 2, yb + hb * 0.26, f"id {tid}", ha="center", va="center", fontsize=9, color=INK2)

    ax.text(x0, ytrain + hb + 3.2, "Training context", ha="left", va="center", fontsize=10, color=INK,
            fontweight="bold")
    ax.text(x0, yserved + hb + 3.2, "Served context", ha="left", va="center", fontsize=10, color=INK,
            fontweight="bold")
    xb = x0
    for i, r in enumerate(toks):
        box(xb, ytrain, widths[i], label_for[r["x_label"]], int(float(r["value"])), i > 0)
        xb += widths[i] + gap
    served = one("fig1", "c_token_strip", "served context")
    box(x0, yserved, widths[0], "<tool_response|>", int(float(served["value"])), False)
    ax.add_patch(FancyArrowPatch((x0 + widths[0] + 0.8, yserved + hb / 2), (x0 + widths[0] + 9, yserved + hb / 2),
                                 arrowstyle="-|>", mutation_scale=10, color=INK2, lw=1.2))
    n_span = cnt["context spans after a completion that end with the opener"]
    n_call = cnt["M0 calls whose prompt ends at <tool_response|> (no opener)"]
    den_span = n_span["unit"].split("of ")[1].rstrip(")")
    den_call = n_call["unit"].split("of ")[1].split(" ")[0]
    nl = chr(10)
    ax.text(xb + 0.8, ytrain + hb / 2,
            f"opener sits in the{nl}unsupervised context:{nl}{int(float(n_span['value']))} of {int(den_span)}{nl}"
            "spans after a completion",
            ha="left", va="center", fontsize=9, color=INK2, linespacing=1.3)
    ax.text(x0 + widths[0] + 11, yserved + hb / 2,
            f"prompt ends here and the{nl}opener is absent:{nl}{int(float(n_call['value']))} of {den_call} "
            f"served calls of one M0 trace",
            ha="left", va="center", fontsize=9, color=INK2, linespacing=1.3)


# ------------------------------------------------------------------ Fig. 2a
GROUP_STYLE = {
    "bf16 panel": (VIOLET, "o"),
    "across panels (base weights bf16 to QAT)": (INK2, "D"),
    "QAT panel": (GREEN, "s"),
}
GROUP_TITLE = {
    "bf16 panel": "bf16 base panel",
    "across panels (base weights bf16 to QAT)": "Across panels (bf16 to QAT base)",
    "QAT panel": "QAT dequant panel",
}


def draw_fig2a(fig, tag=None):
    contrasts = []
    for r in sel("fig2a", "loop"):
        contrasts.append((r["series"], r["x_label"]))
    # y layout: groups with gaps, top to bottom
    ypos, y = {}, 0.0
    last = None
    ghead = {}
    for grp, lab in contrasts:
        if last is not None and grp != last:
            y += 1.15
        if grp != last:
            ghead[grp] = y - 1.15
        ypos[(grp, lab)] = y
        y += 1.0
        last = grp
    ymax = y - 0.3

    gs = GridSpec(1, 5, figure=fig, width_ratios=[1.85, 1, 1, 1, 0.78], left=0.035, right=0.965, top=0.885,
                  bottom=0.14, wspace=0.07)
    ax_lab = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, k]) for k in (1, 2, 3)]
    ax_rep = fig.add_subplot(gs[0, 4])
    for a in axes + [ax_lab, ax_rep]:
        a.set_ylim(ymax, -1.9)
    ax_lab.axis("off")
    ax_rep.axis("off")
    xlim = (-64, 46)
    heads = {"loop": "Loop rate", "submit": "Submit rate", "M5": "Repeat after"+chr(10)+"error (M5)"}
    refs = {r["series"]: r for r in sel("fig2a", "reference_repair")}
    for ax, met in zip(axes, ("loop", "submit", "M5")):
        ax.set_xlim(*xlim)
        ax.set_xticks([-50, 0, 25])
        ax.set_yticks([])
        clean(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color=INK, lw=1.0, zorder=2)
        ax.set_title(heads[met], fontsize=10.5, fontweight="bold", color=INK, pad=5)
        for grp, lab in contrasts:
            r = one("fig2a", met, grp, lab)
            col, mk = GROUP_STYLE[grp]
            yy = ypos[(grp, lab)]
            v, lo, hi = fnum(r["value"]), fnum(r["lower"]), fnum(r["upper"])
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.8, solid_capstyle="round", zorder=3)
            ax.plot([v], [yy], marker=mk, color=col, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                    zorder=4, linestyle="")
    # labels and group headers
    for grp, lab in contrasts:
        ax_lab.text(1.0, ypos[(grp, lab)], L(lab), ha="right", va="center", fontsize=9.5, color=INK,
                    transform=ax_lab.get_yaxis_transform())
    for grp, yh in ghead.items():
        col, _ = GROUP_STYLE[grp]
        sub = ""
        if grp == "bf16 panel":
            sub = f"untuned base A0cap: {int(float(refs[grp]['value']))}/120"
        if grp == "QAT panel":
            sub = f"untuned base A0mc: {int(float(refs[grp]['value']))}/120"
        ax_lab.text(0.0, yh, GROUP_TITLE[grp], ha="left", va="center", fontsize=10, fontweight="bold", color=col,
                    transform=ax_lab.get_yaxis_transform())
        if sub:
            ax_lab.text(0.0, yh + 0.5, sub, ha="left", va="center", fontsize=9, color=INK2,
                        transform=ax_lab.get_yaxis_transform())
    # repair column
    ax_rep.set_title("Repair: resolved\nof 120 for trained\nvs comparator", fontsize=9, fontweight="bold", color=INK,
                     pad=5, linespacing=1.15)
    for grp, lab in contrasts:
        t = int(float(one("fig2a", "repair_treated", grp, lab)["value"]))
        c = int(float(one("fig2a", "repair_comparator", grp, lab)["value"]))
        ax_rep.text(0.5, ypos[(grp, lab)], f"{t} vs {c}", ha="center", va="center", fontsize=9.5, color=INK,
                    transform=ax_rep.get_yaxis_transform())
    fig.text(0.575, 0.05, "paired effect in percentage points (trained minus comparator) with 95% CI",
             ha="center", va="center", fontsize=9.5, color=INK2)
    if tag:
        fig.text(0.02, 0.975, tag, fontsize=13, fontweight="bold", color=INK, va="top")


# ------------------------------------------------------------------ Fig. 2b
ARM_STYLE = {
    "untuned base (A0mc)": (INK2, "Untuned base\n(A0mc)"),
    "self-generated (A_onpol), training seed 1": (BLUE, "Self generated\n(A_onpol)\ntraining\nseed 1"),
    "self-generated (A_onpol), training seed 2": (BLUE, "Self generated\n(A_onpol)\ntraining\nseed 2"),
    "teacher subset (T_sub)": (ORANGE, "Teacher subset\n(T_sub)"),
}
SEED_MARK = {"1": "o", "2": "s", "3": "^"}
DODGE = {"1": -0.24, "2": 0.0, "3": 0.24}
THR_LABEL = {
    ("loop", 0.30): ("not collapsed at ≤ 0.30", "top"),
    ("loop", 0.40): ("collapsed above 0.40", "bottom"),
    ("submit", 0.80): ("not collapsed at ≥ 0.80", "bottom"),
    ("submit", 0.70): ("collapsed below 0.70", "top"),
}


def draw_fig2b(fig, tag=None, cover=False):
    k = 1.25 if cover else 1.0
    geom = (dict(left=0.075, right=0.975, top=0.615, bottom=0.255) if cover
            else dict(left=0.105, right=0.965, top=0.855, bottom=0.265))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[4, 3], hspace=0.1, wspace=0.12, **geom)
    facets = [("holdout", "Known holdout\n(40 tasks and 3 decoding seeds)"),
              ("external", "External panel\n(12 tasks and 3 decoding seeds)")]
    ylabels = {"loop": "Loop rate", "submit": "Submit rate"}
    for ri, met in enumerate(("loop", "submit")):
        for ci, (facet, title) in enumerate(facets):
            ax = fig.add_subplot(gs[ri, ci])
            panel = f"{facet}-{met}"
            arms = [k for k in ARM_STYLE if sel("fig2b", panel, k)]
            n = len(arms)
            ax.set_xlim(-0.55, n - 0.45)
            ax.set_ylim(-0.04, 1.04)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
            clean(ax, "y")
            ax.spines["bottom"].set_color(AXIS)
            if ci == 1:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(ylabels[met], fontsize=10 * k)
            if ri == 0:
                ax.set_title(title, fontsize=10.5 * k, fontweight="bold", color=INK, pad=6, linespacing=1.2)
                ax.tick_params(labelbottom=False, length=0)
            ax.tick_params(axis="y", labelsize=9.5 * k)
            ax.set_xticks(range(n))
            if ri == 1:
                ax.set_xticklabels([ARM_STYLE[a][1] for a in arms], fontsize=9 * k, linespacing=1.15)
            # thresholds
            for r in sel("fig2b", panel, "frozen threshold"):
                val = fnum(r["value"])
                ax.axhline(val, color=MUTED, lw=1.0, zorder=1)
            xs_lab = -0.5 if met == "loop" else n - 0.5
            ha = "left" if met == "loop" else "right"
            for r in sel("fig2b", panel, "frozen threshold"):
                val = fnum(r["value"])
                txt, va = THR_LABEL[(met, round(val, 2))]
                off = 0.012 if va == "bottom" else -0.012
                ax.text(xs_lab, val + off, txt, ha=ha, va=va, fontsize=9 * k, color=INK2)
            for i, arm in enumerate(arms):
                col = ARM_STYLE[arm][0]
                m = one("fig2b", panel, arm, seed="mean")
                ax.plot([i - 0.33, i + 0.33], [fnum(m["value"])] * 2, color=col, lw=2.2 * k, zorder=3,
                        solid_capstyle="butt")
                for s in ("1", "2", "3"):
                    r = one("fig2b", panel, arm, seed=s)
                    ax.plot([i + DODGE[s]], [fnum(r["value"])], marker=SEED_MARK[s], color=col, markersize=6.5 * k,
                            markeredgecolor=SURFACE, markeredgewidth=0.8, linestyle="", zorder=4)
    handles = [Line2D([0], [0], marker=SEED_MARK[s], linestyle="", color=INK2, markersize=6.5 * k,
                      label=f"decoding seed {s}") for s in ("1", "2", "3")]
    handles.append(Line2D([0], [0], color=INK2, lw=2.2, label="three seed mean"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5 * k, bbox_to_anchor=(0.54, 0.005),
               handletextpad=0.4, columnspacing=1.4)
    fig.text(0.54, 0.105 if not cover else 0.105, "Grey lines mark the frozen collapse criteria. On the external panel submit means a parsed submit_patch attempt.",
             ha="center", va="center", fontsize=9 * k, color=INK2)
    if tag:
        fig.text(0.02, 0.975, tag, fontsize=13, fontweight="bold", color=INK, va="top")


def draw_cover(fig):
    fig.text(0.04, 0.955, "Same model and recipe with a different trajectory source.\nThe teacher subset adapter met the collapse criteria.\nThe self generated adapters did not.",
             ha="left", va="top", fontsize=22, fontweight="bold", color=INK, linespacing=1.2)
    fig.text(0.04, 0.775, "Gemma 4 E4B with LoRA finetuning at equal update counts on a known holdout and a 12 task external panel",
             ha="left", va="top", fontsize=13.5, color=INK2)
    draw_fig2b(fig, cover=True)


# ------------------------------------------------------------------ output
def save(fig, stem):
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{stem}.{ext}", dpi=DPI)
    plt.close(fig)
    print("wrote", stem)


def main():
    f = plt.figure(figsize=(7.2, 5.6))
    draw_fig1(f)
    save(f, "fig1")

    f = plt.figure(figsize=(7.2, 5.6))
    draw_fig2a(f)
    save(f, "fig2a")

    f = plt.figure(figsize=(7.2, 5.3))
    draw_fig2b(f)
    save(f, "fig2b")

    f = plt.figure(figsize=(12, 6.75))
    draw_cover(f)
    f.savefig(HERE / "cover.png", dpi=100)
    plt.close(f)
    print("wrote cover")

    f = plt.figure(figsize=(7.2, 10.9))
    sa, sb = f.subfigures(2, 1, height_ratios=[5.6, 5.3], hspace=0.0)
    draw_fig2a(sa, tag="a")
    draw_fig2b(sb, tag="b")
    save(f, "fig2")


if __name__ == "__main__":
    main()
