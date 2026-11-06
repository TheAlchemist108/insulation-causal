"""Figures: event study and subgroup savings. Single series each, so one hue."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "outputs"
BLUE, INK, INK2, GRID, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF})


def style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


es = pd.read_csv(OUT / "event_study.csv").sort_values("e")
pct = lambda v: -(np.exp(v) - 1) * 100
fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(0, color=INK2, lw=0.8); ax.axvline(-0.5, color=GRID, lw=1, ls="--")
ax.fill_between(es.e, pct(es.lo), pct(es.hi), color=BLUE, alpha=0.15, lw=0)
ax.plot(es.e, pct(es.att), color=BLUE, lw=2, marker="o", ms=6, mec=SURF, mew=2)
ax.set_xlabel("Years since cavity wall insulation (gas year; base = year before)")
ax.set_ylabel("Gas saving vs comparison homes (%)")
ax.set_title("Cavity wall insulation: gas savings by year", loc="left", color=INK, fontsize=12)
ax.text(-3.9, pct(es.hi).max() * 0.92, "before install", color=INK2, fontsize=9)
style(ax); fig.tight_layout(); fig.savefig(OUT / "fig_event_study.png", dpi=200)

sub = pd.read_csv(OUT / "subgroups.csv")
sub = sub[sub.group != "Q1 minus Q5"].iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7, 5.2))
y = np.arange(len(sub))
ax.hlines(y, pct(sub.hi), pct(sub.lo), color=BLUE, lw=2)
ax.plot(pct(sub.att), y, "o", color=BLUE, ms=7, mec=SURF, mew=2)
main = pd.read_csv(OUT / "att_table.csv").iloc[0]
ax.axvline(pct(main.att), color=INK2, lw=0.8, ls="--")
ax.text(pct(main.att) + 0.2, len(sub) - 0.4, f"all homes {pct(main.att):.1f}%", color=INK2, fontsize=9)
ax.set_yticks(y, [f"{g}  (n={int(n)})" for g, n in zip(sub.group, sub.n_treated)])
ax.set_xlabel("Gas saving from cavity wall insulation (%, 95% CI)")
ax.set_title("Who saves most?", loc="left", color=INK, fontsize=12)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "fig_subgroups.png", dpi=200)
print("ok")
