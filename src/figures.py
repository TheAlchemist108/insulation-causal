"""Paper figures 1 and 2 (PNG, for the LaTeX/arXiv version), drawn from outputs/ only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
O, V = R / "outputs", R / "outputs" / "revision"
F = R / "paper" / "figures"; F.mkdir(parents=True, exist_ok=True)
BLUE, GREY, INK, INK2, GRID = "#2a78d6", "#a8a7a2", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "DejaVu Sans"})
pct = lambda v: -(np.exp(v) - 1) * 100


def style(ax, axis="y"):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=0.7); ax.set_axisbelow(True)


# Figure 1: event study, this paper vs two-way fixed effects (all gas-heated homes)
es = pd.read_csv(V / "event_study_joint.csv")
es = pd.concat([es, pd.DataFrame([dict(e=-1, att=0, lo=0, hi=0)])]).sort_values("e")
tw = pd.read_csv(V / "twfe_event.csv"); tw = tw[tw.spec.str.startswith("T1")]
tw = pd.concat([tw, pd.DataFrame([dict(e=-1, coef=0, lo=0, hi=0)])]).sort_values("e")
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.axhline(0, color=INK2, lw=0.7); ax.axvline(-0.5, color=GREY, lw=0.8, ls="--")
ax.errorbar(tw.e + 0.13, pct(tw.coef), yerr=[pct(tw.coef) - pct(tw.hi), pct(tw.lo) - pct(tw.coef)], fmt="o", ms=4.5,
            mfc="white", mec=GREY, ecolor=GREY, elinewidth=1.2, label="Two-way fixed effects, all gas-heated homes, 2005–17")
ax.errorbar(es.e - 0.1, pct(es.att), yerr=[pct(es.att) - pct(es.hi), pct(es.lo) - pct(es.att)], fmt="o", ms=5,
            color=BLUE, ecolor=BLUE, elinewidth=1.6, label="Doubly robust stacked DiD (this paper)")
ax.set_xticks(range(-5, 6)); ax.set_xlabel("Years since installation (gas years); base = year before installation")
ax.set_ylabel("Gas saving vs comparison homes (%)"); ax.set_ylim(-4.5, 14)
ax.legend(frameon=False, loc="upper left", fontsize=8); style(ax)
fig.tight_layout(); fig.savefig(F / "fig1_event_study.png", dpi=300); plt.close(fig)

# Figure 2: subgroup savings
sub = pd.read_csv(O / "subgroups.csv"); sub = sub[sub.group != "Q1 minus Q5"].reset_index(drop=True)
lab = {"IMD Q1": "IMD Q1 (most deprived)", "IMD Q5": "IMD Q5 (least deprived)"}
sub["label"] = [f"{lab.get(g, g)} ({int(n)})" for g, n in zip(sub.group, sub.n_treated)]
sub = sub.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
y = np.arange(len(sub))
hl = sub.group.isin(["IMD Q1", "Council tax A"])
for i, r in sub.iterrows():
    c = BLUE if hl[i] else GREY
    ax.hlines(i, pct(r.hi), pct(r.lo), color=c, lw=1.8); ax.plot(pct(r.att), i, "o", color=c, ms=5.5)
main = pd.read_csv(O / "att_table.csv").iloc[0]
ax.axvline(pct(main.att), color=INK2, lw=0.8, ls="--")
ax.text(pct(main.att) + 0.2, len(sub) - 0.5, f"all homes {pct(main.att):.1f}%", color=INK2, fontsize=8)
ax.set_yticks(y, sub.label, fontsize=8); ax.set_xlabel("Gas saving from cavity wall insulation (%, 95% CI)")
style(ax, "x"); fig.tight_layout(); fig.savefig(F / "fig2_subgroups.png", dpi=300); plt.close(fig)

print("figures written")
