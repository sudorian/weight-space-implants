#!/usr/bin/env python3
"""Draw the README's results figure, in a light and a dark version.

Every number is read from a result file in this repository, and checked against
the numbers quoted in README.md before anything is drawn:

  run2-gpt2/artifacts/baseline_quick.json         GPT-2 unedited, run 2's own count
  independent-audit/run2/results/stage1.json       GPT-2 edited (G3-robust), fresh problems
  independent-audit/run2/results/stage2.json       GPT-2 edited, fresh prose
  independent-audit/run3/results/stage1.json       Gemma unedited and edited (A313), fresh problems
  independent-audit/run3/results/stage2.json       Gemma edited, fresh prose

Usage:  pip install matplotlib
        python figures/make_results_figure.py            (writes figures/results-light.png and results-dark.png)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

HERE = Path(__file__).resolve().parent
LIMIT = 1.5  # the text bar: at least 98.5% top-1 agreement, so at most 1.5% of predictions may change

# Light: GitHub's light background. Dark: GitHub's default dark background.
# Two steps of one blue: the lighter step (nearer the background) is the unedited model.
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#77756f",
                  grid="#e1e0d9", axis="#c3c2b7", before="#86b6ef", after="#2a78d6"),
    "dark": dict(surface="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#9a988f",
                 grid="#262a30", axis="#3a3f46", before="#1c5cab", after="#3987e5"),
}


def load(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text())


def read_results(root: Path) -> list[dict]:
    base2 = load(root, "run2-gpt2/artifacts/baseline_quick.json")["arithmetic"]["G3"]
    fresh2 = load(root, "independent-audit/run2/results/stage1.json")["G3-robust"]["random"]
    prose2 = load(root, "independent-audit/run2/results/stage2.json")["pooled_prose"]["G3-robust"]
    arith3 = load(root, "independent-audit/run3/results/stage1.json")
    text3 = load(root, "independent-audit/run3/results/stage2.json")
    domains = [text3[d] for d in ("fiction", "web", "speeches")]
    agree3 = sum(d["top1_agreement"] * d["tokens"] for d in domains) / sum(d["tokens"] for d in domains)
    rows = [
        dict(model="GPT-2 small", run="run 2",
             before=(base2["correct"], base2["count"]), after=(fresh2["correct"], fresh2["count"]),
             agree=prose2["top1_agreement"]),
        dict(model="Gemma 3 270M IT", run="run 3",
             before=(arith3["host_uniform"]["correct"], arith3["host_uniform"]["count"]),
             after=(arith3["a313_uniform"]["correct"], arith3["a313_uniform"]["count"]),
             agree=agree3),
    ]
    quoted = {  # as written in README.md
        "GPT-2 small": ((0, 2000), (19988, 20000), "98.66"),
        "Gemma 3 270M IT": ((65, 25000), (24960, 25000), "97.73"),
    }
    for r in rows:
        before, after, agree = quoted[r["model"]]
        got = (r["before"], r["after"], f"{100 * r['agree']:.2f}")
        if got != (before, after, agree):
            raise SystemExit(f"{r['model']}: result files give {got}, README quotes {(before, after, agree)}")
    return rows


def pct(n: int, d: int) -> float:
    return 100.0 * n / d


def pct_label(v: float) -> str:
    return "0%" if v == 0 else f"{v:.2f}%"


def rounded_bar(ax, y: float, width: float, height: float, color: str, radius_px: float = 8.0) -> None:
    """Horizontal bar from 0 to width: rounded at the data end, square at the baseline (one path)."""
    (x0, y0), (x1, y1) = ax.transData.transform((0, 0)), ax.transData.transform((1, 1))
    rx = min(radius_px / abs(x1 - x0), width / 2)
    ry = min(radius_px / abs(y1 - y0), height / 2)
    b, t, k = y - height / 2, y + height / 2, 0.5523  # k: cubic Bezier quarter circle
    verts = [(0, b), (width - rx, b),
             (width - rx + k * rx, b), (width, b + ry - k * ry), (width, b + ry),
             (width, t - ry),
             (width, t - ry + k * ry), (width - rx + k * rx, t), (width - rx, t),
             (0, t), (0, b)]
    P = MplPath
    codes = [P.MOVETO, P.LINETO, P.CURVE4, P.CURVE4, P.CURVE4, P.LINETO,
             P.CURVE4, P.CURVE4, P.CURVE4, P.LINETO, P.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, linewidth=0, zorder=3))


def style_axis(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", colors=t["muted"], labelsize=9.5, length=0, pad=6)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def draw(rows: list[dict], theme: str, out: Path) -> None:
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": t["ink"],
    })
    fig = plt.figure(figsize=(9.0, 3.9), dpi=200, facecolor=t["surface"])
    left, right, bottom, top, gap = 0.205, 0.975, 0.215, 0.725, 0.07
    width = (right - left - gap) / 2
    ax1 = fig.add_axes((left, bottom, width, top - bottom))
    ax2 = fig.add_axes((left + width + gap, bottom, width, top - bottom))
    ys = [1.0, 0.0]  # GPT-2 on top
    for ax in (ax1, ax2):
        ax.set_ylim(-0.55, 1.55)

    # Panel 1: dumbbells, unedited -> edited.
    ax1.set_xlim(0, 100)
    ax1.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    style_axis(ax1, t)
    for r, y in zip(rows, ys):
        b, a = pct(*r["before"]), pct(*r["after"])
        ax1.plot([b, a], [y, y], color=t["before"], linewidth=2.2, solid_capstyle="round", zorder=2)
        ax1.plot([b], [y], marker="o", markersize=9.5, color=t["before"],
                 markeredgecolor=t["surface"], markeredgewidth=1.6, zorder=4, clip_on=False)
        ax1.plot([a], [y], marker="o", markersize=9.5, color=t["after"],
                 markeredgecolor=t["surface"], markeredgewidth=1.6, zorder=4, clip_on=False)
        ax1.annotate(pct_label(b), (b, y), xytext=(0, 9), textcoords="offset points",
                     ha="left", va="bottom", fontsize=10, color=t["ink2"])
        ax1.annotate(pct_label(a), (a, y), xytext=(0, 9), textcoords="offset points",
                     ha="right", va="bottom", fontsize=10.5, color=t["ink"], fontweight="semibold")
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=8, color=t["before"],
                      markeredgecolor=t["surface"], label="Unedited"),
               Line2D([], [], marker="o", linestyle="none", markersize=8, color=t["after"],
                      markeredgecolor=t["surface"], label="Edited")]
    leg = ax1.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False,
                     fontsize=9.5, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    # Panel 2: share of next-token predictions changed, against the limit.
    ax2.set_xlim(0, 3.0)
    ax2.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], ["0%", "0.5%", "1%", "1.5%", "2%", "2.5%", "3%"])
    style_axis(ax2, t)
    for r, y in zip(rows, ys):
        changed = 100.0 * (1.0 - r["agree"])
        rounded_bar(ax2, y, changed, 0.24, t["after"])
        # The value sits inside the bar's end (both bars are long enough), so the limit line never crosses it.
        ax2.annotate(f"{changed:.2f}%", (changed, y), xytext=(-7, 0), textcoords="offset points",
                     ha="right", va="center", fontsize=10.5, color="#ffffff", fontweight="semibold", zorder=5)
        ax2.annotate("passes" if changed <= LIMIT else "fails", (3.0, y), ha="right", va="center",
                     fontsize=10, color=t["ink2"])
    ax2.axvline(LIMIT, ymin=0.0, ymax=0.86, color=t["ink2"], linewidth=1.2, linestyle=(0, (3, 2.5)), zorder=4)
    ax2.annotate("Limit 1.5%", (LIMIT, 1.36), xytext=(-5, 0), textcoords="offset points",
                 ha="right", va="center", fontsize=9.5, color=t["ink2"])

    # Panel titles and row labels.
    title_y = top + 0.175
    for ax, title, sub in ((ax1, "3-digit addition on fresh problems", "Share answered exactly right"),
                           (ax2, "Side effects on ordinary text",
                            "Next-token predictions changed on fresh prose")):
        x = ax.get_position().x0
        fig.text(x, title_y, title, fontsize=12.5, fontweight="semibold", color=t["ink"], ha="left", va="bottom")
        fig.text(x, title_y - 0.012, sub, fontsize=9.5, color=t["ink2"], ha="left", va="top")
    inv = fig.transFigure.inverted()
    for r, y in zip(rows, ys):
        yf = inv.transform(ax1.transData.transform((0, y)))[1]
        fig.text(0.025, yf + 0.012, r["model"], fontsize=11, fontweight="semibold", color=t["ink"],
                 ha="left", va="bottom")
        fig.text(0.025, yf + 0.004, r["run"], fontsize=9.5, color=t["ink2"], ha="left", va="top")

    fig.text(0.025, 0.035,
             "Independent re-test with fresh problems (20,000 for GPT-2, 25,000 for Gemma) and fresh prose "
             "(1,536 windows of 64 tokens).\nGPT-2's unedited score is run 2's own count on 2,000 problems. "
             "Drawn by figures/make_results_figure.py from the result files in this repository.",
             fontsize=8.5, color=t["muted"], ha="left", va="bottom", linespacing=1.5)

    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=HERE.parent, help="repository root (default: the parent of figures/)")
    ap.add_argument("--out", type=Path, default=HERE, help="output folder (default: figures/)")
    args = ap.parse_args()
    rows = read_results(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = args.out / f"results-{theme}.png"
        draw(rows, theme, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
