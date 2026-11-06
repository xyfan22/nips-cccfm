"""Reproduce the configuration sweep, with explicit rank and hidden width."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANKS = (32, 64, 128, 256, 512)
WIDTHS = (128, 256, 512, 1024, 2048, 1024)


def read_means(path):
    with path.open(newline="") as handle:
        values = {row["metric"]: float(row["mean"]) for row in csv.DictReader(handle)}
    for name in ("w2_sinkhorn", "moment"):
        if not math.isfinite(values[name]):
            raise ValueError(f"Non-finite {name} in {path}")
    return values


def load_rows(root):
    rows = []
    for holdout in ("2line", "3line"):
        anchor_path = root / (
            f"results/baselines/linear_ahlmann_eltze_{holdout}"
            "_shifted_control_summary_uniform_n.csv"
        )
        anchor = read_means(anchor_path)
        for index, rank in enumerate((*RANKS, "full")):
            stem = (
                f"r31_mp_concat_pca_{holdout}"
                if rank == "full"
                else f"r3lr_r{rank}_pca_{holdout}_linear"
            )
            source_path = root / f"results/cccfm/{stem}_uniform_n_summary.csv"
            values = read_means(source_path)
            row = {
                "holdout": holdout,
                "rank": rank,
                "hidden_width": WIDTHS[index],
                "source_csv": str(source_path.relative_to(root)),
                "anchor_csv": str(anchor_path.relative_to(root)),
            }
            for name in ("w2_sinkhorn", "moment"):
                # The submitted figure used four-decimal inputs; retain its data.
                value = round(values[name], 4)
                reference = round(anchor[name], 4)
                if reference <= 0:
                    raise ValueError(f"Invalid anchor {name}: {reference}")
                row[name] = value
                row[f"anchor_{name}"] = reference
                row[f"{name}_change_pct"] = 100 * (value / reference - 1)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("rank_curve.pdf")
    )
    args = parser.parse_args()
    rows = load_rows(args.repo_root.resolve())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )
    fig, left = plt.subplots(figsize=(5.4, 3.4))
    right = left.twinx()
    blue, red = "#1f77b4", "#d62728"
    handles = []
    for metric, axis, color, marker, label in (
        ("w2_sinkhorn", left, blue, "o", r"$W_2$"),
        ("moment", right, red, "s", "Moment"),
    ):
        for holdout, linestyle, face in (
            ("2line", "-", color),
            ("3line", "--", "white"),
        ):
            values = [row for row in rows if row["holdout"] == holdout]
            (line,) = axis.plot(
                range(6),
                [row[f"{metric}_change_pct"] for row in values],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
                markerfacecolor=face,
                markeredgecolor=color,
                linewidth=1.4,
                label=f"{label} ({holdout.replace('line', '-line')})",
            )
            handles.append(line)
    left.set_xticks(
        range(6), [f"{rank}\n{width}" for rank, width in zip((*RANKS, "full"), WIDTHS)]
    )
    left.set_xlabel(r"Configuration: rank $r$ / hidden width $H$")
    left.set_ylabel(r"$W_2$ change vs. Linear-shifted anchor (%)", color=blue)
    right.set_ylabel("Moment change vs. same anchor (%)", color=red)
    left.set_ylim(-22, 42)
    right.set_ylim(-2, 45)
    left.set_xlim(-0.25, 5.25)
    left.axhline(0, color="0.6", linewidth=0.7, linestyle=":", zorder=0)
    left.tick_params(axis="y", colors=blue)
    right.tick_params(axis="y", colors=red)
    for axis in (left, right):
        axis.spines["top"].set_visible(False)
    left.legend(handles=handles, loc="upper left", ncol=2, frameon=False)
    left.text(0.02, 0.27, r"$r\leq128$: $W_2$ near anchor", transform=left.transAxes, fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.85, bottom=0.24, top=0.96)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        bbox_inches="tight",
        metadata={"Title": "Residual-flow rank and hidden-width configurations", "Author": ""},
    )
    plt.close(fig)
    data_path = args.output.with_suffix(".csv")
    with data_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output} and {data_path} ({len(rows)} configurations)")


if __name__ == "__main__":
    main()
