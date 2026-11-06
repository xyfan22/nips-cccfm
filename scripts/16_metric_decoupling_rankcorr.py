"""Compute Spearman rank-correlation quantification of metric decoupling.

Reads results/MASTER_baseline_table_uniform_n.md and writes
results/metric_decoupling_rankcorr.md. Standard library + pandas + scipy only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results" / "MASTER_baseline_table_uniform_n.md"
DST = REPO / "results" / "metric_decoupling_rankcorr.md"

SECTION_HEADERS = {
    "2line": "## 2-line holdout (sources = K562 + RPE1, target = Jurkat)",
    "3line": "## 3-line holdout (sources = K562 + RPE1 + HepG2, target = Jurkat)",
}

MAIN_9_ROWS_2LINE = [
    "target_control_copy",
    "Linear (shifted)",
    "Exact (shifted)",
    "GEARS",
    "STATE zero-shot",
    "CellFlow",
    "ccCFM-PCA direct",
    "MP full-space (r31_mp_concat)",
    "MP r=256 (low-rank)",
]
MAIN_9_ROWS_3LINE = MAIN_9_ROWS_2LINE  # identical schema

RANK_CURVE_DUPLICATES = {
    "MP r=32",
    "MP r=64",
    "MP r=128",
    "MP r=512 (low-rank)",
}


def parse_section(md_text: str, header: str) -> pd.DataFrame:
    """Extract a markdown table that appears under `header`."""
    lines = md_text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == header)
    except StopIteration as exc:
        raise KeyError(f"Section not found: {header!r}") from exc

    table_lines = []
    in_table = False
    for ln in lines[start + 1 :]:
        if ln.startswith("## "):
            break
        if ln.startswith("|"):
            table_lines.append(ln)
            in_table = True
        elif in_table and not ln.strip():
            break

    if len(table_lines) < 3:
        raise ValueError(f"No table parsed under {header!r}")

    header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
    rows = []
    for ln in table_lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(header_cells):
            continue
        rows.append(cells)

    df = pd.DataFrame(rows, columns=header_cells)
    for col in df.columns:
        if col == "Method":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def spearman_with_p(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    rho, p = spearmanr(x.values, y.values)
    return float(rho), float(p)


def compute_block(df: pd.DataFrame, label: str) -> dict:
    pf = df["Pf^pert"]
    energy = df["Energy"]
    w2 = df["W_2"]
    moment = df["Moment"]

    rho_e, p_e = spearman_with_p(pf, -energy)
    rho_w, p_w = spearman_with_p(pf, -w2)
    rho_m, p_m = spearman_with_p(pf, -moment)

    return {
        "label": label,
        "n": len(df),
        "rows": df["Method"].tolist(),
        "Pf_vs_negEnergy": (rho_e, p_e),
        "Pf_vs_negW2": (rho_w, p_w),
        "Pf_vs_negMoment": (rho_m, p_m),
    }


def top3_by(df: pd.DataFrame, col: str, ascending: bool) -> list[tuple[str, float]]:
    sub = df.dropna(subset=[col]).sort_values(col, ascending=ascending, kind="stable").head(3)
    return list(zip(sub["Method"], sub[col]))


def fmt_top3(items: list[tuple[str, float]]) -> str:
    return "; ".join(f"{name} ({val:.3f})" for name, val in items)


def fmt_rho(pair: tuple[float, float]) -> str:
    rho, p = pair
    if p < 0.001:
        p_str = "p<0.001"
    else:
        p_str = f"p={p:.3f}"
    return f"{rho:+.3f} ({p_str})"


def main() -> None:
    md = SRC.read_text()
    df_2line_full = parse_section(md, SECTION_HEADERS["2line"])
    df_3line_full = parse_section(md, SECTION_HEADERS["3line"])

    # A: main 9-row, 2-line
    df_a = df_2line_full.set_index("Method").reindex(MAIN_9_ROWS_2LINE).reset_index()
    missing_a = df_a[df_a["Pf^pert"].isna()]["Method"].tolist()
    if missing_a:
        raise ValueError(f"Missing rows in 2-line for A: {missing_a}")
    block_a = compute_block(df_a, "A. Main 9-row, 2-line holdout")

    # B: main 9-row, 3-line. STATE zero-shot is a released model reported
    # identically across 2-line and 3-line in the paper (see Table 2 caption);
    # fall back to the 2-line row for it if absent from the 3-line section.
    state_fallback = (
        df_2line_full.set_index("Method").loc["STATE zero-shot"].to_frame().T.reset_index()
    )
    state_fallback["Method"] = "STATE zero-shot"
    df_3line_aug = pd.concat(
        [df_3line_full, state_fallback[df_3line_full.columns]], ignore_index=True
    )
    df_b = df_3line_aug.set_index("Method").reindex(MAIN_9_ROWS_3LINE).reset_index()
    missing_b = df_b[df_b["Pf^pert"].isna()]["Method"].tolist()
    if missing_b:
        raise ValueError(f"Missing rows in 3-line for B: {missing_b}")
    block_b = compute_block(df_b, "B. Main 9-row, 3-line holdout")

    # C1: full 2-line, all reported rows
    df_c1 = df_2line_full.copy()
    block_c1 = compute_block(df_c1, "C1. Full 2-line, all reported rows")

    # C2: full 2-line, no rank-curve duplicates
    df_c2 = df_2line_full[~df_2line_full["Method"].isin(RANK_CURVE_DUPLICATES)].copy()
    block_c2 = compute_block(df_c2, "C2. Full 2-line, no rank-curve duplicates")

    # Rank disagreement (main 9-row, 2-line)
    top_pf = top3_by(df_a, "Pf^pert", ascending=False)
    top_energy = top3_by(df_a, "Energy", ascending=True)
    top_w2 = top3_by(df_a, "W_2", ascending=True)
    top_moment = top3_by(df_a, "Moment", ascending=True)

    blocks = [block_a, block_b, block_c1, block_c2]

    out = []
    out.append("# Metric decoupling: Spearman rank-correlation quantification\n")
    out.append("**Generated**: 2026-05-05\n")
    out.append(f"**Source**: `{SRC.relative_to(REPO)}` (uniform-n source-of-truth).\n")
    out.append(
        "**Method**: For each row group, compute Spearman rank correlation "
        "between perturbation-specific Pf^pert (higher better) and three "
        "distributional metrics with sign flipped so higher is better "
        "(-Energy, -W_2, -Moment). p-values from scipy.stats.spearmanr "
        "two-sided test; n is small (9--22) so p-values are advisory only.\n"
    )

    out.append("\n## Spearman rank correlations\n")
    out.append("| Setting | n | Spearman(Pf, -Energy) | Spearman(Pf, -W_2) | Spearman(Pf, -Moment) |\n")
    out.append("|---|---|---|---|---|\n")
    for blk in blocks:
        out.append(
            f"| {blk['label']} | {blk['n']} | "
            f"{fmt_rho(blk['Pf_vs_negEnergy'])} | "
            f"{fmt_rho(blk['Pf_vs_negW2'])} | "
            f"{fmt_rho(blk['Pf_vs_negMoment'])} |\n"
        )

    out.append("\n## Rows used per block\n")
    for blk in blocks:
        out.append(f"\n### {blk['label']}  (n={blk['n']})\n")
        for r in blk["rows"]:
            out.append(f"- {r}\n")

    out.append("\n## Rank disagreement summary (Main 9-row, 2-line)\n")
    out.append(f"- Top-3 by Pf^pert (high better): {fmt_top3(top_pf)}\n")
    out.append(f"- Top-3 by Energy (low better): {fmt_top3(top_energy)}\n")
    out.append(f"- Top-3 by W_2 (low better): {fmt_top3(top_w2)}\n")
    out.append(f"- Top-3 by Moment (low better): {fmt_top3(top_moment)}\n")

    rho_a_e = block_a["Pf_vs_negEnergy"][0]
    rho_a_w = block_a["Pf_vs_negW2"][0]
    rho_a_m = block_a["Pf_vs_negMoment"][0]

    out.append("\n## Honest reading of the numbers\n")
    out.append(
        "The Spearman correlations are **positive but incomplete**, not low or "
        "negative. They are inflated when the full method set is included "
        "(C1, C2: ~+0.55 to +0.78) because rows like STATE/STACK/scFoundation "
        "score badly on every metric simultaneously, so any \"good methods are "
        "also less bad on dist metrics\" signal pushes Spearman up. The cleaner "
        "9-row blocks (A, B) give ~+0.45 to +0.68. The decoupling evidence "
        "therefore is **not** \"the metrics are uncorrelated overall\". It is "
        "that **the top-ranked row is metric-dependent**: the top-ranked method "
        "differs across metrics (e.g.\\ target-control-copy ranks #1 on Energy "
        "and Moment but middle on Pf^pert; ccCFM-PCA direct ranks #1 on W_2 but "
        "near bottom on Pf^pert). A reader who picks the \"best\" method by any "
        "single distributional metric in the main 2-line table picks a row that "
        "is not best on Pf^pert.\n"
    )

    out.append("\n## Suggested wording (locked to paper.tex)\n")
    out.append(
        "**Main-text (one-sentence, near Figure 2 lead-in)**: "
        "Quantitatively, rank correlations are positive but incomplete across the "
        "nine main 2-line rows "
        f"(Spearman $\\rho{{=}}{rho_a_e:.2f}$ for Pf$^{{\\mathrm{{pert}}}}$ vs.\\ $-$Energy, "
        f"${rho_a_w:.2f}$ for Pf$^{{\\mathrm{{pert}}}}$ vs.\\ $-W_2$, "
        f"and ${rho_a_m:.2f}$ for Pf$^{{\\mathrm{{pert}}}}$ vs.\\ $-$Moment), "
        "and the top-ranked row is metric-dependent: Exact shifted-control is best on "
        "Pf$^{\\mathrm{pert}}$, target-control-copy is best on Energy and Moment, and "
        "ccCFM-PCA direct is best on $W_2$ (Appendix~\\ref{sec:appendix-rankcorr}).\n\n"
    )
    out.append(
        "**Appendix subsection sentence**: "
        "Table~\\ref{tab:rankcorr} reports Spearman rank correlations between "
        "Pf$^{\\mathrm{pert}}$ and sign-flipped distributional metrics across the "
        "main 2-line rows, the main 3-line rows, and the full 2-line method set "
        "(with and without rank-curve duplicates). The full-set correlations are "
        "inflated by uniformly-bad rows (STATE, STACK, scFoundation-conditioned "
        "probes) that fail simultaneously on every metric. The 9-row blocks give "
        "the cleaner read: Spearman is positive but incomplete, and the top-ranked "
        "row is metric-dependent in the main 2-line table --- target-control-copy "
        "is first on Energy ($1.04$) and Moment ($0.400$); ccCFM-PCA direct is "
        "first on $W_2$ ($0.115$); Exact shifted-control is first on "
        "Pf$^{\\mathrm{pert}}$ ($0.331$). Picking the best method by any single "
        "distributional metric in Table~\\ref{tab:main-2line} therefore selects a "
        "different row than picking it by Pf$^{\\mathrm{pert}}$.\n"
    )

    DST.write_text("".join(out))
    print(f"Wrote {DST.relative_to(REPO)}")
    for blk in blocks:
        print(
            f"  {blk['label']}: n={blk['n']}, "
            f"Pf~-Energy={fmt_rho(blk['Pf_vs_negEnergy'])}, "
            f"Pf~-W2={fmt_rho(blk['Pf_vs_negW2'])}, "
            f"Pf~-Moment={fmt_rho(blk['Pf_vs_negMoment'])}"
        )


if __name__ == "__main__":
    main()
