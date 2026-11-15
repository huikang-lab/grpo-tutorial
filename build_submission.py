"""
Render the Assignment 3 report as a LaTeX PDF and package it with starter.py into one ZIP file.

    uv run --with matplotlib python build_submission.py --checkpoint-url https://www.kaggle.com/models/...

Every number in the report is read from the runs: output/zero_shot/L<len>.json for the zero-shot baselines,
output/tb/<run>/<ts>/eval_metrics.jsonl for the training curves, and output/rollouts/<run>/<ts>/eval/step_NNNN.jsonl for
the outcome table. The conceptual answers are ANSWERS.md and the analysis is REVIEW.md. The PDF is compiled with tectonic (brew install tectonic),
as in Assignments 1 and 2. Writes REPORT.tex, REPORT.pdf, accuracy.pdf and SUBMISSION.zip.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, List

HERE: Path = Path(__file__).parent
RUNS: List[str] = ["grpo_L256", "dr_grpo_L256", "grpo_L512", "dr_grpo_L512"]
METHOD: Dict[str, str] = {"grpo": "GRPO", "dr_grpo": "Dr.GRPO"}
# Categorical slots 1 and 2 of the dataviz reference palette, validated for colour-vision separation on white. The
# method keeps its colour in both panels; the zero-shot baselines are grey and dashed.
COLOUR: Dict[str, str] = {"grpo": "#2a78d6", "dr_grpo": "#eb6834"}
REFERENCE: str = "#87919c"
ZIP_FOLDER: str = "GStar-Assignment-3-GRPO"
VIEWER_URL: str = "https://huikang-lab.github.io/grpo-tutorial/"

ESCAPES: Dict[str, str] = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
}
UNICODE: Dict[str, str] = {"π": r"$\pi$", "θ": r"$\theta$", "ε": r"$\varepsilon$", "∇": r"$\nabla$", "−": "-",
                           "·": r"$\cdot$", "≈": r"$\approx$", "×": r"$\times$", "–": "--", "—": "---"}
# Same layout as Assignment 2's report.
PREAMBLE: str = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[hmargin=0.8in,vmargin=0.6in]{geometry}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{needspace}
\usepackage[colorlinks=true,urlcolor=linkcolor]{hyperref}
\definecolor{linkcolor}{HTML}{23475C}
\hypersetup{pdfauthor={Hui-Kang Tong},pdftitle={GRPO for Countdown}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlength{\emergencystretch}{2em}
\titleformat{\section}{\normalfont\large\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}
\setlist[itemize]{leftmargin=1.3em,topsep=2pt,itemsep=2pt,parsep=0pt}
\lstset{basicstyle=\ttfamily\scriptsize,columns=fullflexible,keepspaces=true,breaklines=true,frame=single,rulecolor=\color{gray}}
\pagestyle{plain}
\clubpenalty=10000
\widowpenalty=10000
\begin{document}"""


# ANSWERS.md writes subscripts the way plain text does; these become math before the rest is escaped.
MATH: Dict[str, str] = {"π_θold": r"$\pi_{\theta_\mathrm{old}}$", "π_θ": r"$\pi_\theta$", "ratio_t": r"$\mathrm{ratio}_t$",
                        "A_t": r"$A_t$", "1−ε": r"$1-\varepsilon$", "1+ε": r"$1+\varepsilon$"}


def escape(text: str) -> str:
    stash: List[str] = []
    for plain, latex in MATH.items():
        while plain in text:
            text = text.replace(plain, f"\x00{len(stash)}\x00", 1)
            stash.append(latex)
    out = "".join(ESCAPES.get(char, char) for char in text)
    for char, latex in UNICODE.items():
        out = out.replace(char, latex)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m[1])], out)


def jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest(folder: Path) -> Path:
    return sorted(folder.iterdir(), key=lambda p: int(p.name))[-1]


def classify(row: Dict[str, Any]) -> str:
    if row["reward"] == 1.0:
        return "correct"
    if row["equation"] is None:
        return "truncated" if row["finish_reason"] == "length" else "no answer"
    if row["result"] is None:
        return "invalid"
    used = [int(x) for x in re.findall(r"\d+", row["equation"])]
    if any(used.count(x) > row["numbers"].count(x) for x in used):
        return "wrong numbers"
    return "wrong value"


def eval_rows(run: str, step: int) -> List[Dict[str, Any]]:
    return jsonl(latest(HERE / "output" / "rollouts" / run) / "eval" / f"step_{step:04d}.jsonl")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    kinds = [classify(r) for r in rows]
    by_size = {n: [r for r in rows if len(r["numbers"]) == n] for n in (3, 4)}
    correct = [r for r in rows if r["reward"] == 1.0]
    return {
        "accuracy": 100 * len(correct) / len(rows),
        "acc3": 100 * sum(r["reward"] == 1.0 for r in by_size[3]) / len(by_size[3]),
        "acc4": 100 * sum(r["reward"] == 1.0 for r in by_size[4]) / len(by_size[4]),
        "truncated": kinds.count("truncated"),
        "truncated4": sum(classify(r) == "truncated" for r in by_size[4]),
        "wrong": len(rows) - len(correct) - kinds.count("truncated"),
        "kinds": {k: kinds.count(k) for k in set(kinds)},
        "avg_tokens": sum(r["output_tokens"] for r in rows) / len(rows),
        "avg_tokens_correct": sum(r["output_tokens"] for r in correct) / max(1, len(correct)),
        "subset": sum(len(re.findall(r"\d+", r["equation"])) < len(r["numbers"]) for r in correct),
        "single": sum(bool(re.fullmatch(r"\s*\d+\s*", r["equation"])) for r in correct),
        "with_equals": sum(bool(r["equation"]) and "=" in r["equation"] for r in rows),
        "n": len(rows), "n4": len(by_size[4]),
    }


def flat_groups(run: str, first: int, last: int) -> Dict[str, int]:
    """Zero-advantage groups over training steps first..last, by problem size and whether all eight solved it."""
    base = latest(HERE / "output" / "rollouts" / run) / "train"
    counts: Dict[str, int] = {"groups": 0, "solved3": 0, "failed4": 0, "flat": 0}
    for step in range(first, last + 1):
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for row in jsonl(base / f"step_{step:04d}.jsonl"):
            groups.setdefault(row["group"], []).append(row)
        for rows in groups.values():
            counts["groups"] += 1
            rewards = {r["reward"] for r in rows}
            if len(rewards) > 1:
                continue
            counts["flat"] += 1
            size = len(rows[0]["numbers"])
            counts["solved3"] += size == 3 and rewards == {1.0}
            counts["failed4"] += size == 4 and rewards != {1.0}
    return counts


def load() -> Dict[str, Any]:
    zero = {z["max_tokens"]: z for z in (json.loads(p.read_text()) for p in (HERE / "output" / "zero_shot").glob("L*.json"))}
    runs = {}
    for run in RUNS:
        tb = latest(HERE / "output" / "tb" / run)
        config = json.loads((tb / "config.json").read_text())
        curve = jsonl(tb / "eval_metrics.jsonl")
        train = jsonl(tb / "train_metrics.jsonl")
        best = max(curve, key=lambda r: r["accuracy"])
        runs[run] = {"config": config, "curve": curve, "train": train, "best": best, "start": summarize(eval_rows(run, 0)),
                     "final": summarize(eval_rows(run, curve[-1]["step"])), "final_step": curve[-1]["step"],
                     "first_signal": next((r["step"] for r in train if r["reward_mean"] > 0), None),
                     "sec_per_step": sum(r["t_gen"] + r["t_train"] for r in train) / len(train),
                     "flat": flat_groups(run, 41, 80)}
    return {"zero": zero, "runs": runs}


def plot(data: Dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c5cad1", "axes.labelcolor": "#26313d", "xtick.color": "#52606d",
                         "ytick.color": "#52606d"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, length in zip(axes, (256, 512)):
        for loss in ("grpo", "dr_grpo"):
            curve = data["runs"][f"{loss}_L{length}"]["curve"]
            steps, acc = [r["step"] for r in curve], [r["accuracy"] for r in curve]
            ax.plot(steps, acc, color=COLOUR[loss], linewidth=2, marker="o", markersize=3.5, label=METHOD[loss])
            ax.annotate(f"{METHOD[loss]} {acc[-1]:.1f}%", (steps[-1], acc[-1]), xytext=(4, 6 if loss == "grpo" else -10),
                        textcoords="offset points", color=COLOUR[loss], fontsize=8)
        for ref in (length, 1024):
            value = data["zero"][ref]["accuracy"]
            ax.axhline(value, color=REFERENCE, linestyle="--", linewidth=1)
            ax.annotate(f"zero-shot L{ref} {value:.1f}%", (96, value), xytext=(-2, 3 if value < 10 else -9), textcoords="offset points",
                        color=REFERENCE, fontsize=7.5, ha="right")
        ax.set_title(f"Trained at L{length}", fontsize=10, color="#26313d", loc="left")
        ax.set_xlabel("GRPO step")
        ax.set_xlim(0, 96)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axes[0].set_ylabel("eval accuracy (%)")
    axes[0].set_ylim(0, 70)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def answers_latex() -> List[str]:
    """ANSWERS.md: a numbered question opens each answer, bullets stay bullets, every other line is a paragraph."""
    parts: List[str] = []
    bullets = False
    for line in (HERE / "ANSWERS.md").read_text().splitlines():
        if not line.strip():
            continue
        is_bullet = line.lstrip().startswith("- ")
        if bullets and not is_bullet:
            parts.append(r"\end{itemize}")
            bullets = False
        if re.match(r"^\d+\. ", line):
            parts.append(r"\Needspace{4\baselineskip}\textbf{Question " + line.split(".")[0] + r"}\begin{quote}"
                         + escape(line.split(". ", 1)[1]) + r"\end{quote}")
        elif is_bullet:
            if not bullets:
                parts.append(r"\begin{itemize}")
                bullets = True
            parts.append(r"\item " + escape(line.lstrip()[2:]))
        else:
            parts.append(escape(line.strip()) + r"\par")
    if bullets:
        parts.append(r"\end{itemize}")
    return parts


def review_latex() -> List[str]:
    """REVIEW.md: its first paragraph quotes the handout's prompt, a line ending without punctuation heads the bullets
    under it, and every other line is a paragraph."""
    paragraphs = [block.splitlines() for block in re.split(r"\n\s*\n", (HERE / "REVIEW.md").read_text().strip())]
    parts: List[str] = [r"\begin{quote}" + " ".join(escape(line.strip()) for line in paragraphs[0]) + r"\end{quote}"]
    for block in paragraphs[1:]:
        bullets = False
        for line in block:
            if not line.strip():
                continue
            is_bullet = line.lstrip().startswith("- ")
            if bullets and not is_bullet:
                parts.append(r"\end{itemize}")
                bullets = False
            if is_bullet:
                if not bullets:
                    parts.append(r"\begin{itemize}")
                    bullets = True
                parts.append(r"\item " + escape(line.lstrip()[2:]))
            elif not line.rstrip().endswith((".", ":", "?", "!")):
                parts.append(r"\Needspace{3\baselineskip}\textbf{" + escape(line.strip()) + r"}\par")
            else:
                parts.append(escape(line.strip()) + r"\par")
        if bullets:
            parts.append(r"\end{itemize}")
    return parts


def report(data: Dict[str, Any], checkpoint_url: str | None) -> str:
    zero, runs = data["zero"], data["runs"]
    g256, d256, g512, d512 = (runs[r] for r in RUNS)
    date = datetime.date.today().strftime("%-d %B %Y")
    pct = lambda x: f"{x:.1f}\\%"
    out: List[str] = [PREAMBLE, ""]
    out.append(r"\begin{center}{\LARGE GRPO for Countdown\par}\vspace{8pt}" + "\n\n"
               r"Hui-Kang Tong\\[3pt]GStar Bootcamp 2026 | Assignment 3\\[3pt]" + date + r"\end{center}\vspace{6pt}")
    link = (r"\url{" + checkpoint_url + "}") if checkpoint_url else r"\textcolor{red}{\textbf{[link to be added]}}"
    out.append(r"\textbf{Model checkpoints:} " + link + r"\par")
    out.append(r"\textbf{Rollout viewer:} \url{" + VIEWER_URL + r"} (every training rollout of the runs below)\par")
    out.append(r"Tasks 1--5 are implemented in \texttt{starter.py}. Every run uses the default settings "
               r"(80 GRPO steps of 16 prompts $\times$ 8 samples, learning rate $7\times10^{-6}$, seed 42) on one NVIDIA H200 "
               r"through Modal. Accuracy is measured on the 1024 test problems with one sample per problem at temperature 1, "
               r"so differences of about one point are within sampling noise.\par")

    out.append(r"\Needspace{6\baselineskip}\section*{Conceptual questions}")
    out.extend(answers_latex())

    out.append(r"\Needspace{10\baselineskip}\section*{Zero-shot baselines}")
    out.append(r"\begin{center}\begin{tabular}{lrrrrr}\toprule "
               r"Budget & Accuracy & 3 numbers & 4 numbers & Truncated & Mean tokens\\\midrule")
    for length in (256, 512, 1024):
        z = zero[length]
        out.append(f"L{length} & {pct(z['accuracy'])} & {pct(z['acc_3num'])} & {pct(z['acc_4num'])} & "
                   f"{pct(100 * z['truncated_frac'])} & {z['avg_tokens']:.0f}\\\\")
    out.append(r"\bottomrule\end{tabular}\end{center}")
    out.append(r"These match the handout's expected 0.1--0.2\%, 15\% and 38\%. The budget, not the model's knowledge, "
               r"is what limits it: at L256 almost every answer is cut off while the model is still thinking.\par")

    out.append(r"\Needspace{14\baselineskip}\section*{Main results}")
    out.append(r"\begin{center}\begin{tabular}{llrrrrrr}\toprule "
               r"Method & Budget & Step 0 & Final (step 80) & Best (step) & 3 numbers & 4 numbers & Mean tokens\\\midrule")
    for run in RUNS:
        r = runs[run]
        out.append(f"{METHOD[r['config']['loss_type']]} & L{r['config']['max_tokens']} & {pct(r['start']['accuracy'])} & "
                   f"\\textbf{{{pct(r['final']['accuracy'])}}} & {pct(r['best']['accuracy'])} ({r['best']['step']}) & "
                   f"{pct(r['final']['acc3'])} & {pct(r['final']['acc4'])} & {r['final']['avg_tokens']:.0f}\\\\")
    out.append(r"\midrule Zero-shot & L1024 & \multicolumn{2}{r}{" + pct(zero[1024]["accuracy"]) + r"} & & "
               + f"{pct(zero[1024]['acc_3num'])} & {pct(zero[1024]['acc_4num'])} & {zero[1024]['avg_tokens']:.0f}" + r"\\")
    out.append(r"\bottomrule\end{tabular}\end{center}")
    out.append(r"\begin{center}\includegraphics[width=\linewidth]{accuracy.pdf}\end{center}")
    out.append(r"\vspace{-6pt}{\small Eval accuracy every 10 steps. Dashed: the zero-shot baseline at the same budget and "
               r"at L1024, the target to beat.}\par")
    out.append(
        f"All four trained models beat the L1024 zero-shot score of {pct(zero[1024]['accuracy'])} while generating "
        f"less than half as many tokens per answer. The L512 runs reach it within 10 steps "
        f"(GRPO {pct(g512['curve'][1]['accuracy'])}, Dr.GRPO {pct(d512['curve'][1]['accuracy'])}), "
        f"and the L256 runs between steps 30 and 40. GRPO finished ahead of Dr.GRPO at both budgets "
        f"({pct(g256['final']['accuracy'])} vs {pct(d256['final']['accuracy'])} at L256, "
        f"{pct(g512['final']['accuracy'])} vs {pct(d512['final']['accuracy'])} at L512), "
        f"but these are single runs with one seed, and the handout puts the seed-to-seed spread at about two points at "
        f"L512 and more at L256. Dr.GRPO L256 peaked at {pct(d256['best']['accuracy'])} at step {d256['best']['step']} "
        r"before ending lower, so its gap to GRPO is not established by these runs.\par")

    out.append(r"\Needspace{12\baselineskip}\section*{Analysis of successful and failed cases}")
    out.extend(review_latex())
    out.append(r"\Needspace{9\baselineskip}The final eval answers of each trained model (1024 each), by outcome. "
               r"\emph{Cut off}: the budget ran out before \texttt{<answer>}.\par")
    out.append(r"\begin{center}\begin{tabular}{lrrrr}\toprule "
               r"Run & Correct & Cut off & Cut off, 4 numbers & Answered wrong\\\midrule")
    for run in RUNS:
        f = runs[run]["final"]
        out.append(f"{METHOD[runs[run]['config']['loss_type']]} L{runs[run]['config']['max_tokens']} & "
                   f"{f['n'] - f['truncated'] - f['wrong']} & {f['truncated']} & {f['truncated4']} & {f['wrong']}\\\\")
    out.append(r"\bottomrule\end{tabular}\end{center}")
    out.append(r"\end{document}")
    return "\n\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-url", default=None, help="where the four trained checkpoints are hosted")
    args = parser.parse_args()
    data = load()
    plot(data, HERE / "accuracy.pdf")
    (HERE / "REPORT.tex").write_text(report(data, args.checkpoint_url))
    subprocess.run(["tectonic", "--keep-logs", "REPORT.tex"], cwd=HERE, check=True)
    with zipfile.ZipFile(HERE / "SUBMISSION.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("starter.py", "REPORT.pdf"):
            archive.write(HERE / name, f"{ZIP_FOLDER}/{name}")
    print(f"wrote REPORT.pdf and SUBMISSION.zip{'' if args.checkpoint_url else ' (checkpoint link still missing)'}")


if __name__ == "__main__":
    main()
