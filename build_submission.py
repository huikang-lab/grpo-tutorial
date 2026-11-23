"""
Render the Assignment 3 report as a LaTeX PDF and package it with starter.py into one ZIP file.

    uv run --with matplotlib python build_submission.py --checkpoint-url https://www.kaggle.com/models/...

Every number in the report is read from the runs: output/zero_shot/L<len>.json for the zero-shot baselines,
output/tb/<run>/<ts>/eval_metrics.jsonl for the training curves, and output/rollouts/<run>/<ts>/eval/step_NNNN.jsonl for
the outcome table. The conceptual answers are ANSWERS.md, the analysis is REVIEW.md and the extra-credit experiments are
EXTRA.md. The PDF is compiled with tectonic (brew install tectonic),
as in Assignments 1 and 2. Writes REPORT.tex, REPORT.pdf, the figures and SUBMISSION.zip (starter.py, the Problem 2
variants and the PDF).
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
# Problem 2's runs, all at L512, each beside the main run it changes one thing about.
EXTRA_RUNS: Dict[str, str] = {
    "dr_grpo_L512": "Dr.GRPO (main run)",
    "grpo_L512": "GRPO (main run)",
    "dr_grpo_L512_posadv": "1. Dr.GRPO, no negative advantages",
    "grpo_no_len_norm_L512_group_lenpen": "2. GRPO advantages, length penalty",
}
METHOD: Dict[str, str] = {"grpo": "GRPO", "dr_grpo": "Dr.GRPO"}
# Categorical slots 1 and 2 of the dataviz reference palette, validated for colour-vision separation on white. The
# method keeps its colour in both panels; the zero-shot baselines are grey and dashed.
COLOUR: Dict[str, str] = {"grpo": "#2a78d6", "dr_grpo": "#eb6834"}
REFERENCE: str = "#87919c"
ZIP_FOLDER: str = "GStar-Assignment-3-GRPO"
VIEWER_URL: str = "https://huikang-lab.github.io/grpo-tutorial/"
# The appendix's screenshots of the viewer (report_shots/, taken from ./serve.sh with headless Chrome at 1400 px wide),
# each beside the live page it shows.
VIEWER_SHOTS: List[tuple] = [
    ("1-runs", "Runs", "every run with its settings and final eval; a run opens its steps", "viewer/runs.html"),
    ("2-metrics", "Metrics", "the training and eval curves of every run, filterable by length and run", "viewer/metrics.html"),
    ("3-steps", "Steps", "each training step's reward, accuracy, zero-advantage groups, loss and grad norm",
     "viewer/steps.html?run=grpo_L512"),
    ("4-groups", "Groups", "each prompt's eight attempts as reward cells; a zero-advantage group is struck through",
     "viewer/groups.html?run=grpo_L512&step=3"),
    ("5-attempts", "Attempts", "every rollout with its answer, reward and advantage",
     "viewer/attempts.html?run=grpo_L512&step=3&group=0"),
    ("6-attempt", "One attempt", "the prompt and completion, each token shaded by how unlikely the model found it",
     "viewer/attempts.html?run=grpo_L512&step=3&group=0&rollout=1"),
]

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


# The reductions' names, bare or in `backticks`, read as code that may break after an underscore.
CODE: Dict[str, str] = {"masked_mean_drgrpo": r"\texttt{masked\_mean\_\allowbreak drgrpo}",
                        "masked_mean": r"\texttt{masked\_mean}"}


def escape(text: str) -> str:
    stash: List[str] = []
    text = re.sub(r"`([^`]+)`", r"\1", text)
    def keep(latex: str) -> str:
        stash.append(latex)
        return f"\x00{len(stash) - 1}\x00"

    for plain, latex in CODE.items():
        text = re.sub(rf"\b{plain}\b", lambda _: keep(latex), text)
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
        "answered": sum(r["equation"] is not None for r in rows),
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
    extra = {}
    for run in EXTRA_RUNS:
        curve = jsonl(latest(HERE / "output" / "tb" / run) / "eval_metrics.jsonl")
        extra[run] = {"curve": curve, "best": max(curve, key=lambda r: r["accuracy"]),
                      "final": summarize(eval_rows(run, curve[-1]["step"]))}
    return {"zero": zero, "runs": runs, "extra": extra}


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


# Problem 2's figures, one per experiment: its run in colour, the main run it changes in grey. EXTRA.md places each
# with a "[figure extraN.pdf]" line.
EXTRA_PANELS: List[Dict[str, Any]] = [
    {"runs": {"dr_grpo_L512": ("Dr.GRPO", REFERENCE), "dr_grpo_L512_posadv": ("no negative adv.", COLOUR["grpo"])}},
    {"runs": {"grpo_L512": ("GRPO", REFERENCE),
              "grpo_no_len_norm_L512_group_lenpen": ("length penalty", COLOUR["grpo"])}},
]


def plot_extra(data: Dict[str, Any], panel: Dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for ax, key, label, limits, gap in ((axes[0], "accuracy", "eval accuracy (%)", (10, 70), 4.5),
                                        (axes[1], "avg_output_tokens", "mean eval answer tokens", (200, 520), 22)):
        ends = []
        for run, (name, colour) in panel["runs"].items():
            curve = data["extra"][run]["curve"]
            values = [c[key] for c in curve]
            main = colour == REFERENCE
            ax.plot([c["step"] for c in curve], values, color=colour, linewidth=1.4 if main else 2,
                    marker=None if main else "o", markersize=3.5, zorder=2 if main else 3)
            ends.append([values[-1], name, colour])
        # Labels at the line ends, nudged apart so close finishes stay readable.
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
        for value, name, colour in ends:
            ax.annotate(name, (82, value), color="#52606d" if colour == REFERENCE else colour, fontsize=7.5,
                        va="center", annotation_clip=False)
        ax.set_xlim(0, 80)
        ax.set_ylim(*limits)
        ax.set_ylabel(label)
        ax.set_xlabel("GRPO step")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    fig.tight_layout(w_pad=6)
    fig.savefig(path)
    plt.close(fig)


def answers_latex() -> List[str]:
    """ANSWERS.md: a numbered question (or part, 3.1.) opens each answer, bullets stay bullets, every other line is a
    paragraph."""
    parts: List[str] = []
    bullets = False
    for line in (HERE / "ANSWERS.md").read_text().splitlines():
        if not line.strip():
            continue
        is_bullet = line.lstrip().startswith("- ")
        if bullets and not is_bullet:
            parts.append(r"\end{itemize}")
            bullets = False
        # "3." opens a question and "3.1." one of its parts; each quotes the handout's wording.
        question = re.match(r"^(\d+(?:\.\d+)?)\. (.*)", line)
        if question:
            parts.append(r"\Needspace{4\baselineskip}\textbf{Question " + question[1] + r"}\begin{quote}"
                         + escape(question[2]) + r"\end{quote}")
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


def notes_latex(name: str, quote_first: bool) -> List[str]:
    """REVIEW.md or EXTRA.md: a line ending without punctuation heads the bullets under it, "[figure x.pdf]" places a
    figure, every other line is a paragraph, and REVIEW.md's first paragraph quotes the handout's prompt."""
    paragraphs = [block.splitlines() for block in re.split(r"\n\s*\n", (HERE / name).read_text().strip())]
    parts: List[str] = []
    if quote_first:
        parts.append(r"\begin{quote}" + " ".join(escape(line.strip()) for line in paragraphs.pop(0)) + r"\end{quote}")
    for block in paragraphs:
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
            elif re.fullmatch(r"\[figure (\S+)\]", line.strip()):
                # A heading right above a figure moves to the next page with it rather than stay behind alone.
                if parts and parts[-1].startswith(r"\Needspace{3\baselineskip}\textbf"):
                    parts[-1] = parts[-1].replace(r"\Needspace{3\baselineskip}", r"\Needspace{18\baselineskip}", 1)
                parts.append(r"\begin{center}\includegraphics[width=\linewidth]{" + line.strip()[8:-1] + r"}\end{center}")
            elif not line.rstrip().endswith((".", ":", "?", "!")):
                parts.append(r"\Needspace{3\baselineskip}\textbf{" + escape(line.strip()) + r"}\par")
            else:
                parts.append(escape(line.strip()) + r"\par")
        if bullets:
            parts.append(r"\end{itemize}")
    return parts


def part(title: str, subtitle: str) -> str:
    """A part title over the sections that follow it: the assignment's two problems."""
    return (r"{\LARGE\bfseries " + title + r"\par}\vspace{2pt}{\large " + subtitle + r"\par}"
            r"\vspace{4pt}\hrule\vspace{4pt}")


def report(data: Dict[str, Any], checkpoint_url: str | None) -> str:
    zero, runs = data["zero"], data["runs"]
    date = datetime.date.today().strftime("%-d %B %Y")
    pct = lambda x: f"{x:.1f}\\%"
    out: List[str] = [PREAMBLE, ""]
    out.append(r"\begin{center}{\LARGE GRPO for Countdown\par}\vspace{8pt}" + "\n\n"
               r"Hui-Kang Tong\\[3pt]GStar Bootcamp 2026 | Assignment 3\\[3pt]" + date + r"\end{center}\vspace{6pt}")
    link = (r"\url{" + checkpoint_url + "}") if checkpoint_url else r"\textcolor{red}{\textbf{[link to be added]}}"
    out.append(r"\textbf{Model checkpoints:} " + link + r"\par")
    out.append(r"\textbf{Rollout viewer:} \url{" + VIEWER_URL + r"}\par")
    out.append(r"\vspace{10pt}" + part("Problem 1", "GRPO and Dr.GRPO for Countdown"))

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
    out.append(r"These match the handout's expected 0.1--0.2\%, 15\% and 38\%.\par")

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
               r"at L1024.}\par")

    out.append(r"\Needspace{12\baselineskip}\section*{Analysis of successful and failed cases}")
    out.extend(notes_latex("REVIEW.md", quote_first=True))
    out.append(r"\Needspace{9\baselineskip}The final eval answers of each trained model (1024 each), by outcome. "
               r"\emph{Cut off}: the budget ran out before \texttt{<answer>}.\par")
    out.append(r"\begin{center}\begin{tabular}{lrrrr}\toprule "
               r"Run & Correct & Cut off & Cut off, 4 numbers & Answered wrong\\\midrule")
    for run in RUNS:
        f = runs[run]["final"]
        out.append(f"{METHOD[runs[run]['config']['loss_type']]} L{runs[run]['config']['max_tokens']} & "
                   f"{f['n'] - f['truncated'] - f['wrong']} & {f['truncated']} & {f['truncated4']} & {f['wrong']}\\\\")
    out.append(r"\bottomrule\end{tabular}\end{center}")

    out.append(r"\newpage" + part("Problem 2 (extra credit)", "Two short experiments"))
    out.append(r"\begin{center}\small\begin{tabular}{lrrrrr}\toprule "
               r"Run (L512) & Final & Best (step) & Cut off & Answers correct & Tokens (correct)\\\midrule")
    for run, label in EXTRA_RUNS.items():
        r = data["extra"][run]
        f = r["final"]
        correct = f["n"] - f["truncated"] - f["wrong"]
        out.append(f"{escape(label)} & \\textbf{{{pct(f['accuracy'])}}} & {pct(r['best']['accuracy'])} ({r['best']['step']}) & "
                   f"{f['truncated']} & {pct(100 * correct / f['answered'])} & "
                   f"{f['avg_tokens']:.0f} ({f['avg_tokens_correct']:.0f})\\\\")
        if run == "grpo_L512":
            out.append(r"\midrule")
    out.append(r"\bottomrule\end{tabular}\end{center}")
    out.append(r"\vspace{-6pt}{\small Final eval answers at step 80. \emph{Answers correct}: the share of answers in "
               r"\texttt{<answer>} tags that were correct.}\par")
    out.extend(notes_latex("EXTRA.md", quote_first=False))

    out.append(r"\newpage" + part("Appendix", "The rollout viewer"))
    out.append(r"Every training rollout of the runs above can be browsed at \url{" + VIEWER_URL + r"}.\par")
    for shot, title, what, page in VIEWER_SHOTS:
        out.append(r"\Needspace{14\baselineskip}\textbf{" + title + r"}: " + escape(what) + r".\\"
                   r"{\small\url{" + VIEWER_URL + page + r"}}\par\vspace{2pt}"
                   r"\includegraphics[width=\linewidth]{report_shots/" + shot + r".png}\par\vspace{8pt}")
    out.append(r"\end{document}")
    return "\n\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-url", default=None, help="where the four trained checkpoints are hosted")
    args = parser.parse_args()
    data = load()
    plot(data, HERE / "accuracy.pdf")
    for i, panel in enumerate(EXTRA_PANELS, 1):
        plot_extra(data, panel, HERE / f"extra{i}.pdf")
    (HERE / "REPORT.tex").write_text(report(data, args.checkpoint_url))
    subprocess.run(["tectonic", "--keep-logs", "REPORT.tex"], cwd=HERE, check=True)
    with zipfile.ZipFile(HERE / "SUBMISSION.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        # The two Problem 2 variants ride along, since the report names them.
        for name in ("starter.py", "starter_posadv.py", "starter_lenpen.py", "REPORT.pdf"):
            archive.write(HERE / name, f"{ZIP_FOLDER}/{name}")
    print(f"wrote REPORT.pdf and SUBMISSION.zip{'' if args.checkpoint_url else ' (checkpoint link still missing)'}")


if __name__ == "__main__":
    main()
