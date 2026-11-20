#!/usr/bin/env python3
"""Build llms-full.txt: the whole repository in one file, for uploading to an AI model.

Run from the repository root after changing any file:
    python3 tools/make_llms_full.py

Includes every tracked text file (documents, code, results) in a reading order:
overview and rules first, then experiments (write-up, results, code).
Leaves out images, binary files, the license, and the notes about other projects
and private planning (THREE_PROJECTS.md, COLLAPSE3.md, TODO.md).
"""
import datetime
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "llms-full.txt")
# Left out: binary files, the license, and files about other projects or private planning,
# so the upload is about Last Suit Standing only.
SKIP = {"llms-full.txt", "LICENSE", ".gitignore", "data", "table.png",
        "THREE_PROJECTS.md", "COLLAPSE3.md", "TODO.md"}
TEXT_EXT = (".md", ".py", ".txt")

FIRST = ["README.md", "RULES.md", "VARIANTS.md", "solver.md", "solver.py"]
EXPERIMENT_ORDER = ["scaling", "checkpoint", "matched_steps", "size_shift", "size_shift_followup", "size_shift_mixed", "slot_free", "component_repair", "rule_shift",
                    "rule_shift_exploit", "hidden_gap", "ace_effect", "elo", "straight_rule",
                    "wild_rule", "jacks_rule", "basic_stats"]
LAST = []


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def ordered_files():
    tracked = [f for f in git("ls-files").splitlines()
               if os.path.basename(f) not in SKIP and f.endswith(TEXT_EXT)]
    order = [f for f in FIRST if f in tracked]
    exp = [f for f in tracked if f.startswith("experiments/")]

    def exp_key(f):
        name = os.path.splitext(os.path.basename(f))[0]
        stem = next((e for e in sorted(EXPERIMENT_ORDER, key=len, reverse=True) if name.startswith(e)), name)
        rank = EXPERIMENT_ORDER.index(stem) if stem in EXPERIMENT_ORDER else len(EXPERIMENT_ORDER)
        kind = 0 if f.endswith(".md") else 1 if f.endswith(".txt") else 2  # write-up, results, code
        return (rank, stem, kind, f)

    order += sorted(exp, key=exp_key)
    order += [f for f in LAST if f in tracked]
    order += [f for f in tracked if f not in order]  # anything new goes at the end
    return order


def main():
    files = ordered_files()
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    today = datetime.date.today().isoformat()
    parts = [
        "# Last Suit Standing: the full repository in one file\n",
        f"Generated {today} from commit `{commit}` by `tools/make_llms_full.py`. "
        "Do not edit by hand; rebuild it instead.\n",
        "## Notes for the AI reading this\n",
        "- Last Suit Standing is a small two-player card game, solved exactly, used as a test bed for AI research. "
        "Start with README.md for the game and the main findings.\n"
        "- Every file of the repository follows, each under a `## File:` heading and wrapped in a `~~~~~` fence. "
        "Cite files by their path.\n"
        "- The results files (`*_results.txt`) are the actual outputs of the runs. "
        "The write-ups (`experiments/*.md`) interpret them; where predictions were pre-registered, the write-up says so.\n"
        "- If you are asked to review the work, check claims against the code and results files here. "
        "Do not report results you have not run, and say what you are unsure of.\n",
        "## Contents\n",
        "\n".join(f"- {f}" for f in files) + "\n",
    ]
    for f in files:
        with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
            body = fh.read().rstrip("\n")
        lang = {".md": "markdown", ".py": "python", ".txt": "text"}[os.path.splitext(f)[1]]
        parts.append(f"## File: {f}\n\n~~~~~{lang}\n{body}\n~~~~~\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    words = sum(len(p.split()) for p in parts)
    print(f"wrote llms-full.txt: {len(files)} files, about {words:,} words")


if __name__ == "__main__":
    main()
