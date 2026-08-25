#!/usr/bin/env bash
# Reproduce the design report's numbers and figures from the report scripts.
# Run from the repository root.  Each report_*.py writes its LaTeX macro
# file and figures into ../PSXM_design_report/, the sibling directory that
# holds the report source (see README).
set -euo pipefail

python report_scan.py          # shield-radius scan      -> scan_1000A.npz
python report_robust.py        # convergence / robustness -> robust_scan.npz
python report_background.py    # field quality vs radius  -> results_background.tex
python report_figures.py       # all report figures       -> results_scan.tex, table_scan.tex

echo "Report numbers and figures regenerated."
