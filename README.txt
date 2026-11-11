Reproducibility Materials
Parity in Minimum-Rank Robust Positive Semidefinite Augmentation
Authors: Yongye Huang and Zhining Wang
Corresponding author: mathwangzhining@163.com

SCOPE AND PROVENANCE

This archive accompanies Mathematics_submission_v2. The computations
use the deterministic mathematical families defined in the article; they are
not empirical sensor measurements. The strengthened odd-dimensional optimizer
classification and the complex-field comparison are theoretical results.
The numerical checks use the archived snapshot described below. The v2 revision
consolidates the article into six main sections and Appendices A-C; it does not
claim newly generated numerical results.

The frozen inputs and results are the 21 files in data/, unchanged by SHA-256
comparison with Mathematics_submission (v1). Keep this snapshot separate from
new runs. The original numerical snapshot is linked in the article at:
https://github.com/hiiiy-pm/minimum-row-robust-psd
The local submission archive contains the scripts supplied with this version;
the original repository should not be assumed to contain every later revision.

The snapshot includes 160 construction checks, 10 SDP/scalar comparisons,
72 direct solver runs, 249 stability path points, 81 graph thresholds with
324 rational edge certificates, and eight sparse construction benchmarks.
The six stable figures have PDF, SVG and 600 dpi PNG exports in figures/
(18 files in total), also supplied in Figures.zip with the submission package.

RECORDED ENVIRONMENT AND SETUP

The frozen numerical environment records Python 3.12.14 on 64-bit Windows,
NumPy 2.3.5, SciPy 1.18.1, SymPy 1.14.0, CVXPY 1.9.3 and Clarabel 0.11.1.
Read data/environment.json and data/construction_environment.json for the
recorded platform, seeds, solver versions, timing details and thread settings.
The pinned installation files are requirements.txt and
experiments/requirements-solver.txt. A version record describes the frozen
run; installing it does not by itself establish that a new run reproduces it.

From the root of this archive, create an isolated environment. On Windows:

  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt -r experiments/requirements-solver.txt

Use that environment's Python for the commands below, for example by replacing
"python" with ".\.venv\Scripts\python.exe". On other platforms use the matching
virtual-environment interpreter. Install the complete requirements when running
all experiments; the moment SDP stage uses CVXPY and Clarabel.

REDRAW THE FROZEN FIGURES

  python scripts/make_figures.py

This reads data/ and writes PDF, SVG and 600 dpi PNG exports to
generated/figures/. It leaves the supplied data/ and figures/ unchanged.
To use a fresh explicit output directory:

  python scripts/make_figures.py --data-dir data --output-dir generated/frozen_redraw/figures

The optional --manuscript-dir argument sends PDF exports to a separate build
directory; it is not needed for ordinary regeneration. Select a fresh build
directory when using it. PDF metadata, rendering libraries and fonts may
prevent byte-identical image output even when the plotted values agree.

GENERATE NEW RESULTS AND FIGURES

Use a new output location for each independent run, for example:

  python experiments/run_experiments.py --output-dir generated/rerun_01/data
  python experiments/construction_benchmark.py --output-dir generated/rerun_01/data
  python scripts/make_figures.py --data-dir generated/rerun_01/data --output-dir generated/rerun_01/figures

Both experimental scripts default to generated/data/, and plotting defaults
to generated/figures/. Reusing an output location can replace earlier generated
files, so choose a new run name to retain those results. The scripts reject the
frozen data/ as the experiment output; plotting rejects frozen figures/ as its
normal output. Do not replace the snapshot with a rerun or mix files from
different runs when evaluating the reported counts and tolerances.

The main experiment uses seed 20261030 and generates closed-form checks,
SDP/scalar comparisons, direct nonconvex runs, stability paths, graph-edge
certificates, analytical energy and threshold grids, and four representative
Gram matrices. Check generated/rerun_01/data/summary.json after completion.
The script raises an error if its recorded validation conditions fail.
Inspect the result and environment records before treating a rerun as verified.

Regenerated rotated Gram matrices use a fixed-seed QR construction and can
differ entrywise from the frozen representatives. They illustrate the same
optimal structures; exact equality of the four frozen representative matrices
is not a regeneration guarantee. Nonlinear solver results and timing values
can change with library versions, numerical libraries and hardware. The direct
optimization plot computes repaired/unrepaired counts from its input table;
the frozen success counts are not hard-coded into that plotting step.

DATA, STABLE FIGURE IDENTIFIERS AND PUBLICATION NUMBERING

  Stable identifier          Source data                       Location in paper
  fig01_parity               epsilon_closed_form.csv,           Figure 1
                             epsilon_spectra.csv
  fig02_energy_frontier      figure_energy_frontier.csv         Figure 2
  fig03_optimizer_structure  figure_optimizer_H_1.csv            Figure C1
                             through figure_optimizer_H_4.csv
  fig04_direct_optimization  nonconvex_runs.csv                  Figure C2
  fig05_stability            stability_paths.csv,                Figure 3
                             stability_fits.csv
  fig06_graph_box            figure_graph_curves.csv,            Figure 4
                             k4_scan.csv

Other data files:
  epsilon_sdp_scalar.csv      Scalar optimum and moment SDP comparison.
  nonconvex_factors.npz       Initial/final factors for direct solver runs.
  nonconvex_summary.csv       Direct solver summaries by parameter pair.
  k4_polynomial_certificate.csv
                             Rational edge-polynomial certificates.
  k4_common_direction_certificate.csv
                             Exact common-direction/monotonicity certificate.
  construction_benchmark.csv Sparse storage, checks and timing results.
  environment.json, construction_environment.json
                             Frozen computational environment records.
  summary.json               Frozen experiment summaries and validations.

The sparse benchmark uses seed 20261101. It measures assembly and vector
application after the two-dimensional solutions are available. Its records
include seven warmed timing batches, medians and quartiles, and blockwise
energy, feasibility and spectrum checks; no large dense matrix is allocated.
The graph-threshold formulas are evaluated on the stated parameter intervals.
Display values at open endpoints are limits. Exact threshold formulas and
proofs are in Appendix B of manuscript.pdf; the raw exact certificates remain
in this archive. Appendix A discusses mode dimension and the scope of the
parity formula. Appendix C contains the auxiliary numerical Figures C1/C2.
Equations, propositions and figures in the appendices use the letter of their
own appendix as their numbering prefix.

The full article, including all three appendices, compiles from
01_Manuscript/main.tex (main.tex at the root of Manuscript_Source.zip).
The submission directory supplies two PDF documents: manuscript.pdf and
cover_letter.pdf. The archive and figure package are Reproducibility.zip and
Figures.zip. Document compilation commands and the author pre-submission
checks are in the parent README.md. These scripts do not submit to the journal.
