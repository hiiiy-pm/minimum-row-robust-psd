Reproducibility materials
Parity in minimum-rank robust positive semidefinite augmentation

The frozen results are in data/ and the associated PDF, SVG and 600 dpi PNG
figures are in figures/. The computations use the mathematical test families
defined in the manuscript. The archive contains 160 construction checks,
10 SDP/scalar comparisons, 72 direct solver runs, 249 stability path points,
81 graph thresholds with 324 rational edge certificates, and eight sparse
construction benchmarks.

SETUP
Use Python 3.12. Create and activate a virtual environment and install the
recorded requirements from the root of this archive:

  python -m venv .venv
  python -m pip install -r requirements.txt -r experiments/requirements-solver.txt

The recorded versions, platform information and seeds are in
data/environment.json and data/construction_environment.json.

REDRAW THE FROZEN FIGURES

  python scripts/make_figures.py

This reads data/ and writes PDF, SVG and 600 dpi PNG exports to
generated/figures/. The source archive includes the PDF figures used in the
manuscript. Figure file identifiers are stable even when manuscript numbering
differs. Figure 3 illustrates representative Gram matrices and Figure 4 reports
the direct optimization runs. Both are supplied as supplementary figures.

GENERATE NEW RESULTS AND FIGURES

  python experiments/run_experiments.py
  python experiments/construction_benchmark.py
  python scripts/make_figures.py --data-dir generated/data

The first two commands write to generated/data/ by default. Use --output-dir
on the experimental or plotting scripts to select another output directory.
The main experiment uses seed 20261030. It generates the analytical energy
and threshold grids and four representative Gram matrices as well as the
experiment tables. Regenerated rotated Gram matrices use a fixed-seed QR
construction and can differ entrywise from the frozen representatives.
They represent the same optimal structures. Nonlinear solver outcomes and
timings may change with library versions and hardware. The direct optimization
plot counts repaired and unrepaired outcomes from the supplied table.

DATA AND FIGURE MAP
  fig01_parity                 epsilon_closed_form.csv, epsilon_spectra.csv
  fig02_energy_frontier        figure_energy_frontier.csv
  fig03_optimizer_structure   figure_optimizer_H_1.csv to H_4.csv
  fig04_direct_optimization    nonconvex_runs.csv
  fig05_stability              stability_paths.csv, stability_fits.csv
  fig06_graph_box              figure_graph_curves.csv, k4_scan.csv

  epsilon_sdp_scalar.csv       scalar optimum and moment SDP cross-check
  nonconvex_factors.npz        initial and final factors for the direct runs
  nonconvex_summary.csv        direct solver summaries by parameter pair
  k4_polynomial_certificate.csv and k4_common_direction_certificate.csv
                               exact edge and monotonicity certificates
  construction_benchmark.csv   sparse storage, validation and timed batches

The sparse benchmark uses seed 20261101 and measures assembly and vector
application after the two-dimensional solution is available. The supplied
data record seven warmed timing batches, their median and quartiles, and
blockwise checks of energy, feasibility and spectrum. No large dense matrix
is allocated. The formulas for the graph thresholds are evaluated on the
stated parameter intervals. Display values at open endpoints are limits.
