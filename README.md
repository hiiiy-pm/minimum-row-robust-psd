# Minimum-row robust PSD augmentation

Code, frozen numerical results, exact certificates, and figures for **Minimum-row robust positive semidefinite augmentation with exact parity costs, optimizer structure, and stability**.

The experiments use deterministic mathematical matrix families. No empirical sensor data or external datasets are required.

## Contents

- `experiments/run_experiments.py`: closed-form constructions, SDP/scalar comparisons, direct optimization, stability experiments, and exact graph-box certificates.
- `experiments/construction_benchmark.py`: sparse construction and application benchmarks for eight repetition counts up to 65,537.
- `scripts/make_figures.py`: regenerates the six figures from the frozen data.
- `data/`: numerical results, matrix factors, exact certificates, and recorded execution environments.
- `figures/`: SVG and 600 dpi PNG exports of the six figures.
- `requirements.txt` and `experiments/requirements-solver.txt`: pinned dependencies.

## Setup

Use Python 3.12. From the repository root, create and activate a virtual environment:

```sh
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```sh
source .venv/bin/activate
```

Install the dependencies:

```sh
python -m pip install -r requirements.txt -r experiments/requirements-solver.txt
```

## Reproduce the results

Run the main experiments:

```sh
python experiments/run_experiments.py
```

The random seed is `20261030`. Results are written to `data/`, including 160 closed-form constructions, 10 SDP/scalar comparisons, 72 direct optimizer runs, 249 stability path points, and graph-box thresholds with exact rational certificates.

Run the sparse construction benchmark:

```sh
python experiments/construction_benchmark.py
```

The benchmark uses seed `20261101` and writes `construction_benchmark.csv` and `construction_environment.json` to `data/`. To keep the supplied benchmark outputs, choose another output directory:

```sh
python experiments/construction_benchmark.py --output-dir benchmark-rerun
```

Regenerate the figures from the supplied data:

```sh
python scripts/make_figures.py
```

This writes PDF, SVG, and 600 dpi PNG files to `figures/`.

The experiment and plotting commands replace their corresponding generated files. Use a working copy to preserve the supplied frozen results. Keep `experiments/`, `scripts/`, `data/`, and `figures/` together; paths are resolved relative to the scripts. Timings and local optimizer outcomes can vary with the platform and numerical libraries. The supplied environment records are in `data/environment.json` and `data/construction_environment.json`.
