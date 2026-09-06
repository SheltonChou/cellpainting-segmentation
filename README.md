# Cell Painting instance segmentation

Code and result records for the MSc project **Replicating and Optimising
Instance Segmentation Models in an Image-based Profiling Platform Using Cell
Painting**.

The study compares three ways to improve nucleus segmentation when labelled
images are limited:

1. geometric and intensity augmentation;
2. transfer learning with several quantities of labelled data; and
3. pseudo-labelling across the domain shift from DSB2018 to RxRx1.

The final evaluation reports binary mask IoU, instance F1 after Hungarian
matching, and ranked mask AP over IoU thresholds 0.50-0.95. The archived result
tables used in the report are included under `results/`.

## Key archived results

| Configuration | Test mAP 0.50-0.95 |
|---|---:|
| No augmentation | 0.343 |
| Geometric augmentation | 0.382 |
| Geometric and intensity augmentation | 0.366 |
| Full fine-tuning, full labelled split | 0.393 |
| Frozen encoder, full labelled split | 0.377 |
| Partial fine-tuning, full labelled split | 0.382 |

The authoritative table for the final instance evaluation is
`results/routeB_instance_eval/routeB_summary.csv`. Older intermediate metric
files are retained as experiment records but should not replace that table when
checking the final report.

## Repository contents

```text
.
├── README.md
├── CITATION.cff
├── environment.yml
├── requirements.txt
├── data/                         # local data only; ignored by Git
├── docs/
│   ├── DATA.md
│   ├── REPRODUCIBILITY.md
│   ├── RESULTS.md
│   └── SCRIPT_INDEX.md
├── results/
│   ├── checkpoints/             # generated locally; weights are not in Git
│   ├── multi_seed/
│   ├── routeB_instance_eval/
│   └── visualisations/
└── scripts/
    ├── project_paths.py
    ├── check_data.py
    ├── verify_results.py
    └── ... experiment scripts
```

## Quick verification without data or a GPU

This checks that the archived tables are internally consistent, that all
configurations repeated with multiple seeds contain eight observations, and that the seed list in
the executable script is the same as the documented list.

```bash
git clone https://github.com/SheltonChou/cellpainting-segmentation.git
cd cellpainting-segmentation
python scripts/verify_results.py
```

Expected final line:

```text
All archived result checks passed.
```

## Environment setup

Python 3.11 is recommended. A clean virtual environment avoids conflicts with
newer Cellpose and Albumentations APIs.

```bash
python3.11 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA workstation, install the matching PyTorch build from the official
PyTorch selector before installing the remaining requirements. Every U-Net
script falls back to CPU when CUDA is unavailable; full retraining is much
slower on CPU.

## Data setup

The datasets are not redistributed in this repository. Download them from the
original providers and arrange them as described in [`docs/DATA.md`](docs/DATA.md).
Then run:

```bash
python scripts/check_data.py
```

The default paths are defined relative to the repository. They can be overridden without
editing source code:

```bash
export DSB2018_DIR=/absolute/path/to/stage1_train
export RXRX1_ROOT=/absolute/path/to/rxrx1
export CELLPAINT_RESULTS_DIR=/absolute/path/to/results
export CELLPAINT_CHECKPOINT_DIR=/absolute/path/to/checkpoints
```

## Reproduction levels

### 1. Audit the reported results

```bash
python scripts/verify_results.py
python scripts/statistical_tests.py
python scripts/final_visualisation.py
```

The first command uses only the Python standard library. The other two recreate
the statistical summary and main result plots from archived small result files.

### 2. Evaluate saved checkpoints again

Place checkpoint files in `results/checkpoints/`, then run:

```bash
python scripts/routeB_instance_eval.py \
  --project-root . \
  --dsb-dir data/dsb2018/stage1_train \
  --scope all \
  --bootstrap 2000 \
  --bootstrap-seed 20260831
```

This recreates the final instance CSV, JSON, metrics for each image and text
report under `results/routeB_instance_eval/`.

### 3. Retrain the experiments

The full order and the input/output dependency of each command are documented
in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). The eight seeds used for repeated runs
seeds are:

```text
42, 123, 456, 789, 1024, 2048, 4096, 8192
```

Each seed controls the data shuffle and model randomness. Consequently, the
spread across runs combines split/subset variation with optimisation variation.
Exact floating-point equality across different GPU models, CUDA versions and
operating systems is not expected; the archived values are the reference
record.

A description of every file is available in
[`docs/SCRIPT_INDEX.md`](docs/SCRIPT_INDEX.md).

## Large files

Each model checkpoint is approximately 124 MB and the complete checkpoint
archive is several gigabytes. They are excluded from Git because individual
files exceed GitHub's normal 100 MB limit. Raw DSB2018 and RxRx1 files are also
excluded. The repository contains the code required to regenerate checkpoints
and the small outputs required to audit the reported results.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff).
