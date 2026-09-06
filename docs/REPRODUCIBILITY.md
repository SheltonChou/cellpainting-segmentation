# Detailed reproduction workflow

Run commands from the repository root. All outputs default to `results/`, and
model weights default to `results/checkpoints/`.

## 0. Initial checks

```bash
source .venv/bin/activate
python scripts/check_data.py
python scripts/verify_results.py
```

Record the software and hardware used for a new run:

```bash
python --version
python -m pip freeze > results/environment-freeze.txt
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

## 1. Dataset summaries and Cellpose references

```bash
python scripts/rxrx1_analysis.py
python scripts/baseline_cellpose_full.py
python scripts/baseline_cellpose_rxrx1.py
```

Main outputs:

- `results/rxrx1_analysis/`
- `results/baseline/`
- `results/baseline_rxrx1/`

The DSB2018 Cellpose baseline uses the `cyto` model. The RxRx1 descriptive
baseline uses channel `w1`, the `nuclei` model and diameter 15.

## 2. Main U-Net experiments

```bash
python scripts/train_unet.py
```

This trains the augmentation and transfer learning configurations and writes
histories, summary JSON files and checkpoints. The network is a U-Net with a
ResNet-50 encoder, one input channel and one output class. Images and masks are
resized to 256 x 256. The loss is binary cross-entropy plus Dice loss.

## 3. Individual augmentation ablation

```bash
python scripts/aug_individual.py
```

This separates flips, rotations, elastic deformation, blur and intensity
operations. Outputs are written to `results/aug_individual/`.

## 4. Repeated runs with eight seeds

```bash
python scripts/multi_seed.py
```

Seed order:

```text
42, 123, 456, 789, 1024, 2048, 4096, 8192
```

The script reruns augmentation and transfer learning configurations. It uses a
70/15/15 split after shuffling with the selected seed. Within a seed, configurations use
the same shuffled partition and labelled subset. Across seeds, the partition and
subset change. The resulting standard deviations therefore combine data
sampling and optimisation variation.

Output: `results/multi_seed/multi_seed_results.json`.

## 5. Statistical comparisons

```bash
python scripts/statistical_tests.py
```

The paired Wilcoxon tests operate on the eight validation IoU values in the same
seed order. With eight paired differences that are not zero, the minimum attainable
exact p-value for a two sided test is 0.0078125.

## 6. Pseudo-labelling

Run the volume study and the iterative experiment with confidence thresholds:

```bash
python scripts/pseudo_volume_study.py
python scripts/pseudo_label_final.py
```

The volume study uses 0, 100, 200, 300, 400 and 500 pseudo-labelled RxRx1
images. The threshold experiment evaluates 0.5, 0.6, 0.7, 0.8 and 0.9 for up to
three retraining rounds. Both scripts start from the checkpoint obtained with
geometric augmentation, so `results/checkpoints/aug_geometric_only.pth` must exist.

## 7. Combined experiment

```bash
python scripts/combined_experiment.py
```

This compares geometric augmentation alone with configurations that add pseudo labels,
with and without intensity augmentation. It requires the geometric
checkpoint and the archived pseudo-label selection record under
`results/pseudo_label/`.

## 8. Final instance evaluation

```bash
python scripts/routeB_instance_eval.py \
  --project-root . \
  --dsb-dir data/dsb2018/stage1_train \
  --scope all \
  --bootstrap 2000 \
  --bootstrap-seed 20260831
```

Evaluation details:

- fixed DSB split seed: 42;
- foreground probability threshold: 0.5;
- predicted instances: 8-connected components;
- matching: one to one Hungarian assignment;
- F1 IoU thresholds: 0.50-0.95;
- ranked mask AP: interpolated precision and recall at 101 points;
- uncertainty: 2,000 paired bootstrap resamples at image level.

The command writes:

- `routeB_summary.csv`;
- `routeB_results.json`;
- `routeB_report.txt`; and
- `routeB_per_image_metrics.csv`.

## 9. Figures and final consistency check

```bash
python scripts/paper_result_figures.py
python scripts/final_visualisation.py
python scripts/final_plots.py
python scripts/verify_results.py
```

`paper_result_figures.py` recreates the Route B AP threshold plot, transfer
learning mAP plot and fixed test set density plot directly from the final
archived evaluation. The two broader plotting scripts retain additional
diagnostic and earlier project figures.

## 10. Density analysis

```bash
python scripts/density_vs_performance.py
```

This command reconstructs the seed 42 split and evaluates only the reserved
101 image test set. It writes the image level table and Pearson correlation
summary to `results/density_analysis/`. The density bins are 1-5, 6-15, 16-30,
31-50, 51-100 and 100+ nuclei. The final paper figure can then be recreated
from the Route B image level metrics with `paper_result_figures.py`.

Do not overwrite the archived results until a complete run has finished. For a
new replication, point `CELLPAINT_RESULTS_DIR` to a new directory and retain the
generated environment freeze and terminal log.
