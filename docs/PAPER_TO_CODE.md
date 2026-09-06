# Paper to code correspondence

This page links every empirical part of the dissertation to the script that
implements it and the archived file that records its result. The final
manuscript uses the Route B evaluation for all reported instance metrics.

## Research questions and evidence

| Research component | Implementation | Archived evidence |
|---|---|---|
| DSB2018 preparation and fixed 70/15/15 split | `train_unet.py`, `routeB_instance_eval.py` | `routeB_results.json`, `routeB_per_image_metrics.csv` |
| RxRx1 experiment level split | `rxrx1_analysis.py` | `rxrx1_analysis/split_info.json` |
| Cellpose reference on DSB2018 | `baseline_cellpose_full.py`, `compute_cellpose_map.py` | `baseline/` |
| Cellpose object count reference on 200 RxRx1 images | `baseline_cellpose_rxrx1.py` | `baseline_rxrx1/` |
| U-Net reference configuration | `train_unet.py` | training histories and `unet_all_results.json` |
| Individual augmentation screen | `aug_individual.py` | `aug_individual/individual_aug_results.json` |
| Main augmentation comparison | `train_unet.py`, `multi_seed.py` | `multi_seed/multi_seed_results.json`, Route B files |
| Augmentation feature diagnostic | `aug_feature_impact.py` | `aug_feature_impact/aug_feature_impact.csv` |
| Transfer learning by labelled data quantity | `train_unet.py`, `multi_seed.py` | training histories, repeated run JSON, Route B files |
| Pseudo label volume study | `pseudo_volume_study.py` | `pseudo_volume/volume_results.json` |
| Confidence thresholds and three retraining rounds | `pseudo_label_final.py` | `pseudo_label_final/pseudo_final_results.json` |
| Combined augmentation and pseudo label experiment | `combined_experiment.py`, `routeB_instance_eval.py` | `combined/`, Route B files |
| Density and performance on the fixed test set | `density_vs_performance.py`, `paper_result_figures.py` | `density_analysis/`, `routeB_per_image_metrics.csv` |
| Paired tests over eight seeds | `statistical_tests.py` | `statistical_tests.json` |
| Final binary IoU, instance F1 and mask mAP | `routeB_instance_eval.py` | `routeB_summary.csv`, `routeB_results.json`, `routeB_report.txt` |
| Final figures from Route B results | `paper_result_figures.py` | `routeB_ap_thresholds.png`, `tl_routeB_map.png`, `density_test_only.png` |

Paths in the last column are relative to `results/`.

## Main argument and supporting records

| Dissertation conclusion | Supporting record |
|---|---|
| Geometric augmentation gives the most consistent augmentation improvement. | Eight seed results, paired Wilcoxon tests and the Route B summary |
| Adding intensity augmentation weakens the geometric result. | Augmentation comparison, individual operation screen and feature diagnostic |
| Transfer learning depends on the number of labelled images; full fine-tuning performs best with the complete training split. | Four labelled data quantities across three transfer strategies and the final Route B metrics |
| The tested pseudo label procedure does not provide a reliable improvement. | Confidence threshold runs, the decreasing volume study and combined experiment |
| A higher binary mask IoU does not necessarily mean better separation of individual nuclei. | Binary IoU, Hungarian matched instance F1 and ranked mask AP reported together |
| A general Cellpose model still requires calibration for the local image domain. | DSB2018 baseline metrics and the RxRx1 descriptive object count reference |
| Images with more nuclei tend to have lower binary IoU on the fixed test set. | Image level Route B metrics, Pearson correlations and six density bins |
| The main limitations are the absence of manual RxRx1 instance masks, one image channel for U-Net experiments and finite repeated runs. | Dataset documentation, executable channel selection and eight seed records |
| The work concerns public microscopy images and research use, not a validated clinical or drug selection system. | Dataset sources, model scope and the dissertation ethics chapter |

## Fixed experimental definitions

The executable configuration and the manuscript use the same definitions:

- DSB2018 is shuffled with seed 42 for the principal 468/101/101 split.
- Repeated runs use seeds 42, 123, 456, 789, 1024, 2048, 4096 and 8192.
- Images and binary masks are resized to 256 x 256.
- The model is a U-Net with a ResNet50 encoder, one input channel and one output class.
- Training uses binary cross entropy plus Dice loss, Adam at 0.0001, batch size 8, a cosine schedule, at most 80 epochs and patience 15.
- Geometric augmentation uses flips, rotations and elastic deformation.
- Intensity augmentation uses blur and brightness or contrast changes.
- Transfer learning compares full fine-tuning, a frozen encoder and partial fine-tuning with 40, 100, 250 and all 468 labelled training images.
- Pseudo label experiments test confidence thresholds 0.5, 0.6, 0.7, 0.8 and 0.9 for three rounds, and volumes from 0 to 500 images.
- Route B applies a foreground threshold of 0.5, forms 8-connected components, matches instances with the Hungarian algorithm and reports thresholds from 0.50 to 0.95.
- Mask mAP uses ranked predictions and 101 point interpolation.
- Confidence intervals use 2,000 paired bootstrap samples at image level.
- Density bins are 1-5, 6-15, 16-30, 31-50, 51-100 and 100+ nuclei.

## Final table source

`results/routeB_instance_eval/routeB_summary.csv` is the sole source for the
final test table. Files such as `map_results_full.json`, `full_metrics.json`
and `combined_eval_unified.json` preserve earlier stages of the project and are
not substitutes for the final table.

`budget500` in checkpoint and result names denotes the full labelled split.
The actual split contains 468 training images.

## Audit commands

The following commands check the archived evidence and recreate result figures
without raw microscopy images or checkpoints:

```bash
python scripts/verify_results.py
python scripts/statistical_tests.py
python scripts/paper_result_figures.py
```

The first command checks all 17 final model rows, the common 101 image test set,
all eight seed records, dataset summaries, paired tests, pseudo label settings
and the density analysis. A complete rerun from raw data is described in
`docs/REPRODUCIBILITY.md`.
