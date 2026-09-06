# Script index

| Script | Purpose | Main inputs | Main outputs |
|---|---|---|---|
| `project_paths.py` | Central paths relative to the repository and overrides from environment variables | environment | path objects |
| `check_data.py` | Validate DSB2018 and RxRx1 layout | raw data | terminal report |
| `verify_results.py` | Audit archived seeds and headline result values | small JSON/CSV files | terminal report |
| `rxrx1_analysis.py` | Summarise RxRx1 metadata and prepare the stratified record | RxRx1 metadata | `results/rxrx1_analysis/` |
| `baseline_cellpose.py` | Original Cellpose baseline workflow | DSB2018 | `results/baseline/` |
| `baseline_cellpose_full.py` | Full DSB2018 Cellpose baseline | DSB2018 | `results/baseline/` |
| `baseline_cellpose_rxrx1.py` | Cellpose reference for object counts in RxRx1 channel 1 | RxRx1 images | `results/baseline_rxrx1/` |
| `train_unet.py` | Main augmentation and transfer learning training | DSB2018 | histories, JSON, checkpoints |
| `aug_individual.py` | Individual augmentation ablation | DSB2018 | `results/aug_individual/` |
| `multi_seed.py` | Repeated training with eight seeds | DSB2018 | `results/multi_seed/` |
| `pseudo_volume_study.py` | Pseudo-label volume experiment | DSB2018, RxRx1, geometric checkpoint | `results/pseudo_volume/` |
| `pseudo_label_final.py` | Confidence thresholds and iterative rounds | DSB2018, RxRx1, geometric checkpoint | `results/pseudo_label_final/` |
| `combined_experiment.py` | Augmentation plus pseudo-label experiments | both datasets, checkpoints | `results/combined/` |
| `routeB_instance_eval.py` | Final checkpoint evaluation for individual objects and bootstrap analysis | DSB2018, checkpoints | `results/routeB_instance_eval/` |
| `density_vs_performance.py` | Analysis of density and error on reserved data | DSB2018, checkpoint | `results/density_analysis/` |
| `channel_intensity_analysis.py` | Diagnostic analysis of intensity in RxRx1 channels | RxRx1 | `results/channel_analysis/` |
| `aug_feature_impact.py` | Diagnostic analysis of feature changes under augmentation | RxRx1 | `results/aug_feature_impact/` |
| `cell_type_bias_analysis.py` | Diagnostic analysis of cell type and density | RxRx1, checkpoint | `results/cell_type_bias/` |
| `statistical_tests.py` | Paired Wilcoxon comparisons | archived JSON from repeated runs | `results/statistical_tests.json` |
| `final_visualisation.py` | Main plots from archived summaries | archived results | `results/visualisations/` |
| `final_plots.py` | Additional test and failure plots | archived results, DSB2018, checkpoint | `results/visualisations/` |
| `compute_f1_full.py` | Earlier binary/threshold metric calculation | DSB2018, checkpoints | `results/full_metrics.json` |
| `compute_map_full.py` | Earlier AP calculation | DSB2018, checkpoints | `results/map_results_full.json` |
| `compute_cellpose_map.py` | Cellpose AP extension | DSB2018 | baseline result files |
| `cellpose_vs_unet.py` | Cellpose/U-Net comparison plot | archived results | visualisation |
| `literature_comparison_full.py` | Literature comparison table and plot | archived results | CSV and visualisation |
| `summary_table.py` | Consolidated legacy result table | archived results | `results/full_results_table.csv` |
| `evaluate_combined.py` | Earlier check across combined checkpoints | DSB2018, checkpoints | `results/combined_eval_unified.json` |

For the final reported results for individual objects, use `routeB_instance_eval.py` and
its outputs. The earlier metric scripts are retained to preserve the sequence of
the project record.
