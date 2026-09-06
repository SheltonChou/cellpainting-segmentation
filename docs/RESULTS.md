# Result record

## Final test metrics

The final evaluation contains 101 DSB2018 images reserved for testing. Values below come
from `results/routeB_instance_eval/routeB_summary.csv`.

| Model | Binary IoU | F1@0.70 | mAP 0.50-0.95 |
|---|---:|---:|---:|
| No augmentation | 0.8214 | 0.5670 | 0.3431 |
| Geometric augmentation | 0.8473 | 0.6059 | 0.3824 |
| Geometric and intensity | 0.8339 | 0.5881 | 0.3663 |
| Full fine-tuning, 40 | 0.7591 | 0.4793 | 0.2576 |
| Frozen encoder, 40 | 0.7220 | 0.4239 | 0.2109 |
| Partial fine-tuning, 40 | 0.7500 | 0.4593 | 0.2257 |
| Full fine-tuning, 100 | 0.8097 | 0.5631 | 0.3120 |
| Frozen encoder, 100 | 0.7940 | 0.5390 | 0.3189 |
| Partial fine-tuning, 100 | 0.8141 | 0.5608 | 0.3164 |
| Full fine-tuning, 250 | 0.8284 | 0.5790 | 0.3596 |
| Frozen encoder, 250 | 0.8220 | 0.5709 | 0.3491 |
| Partial fine-tuning, 250 | 0.8272 | 0.5669 | 0.3522 |
| Full fine-tuning, full split | 0.8512 | 0.6264 | 0.3928 |
| Frozen encoder, full split | 0.8367 | 0.5948 | 0.3773 |
| Partial fine-tuning, full split | 0.8410 | 0.5968 | 0.3817 |
| Geometric plus pseudo | 0.8273 | 0.5850 | 0.3810 |
| Geometric, intensity and pseudo | 0.8320 | 0.5944 | 0.3852 |

The filenames use `budget500` for the full labelled split; this is a historical
configuration name rather than a claim that 500 labelled training images were
used.

## Comparisons across repeated runs

Each augmentation and transfer learning entry in
`results/multi_seed/multi_seed_results.json` contains eight validation IoU
values in the documented seed order. The selected paired Wilcoxon results are
stored in `results/statistical_tests.json`.

## Pseudo-label volume

Archived validation IoU values:

| Pseudo-labelled images | Validation IoU |
|---:|---:|
| 0 | 0.8776 |
| 100 | 0.8712 |
| 200 | 0.8688 |
| 300 | 0.8682 |
| 400 | 0.8638 |
| 500 | 0.8632 |

These values are descriptive records of the saved run. Claims about the final test
results should be checked against the Route B instance outputs above.
