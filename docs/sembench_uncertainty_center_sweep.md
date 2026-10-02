# SemBench recursive uncertainty-center sweep

This report evaluates the recursive acquisition policy introduced by
`RunConfig::uncertainty_center_1` and `uncertainty_center_2`. In each
recursive round, the proxy scores every directly unlabeled candidate and
selects the rows minimizing:

```text
min(abs(score - uncertainty_center_1),
    abs(score - uncertainty_center_2))
```

Thus `(0.5, 0.5)` is the original policy: choose the rows closest to 0.5.
The two off-center policies select rows near either indicated score.

## Sweep setup

All runs use release builds, 3,000 rows with fixed-width 1,024-dimensional
DuckDB `FLOAT[1024]` embeddings, the dataset ground-truth oracle labeler,
an initial-label fraction of 0.2, and seeds 42, 43, and 44. Values below are
three-seed means.

| Dimension | Values |
| --- | --- |
| Dataset | Movie, FEVER |
| Initial/training configuration | Random + direct labels; cluster + direct labels; cluster + propagation |
| Total label budget | 64, 128, 256 |
| Cluster count | Equal to label budget for cluster configurations |
| Total rounds | 3, 5 |
| Uncertainty centers | `(0.5, 0.5)`, `(0.1, 0.9)`, `(0.2, 0.8)` |
| Repetitions | 3 seeds |
| Total runs | 324 |

One-shot runs are intentionally omitted: uncertainty centers affect only
recursive acquisition. `cluster + propagation` labels representatives with
the oracle, then expands their labels to their cluster members for training.

## Proxy-quality results: AURAC and accuracy

Each center pair has adjacent AURAC and thresholded proxy-accuracy columns.
Values are three-seed means. Bold marks the largest exact mean separately for
each metric in each row; apparent ties after rounding are not necessarily
exact ties.

### Movie

| Training configuration | Budget | Rounds | `(0.5,0.5)` AURAC | Accuracy | `(0.1,0.9)` AURAC | Accuracy | `(0.2,0.8)` AURAC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random + direct | 64 | 3 | 0.891 | 0.804 | **0.918** | 0.836 | 0.918 | **0.846** |
| Random + direct | 64 | 5 | **0.921** | 0.832 | 0.910 | **0.833** | 0.912 | 0.830 |
| Random + direct | 128 | 3 | **0.916** | **0.864** | 0.898 | 0.854 | 0.899 | 0.855 |
| Random + direct | 128 | 5 | **0.922** | **0.868** | 0.919 | 0.853 | 0.918 | 0.866 |
| Random + direct | 256 | 3 | 0.906 | 0.883 | **0.908** | 0.881 | 0.904 | **0.886** |
| Random + direct | 256 | 5 | **0.908** | 0.890 | 0.908 | **0.890** | 0.904 | 0.884 |
| Cluster + direct | 64 | 3 | 0.917 | 0.832 | 0.921 | **0.848** | **0.921** | 0.841 |
| Cluster + direct | 64 | 5 | 0.925 | **0.848** | 0.909 | 0.819 | **0.931** | 0.843 |
| Cluster + direct | 128 | 3 | 0.915 | 0.866 | 0.919 | **0.867** | **0.923** | 0.866 |
| Cluster + direct | 128 | 5 | **0.923** | 0.865 | 0.921 | **0.871** | 0.915 | 0.869 |
| Cluster + direct | 256 | 3 | **0.910** | 0.889 | 0.901 | 0.884 | 0.908 | **0.891** |
| Cluster + direct | 256 | 5 | 0.906 | 0.895 | **0.908** | 0.894 | 0.907 | **0.895** |
| Cluster + propagation | 64 | 3 | 0.663 | **0.671** | 0.660 | 0.664 | **0.663** | 0.669 |
| Cluster + propagation | 64 | 5 | 0.670 | 0.678 | 0.682 | 0.690 | **0.690** | **0.695** |
| Cluster + propagation | 128 | 3 | **0.709** | 0.706 | 0.698 | 0.697 | 0.708 | **0.708** |
| Cluster + propagation | 128 | 5 | **0.719** | **0.722** | 0.704 | 0.706 | 0.676 | 0.677 |
| Cluster + propagation | 256 | 3 | 0.729 | 0.716 | 0.721 | 0.715 | **0.743** | **0.737** |
| Cluster + propagation | 256 | 5 | 0.703 | 0.708 | 0.712 | 0.711 | **0.724** | **0.720** |

### FEVER

| Training configuration | Budget | Rounds | `(0.5,0.5)` AURAC | Accuracy | `(0.1,0.9)` AURAC | Accuracy | `(0.2,0.8)` AURAC | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random + direct | 64 | 3 | 0.567 | 0.540 | 0.555 | 0.530 | **0.591** | **0.549** |
| Random + direct | 64 | 5 | 0.562 | 0.533 | **0.585** | **0.549** | 0.559 | 0.537 |
| Random + direct | 128 | 3 | 0.576 | 0.557 | **0.611** | **0.573** | 0.581 | 0.562 |
| Random + direct | 128 | 5 | 0.597 | 0.568 | **0.601** | 0.573 | 0.596 | **0.576** |
| Random + direct | 256 | 3 | 0.615 | 0.610 | 0.623 | 0.609 | **0.632** | **0.620** |
| Random + direct | 256 | 5 | 0.623 | 0.619 | 0.611 | 0.605 | **0.643** | **0.623** |
| Cluster + direct | 64 | 3 | 0.567 | 0.539 | 0.569 | 0.540 | **0.569** | **0.546** |
| Cluster + direct | 64 | 5 | **0.575** | **0.545** | 0.555 | 0.536 | 0.573 | 0.543 |
| Cluster + direct | 128 | 3 | 0.564 | 0.545 | **0.570** | **0.560** | 0.554 | 0.544 |
| Cluster + direct | 128 | 5 | **0.576** | **0.558** | 0.562 | 0.544 | 0.557 | 0.545 |
| Cluster + direct | 256 | 3 | 0.618 | 0.608 | **0.627** | **0.621** | 0.618 | 0.608 |
| Cluster + direct | 256 | 5 | 0.616 | 0.605 | 0.619 | 0.616 | **0.638** | **0.617** |
| Cluster + propagation | 64 | 3 | **0.517** | **0.509** | 0.514 | 0.508 | 0.502 | 0.502 |
| Cluster + propagation | 64 | 5 | **0.516** | **0.509** | 0.500 | 0.509 | 0.503 | 0.506 |
| Cluster + propagation | 128 | 3 | 0.511 | 0.511 | **0.527** | 0.514 | 0.519 | **0.514** |
| Cluster + propagation | 128 | 5 | 0.516 | 0.518 | 0.513 | 0.519 | **0.521** | **0.521** |
| Cluster + propagation | 256 | 3 | 0.559 | 0.539 | **0.566** | **0.543** | 0.555 | 0.536 |
| Cluster + propagation | 256 | 5 | **0.555** | 0.535 | 0.538 | 0.530 | 0.549 | **0.538** |

## Findings

- **There is no globally optimal center pair.** Across the 36
  dataset/configuration/budget/round cells, `(0.5,0.5)` and `(0.2,0.8)` each
  have the best mean AURAC in 13 cells; `(0.1,0.9)` leads in 10.
- **FEVER random training benefits most consistently from off-center
  selection.** At budget 64, `(0.2,0.8)` improves AURAC from 0.567 to 0.591
  at three rounds; at budget 128, `(0.1,0.9)` reaches 0.611 versus 0.576 for
  the default; and at budget 256, `(0.2,0.8)` is best at both schedules.
- **Movie random training prefers the original center after the low-budget
  case.** Off-center policies help at budget 64 and three rounds, but the
  default is best at budget 128 for both schedules and tied or best at budget
  256.
- **Propagation is especially interaction-sensitive.** On Movie,
  `(0.2,0.8)` improves propagated AURAC by 0.021 at budget 64/five rounds
  and budget 256/five rounds, but the default is best at budget 128. On
  FEVER, the default remains best at budget 64 and budget 256/five rounds.

The result supports exposing centers as configuration rather than replacing
the default globally. A future policy can tune them from held-out labels or
choose them by workload, budget, and intended number of recursive rounds.

## Reproduction

```bash
cmake -S . -B /private/tmp/kea-framework-release -DCMAKE_BUILD_TYPE=Release
cmake --build /private/tmp/kea-framework-release --target sembench_sweep --parallel 4
/private/tmp/kea-framework-release/sembench_sweep \
  /private/tmp/kea_uncertainty_centers_64_128_256_release.csv \
  --uncertainty-centers
```

The CSV contains every raw run, including timing, proxy accuracy, AURAC, and
cascade-call metrics.
