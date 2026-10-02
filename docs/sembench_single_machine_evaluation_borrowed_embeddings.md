# SemBench single-machine sweep: borrowed DuckDB embeddings

This is the post-optimization successor to
[`sembench_single_machine_evaluation.md`](sembench_single_machine_evaluation.md).
It replaces the per-embedding DuckDB-to-C++ copy with borrowed views into
retained DuckDB `FLOAT[1024]` result chunks and adds cluster propagation as a
separate training-data mode.

## Sweep

| Parameter | Values |
| --- | --- |
| Datasets | SemBench Movie, FEVER |
| Rows per dataset | 3,000 |
| Embedding dimensions | 1,024 (`FLOAT[1024]`) |
| Initial sampler | Random, cluster |
| Training-data mode | Random: direct labels; cluster: direct labels and cluster propagation |
| Label budget | 32, 64, 128, 512, 1,024 |
| Total rounds | 1, 3, 5 |
| Initial-label fraction | 1.000 for one round; 0.200 for recursive runs |
| Recursive uncertainty centers | 0.5, 0.5 (the default closest-to-0.5 policy) |
| Labeler | Dataset ground-truth oracle |
| Runs | 270 total; reported values average three seeds |

`direct` trains only on oracle-labeled examples. `cluster propagation` labels
representatives directly, then trains on all rows in covered clusters using the
majority direct label per cluster. Pseudo-labels do not consume label budget
and do not exclude rows from later direct labels.

## End-to-end results

All times are milliseconds. In every numeric cell, values are ordered **round
1 / round 3 / round 5**. The timing columns intentionally match the original
single-machine report; random and cluster sampling retain their respective
schemas.

### Random sampling (direct labels)

| Dataset | Budget | Rounds | Initial selection | Initial fetch | Recursive candidate prep | Recursive select/rank | Sim. labeling | Training | Est. total | Accuracy | AURAC |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| FEVER | 32 | 1 / 3 / 5 | 0.4 / 0.3 / 0.3 | 4.0 / 1.0 / 1.0 | 0.0 / 4.0 / 3.9 | 0.0 / 5.8 / 11.6 | 426.7 / 426.7 / 426.7 | 1.0 / 2.1 / 3.4 | 432.0 / 440.0 / 446.8 | 0.540 / 0.530 / 0.521 | 0.566 / 0.561 / 0.545 |
| FEVER | 64 | 1 / 3 / 5 | 0.3 / 0.3 / 0.3 | 7.3 / 1.7 / 1.7 | 0.0 / 3.7 / 3.9 | 0.0 / 5.9 / 11.7 | 853.3 / 853.3 / 853.3 | 1.6 / 3.1 / 5.3 | 862.6 / 868.0 / 876.3 | 0.557 / 0.540 / 0.533 | 0.590 / 0.567 / 0.562 |
| FEVER | 128 | 1 / 3 / 5 | 0.4 / 0.3 / 0.3 | 14.5 / 3.1 / 3.1 | 0.0 / 3.7 / 3.9 | 0.0 / 6.0 / 12.0 | 1706.7 / 1706.7 / 1706.7 | 3.2 / 5.2 / 9.4 | 1724.7 / 1725.1 / 1735.3 | 0.577 / 0.557 / 0.568 | 0.604 / 0.576 / 0.597 |
| FEVER | 512 | 1 / 3 / 5 | 0.6 / 0.4 / 0.4 | 56.4 / 11.6 / 11.6 | 0.0 / 3.8 / 4.0 | 0.0 / 5.9 / 11.6 | 6826.7 / 6826.7 / 6826.7 | 16.5 / 25.5 / 39.7 | 6900.1 / 6873.8 / 6893.9 | 0.673 / 0.677 / 0.684 | 0.681 / 0.700 / 0.687 |
| FEVER | 1,024 | 1 / 3 / 5 | 0.8 / 0.4 / 0.7 | 111.7 / 22.7 / 30.3 | 0.0 / 4.0 / 4.2 | 0.0 / 5.5 / 10.8 | 13653.3 / 13653.3 / 13653.3 | 58.3 / 97.4 / 161.4 | 13824.1 / 13783.4 / 13860.7 | 0.765 / 0.773 / 0.769 | 0.765 / 0.846 / 0.839 |
| Movie | 32 | 1 / 3 / 5 | 0.4 / 0.3 / 0.3 | 4.0 / 1.0 / 1.1 | 0.0 / 5.1 / 3.8 | 0.0 / 6.1 / 11.8 | 426.7 / 426.7 / 426.7 | 1.9 / 2.7 / 4.2 | 433.0 / 441.9 / 448.0 | 0.803 / 0.802 / 0.810 | 0.908 / 0.902 / 0.913 |
| Movie | 64 | 1 / 3 / 5 | 0.3 / 0.3 / 0.3 | 7.3 / 1.6 / 1.7 | 0.0 / 3.7 / 3.9 | 0.0 / 6.0 / 12.1 | 853.3 / 853.3 / 853.3 | 2.4 / 4.5 / 7.3 | 863.4 / 869.4 / 878.5 | 0.831 / 0.804 / 0.832 | 0.922 / 0.891 / 0.921 |
| Movie | 128 | 1 / 3 / 5 | 0.4 / 0.3 / 0.3 | 14.4 / 3.1 / 3.1 | 0.0 / 3.9 / 3.9 | 0.0 / 6.1 / 12.2 | 1706.7 / 1706.7 / 1706.7 | 2.9 / 5.7 / 9.0 | 1724.4 / 1725.8 / 1735.2 | 0.846 / 0.864 / 0.868 | 0.915 / 0.916 / 0.922 |
| Movie | 512 | 1 / 3 / 5 | 0.6 / 0.4 / 0.4 | 56.5 / 11.5 / 11.6 | 0.0 / 3.8 / 4.0 | 0.0 / 6.0 / 11.6 | 6826.7 / 6826.7 / 6826.7 | 9.5 / 22.4 / 35.6 | 6893.2 / 6870.7 / 6889.9 | 0.859 / 0.907 / 0.909 | 0.892 / 0.915 / 0.918 |
| Movie | 1,024 | 1 / 3 / 5 | 0.8 / 0.4 / 0.6 | 112.5 / 23.0 / 36.9 | 0.0 / 3.8 / 4.7 | 0.0 / 5.5 / 12.1 | 13653.3 / 13653.3 / 13653.3 | 24.6 / 48.1 / 89.1 | 13791.2 / 13734.2 / 13796.6 | 0.885 / 0.918 / 0.925 | 0.893 / 0.948 / 0.953 |

### Cluster sampling (direct labels)

**Embedding load** reads fixed-width arrays and retains their chunks while
candidate views borrow contiguous float buffers; it performs no full embedding
copy. **K-means + reps** copies those views into mlpack's matrix, runs
k-means, and selects representatives.

| Dataset | Budget / K | Rounds | Embedding load | K-means + reps | Initial fetch | Recursive candidate prep | Recursive select/rank | Sim. labeling | Training | Est. total | Accuracy | AURAC |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| FEVER | 32 | 1 / 3 / 5 | 1.2 / 1.0 / 1.1 | 572.8 / 568.6 / 569.5 | 0.9 / 0.6 / 0.6 | 0.0 / 0.2 / 0.3 | 0.0 / 5.7 / 11.5 | 426.7 / 426.7 / 426.7 | 1.4 / 2.9 / 4.7 | 1007.2 / 1009.2 / 1017.8 | 0.528 / 0.511 / 0.513 | 0.548 / 0.513 / 0.518 |
| FEVER | 64 | 1 / 3 / 5 | 1.0 / 1.0 / 1.0 | 1164.6 / 1162.9 / 1161.2 | 1.2 / 0.7 / 0.7 | 0.0 / 0.1 / 0.3 | 0.0 / 5.9 / 11.7 | 853.3 / 853.3 / 853.3 | 2.5 / 4.5 / 7.3 | 2026.1 / 2032.1 / 2039.3 | 0.548 / 0.539 / 0.545 | 0.569 / 0.567 / 0.575 |
| FEVER | 128 | 1 / 3 / 5 | 1.1 / 1.0 / 0.9 | 2460.0 / 2455.7 / 1917.5 | 1.9 / 0.8 / 1.0 | 0.0 / 0.2 / 0.3 | 0.0 / 5.9 / 12.7 | 1706.7 / 1706.7 / 1706.7 | 4.6 / 9.1 / 14.8 | 4177.9 / 4182.8 / 3656.8 | 0.572 / 0.545 / 0.558 | 0.589 / 0.564 / 0.576 |
| FEVER | 512 | 1 / 3 / 5 | 3.1 / 0.6 / 0.6 | 7170.0 / 6460.3 / 6118.1 | 3.1 / 1.1 / 0.9 | 0.0 / 0.1 / 0.3 | 0.0 / 3.8 / 7.4 | 6826.7 / 6826.7 / 6826.7 | 13.8 / 26.1 / 48.6 | 14022.4 / 13321.0 / 13005.8 | 0.681 / 0.669 / 0.674 | 0.690 / 0.676 / 0.685 |
| FEVER | 1,024 | 1 / 3 / 5 | 0.7 / 0.7 / 0.7 | 26789.6 / 27649.4 / 29143.9 | 5.9 / 1.7 / 1.6 | 0.0 / 0.1 / 0.2 | 0.0 / 3.9 / 6.0 | 13653.3 / 13653.3 / 13653.3 | 41.0 / 64.8 / 74.0 | 40493.9 / 41376.3 / 42883.3 | 0.764 / 0.774 / 0.777 | 0.773 / 0.852 / 0.834 |
| Movie | 32 | 1 / 3 / 5 | 1.3 / 1.0 / 1.0 | 572.6 / 568.9 / 569.5 | 0.9 / 0.6 / 0.6 | 0.0 / 0.2 / 0.3 | 0.0 / 5.9 / 11.8 | 426.7 / 426.7 / 426.7 | 0.9 / 2.1 / 3.5 | 1007.0 / 1008.8 / 1016.9 | 0.830 / 0.828 / 0.823 | 0.932 / 0.924 / 0.920 |
| Movie | 64 | 1 / 3 / 5 | 1.0 / 1.0 / 1.0 | 1165.4 / 1168.8 / 1162.4 | 1.3 / 0.7 / 0.7 | 0.0 / 0.2 / 0.3 | 0.0 / 6.1 / 12.2 | 853.3 / 853.3 / 853.3 | 1.5 / 3.6 / 5.5 | 2026.0 / 2037.6 / 2038.8 | 0.858 / 0.832 / 0.848 | 0.939 / 0.917 / 0.925 |
| Movie | 128 | 1 / 3 / 5 | 1.0 / 1.0 / 1.0 | 2463.5 / 2459.7 / 2454.0 | 1.9 / 0.8 / 0.8 | 0.0 / 0.2 / 0.3 | 0.0 / 6.1 / 12.3 | 1706.7 / 1706.7 / 1706.7 | 2.9 / 5.7 / 10.1 | 4179.4 / 4183.6 / 4188.9 | 0.869 / 0.866 / 0.865 | 0.938 / 0.915 / 0.923 |
| Movie | 512 | 1 / 3 / 5 | 1.0 / 1.0 / 1.1 | 15681.2 / 16251.3 / 15690.2 | 6.0 / 1.9 / 1.7 | 0.0 / 0.2 / 0.3 | 0.0 / 6.8 / 12.4 | 6826.7 / 6826.7 / 6826.7 | 10.5 / 23.9 / 46.7 | 22528.8 / 23115.7 / 22585.1 | 0.872 / 0.905 / 0.913 | 0.906 / 0.924 / 0.925 |
| Movie | 1,024 | 1 / 3 / 5 | 1.0 / 1.2 / 2.2 | 58389.1 / 59577.1 / 61507.8 | 10.7 / 3.1 / 3.0 | 0.0 / 0.2 / 0.4 | 0.0 / 6.9 / 12.0 | 13653.3 / 13653.3 / 13653.3 | 24.2 / 54.8 / 115.4 | 72082.4 / 73300.7 / 75308.6 | 0.890 / 0.925 / 0.931 | 0.905 / 0.952 / 0.956 |

### Cluster sampling with propagation

This uses the identical cluster-sampling columns above; only the training-data
builder changes from direct labels to cluster propagation.

| Dataset | Budget / K | Rounds | Embedding load | K-means + reps | Initial fetch | Recursive candidate prep | Recursive select/rank | Sim. labeling | Training | Est. total | Accuracy | AURAC |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| FEVER | 32 | 1 / 3 / 5 | 0.7 / 0.7 / 0.7 | 362.1 / 398.5 / 386.2 | 0.7 / 0.6 / 0.5 | 0.0 / 0.1 / 0.2 | 0.0 / 4.0 / 8.3 | 426.7 / 426.7 / 426.7 | 448.1 / 420.3 / 504.6 | 1240.6 / 1253.6 / 1330.0 | 0.505 / 0.504 / 0.502 | 0.500 / 0.499 / 0.493 |
| FEVER | 64 | 1 / 3 / 5 | 0.9 / 0.7 / 0.6 | 821.7 / 751.0 / 683.9 | 1.1 / 0.5 / 0.4 | 0.0 / 0.1 / 0.2 | 0.0 / 3.9 / 7.5 | 853.3 / 853.3 / 853.3 | 736.8 / 440.9 / 640.7 | 2416.9 / 2053.1 / 2188.9 | 0.508 / 0.509 / 0.509 | 0.509 / 0.517 / 0.516 |
| FEVER | 128 | 1 / 3 / 5 | 0.8 / 0.9 / 0.8 | 1777.7 / 1624.8 / 1481.7 | 1.4 / 0.7 / 0.6 | 0.0 / 0.3 / 0.2 | 0.0 / 4.2 / 7.7 | 1706.7 / 1706.7 / 1706.7 | 1809.8 / 425.0 / 1024.5 | 5299.1 / 3765.0 / 4225.0 | 0.525 / 0.511 / 0.518 | 0.532 / 0.511 / 0.516 |
| FEVER | 512 | 1 / 3 / 5 | 0.7 / 0.7 / 0.9 | 6737.4 / 6747.1 / 7961.5 | 3.2 / 1.2 / 1.0 | 0.0 / 0.1 / 0.2 | 0.0 / 3.7 / 7.5 | 6826.7 / 6826.7 / 6826.7 | 10834.0 / 849.8 / 1019.3 | 24404.7 / 14431.8 / 15820.6 | 0.578 / 0.565 / 0.563 | 0.592 / 0.616 / 0.613 |
| FEVER | 1,024 | 1 / 3 / 5 | 1.0 / 1.1 / 1.2 | 33992.9 / 32229.8 / 28925.2 | 6.3 / 9.7 / 1.8 | 0.0 / 0.1 / 0.2 | 0.0 / 4.2 / 6.9 | 13653.3 / 13653.3 / 13653.3 | 12803.4 / 669.0 / 940.6 | 60461.5 / 46571.6 / 43534.8 | 0.645 / 0.629 / 0.624 | 0.696 / 0.709 / 0.714 |
| Movie | 32 | 1 / 3 / 5 | 1.5 / 1.7 / 1.7 | 587.0 / 589.8 / 624.6 | 0.9 / 0.7 / 0.6 | 0.0 / 0.2 / 0.4 | 0.0 / 6.2 / 14.7 | 426.7 / 426.7 / 426.7 | 709.2 / 316.8 / 898.9 | 1731.5 / 1348.6 / 1974.7 | 0.707 / 0.703 / 0.702 | 0.709 / 0.695 / 0.699 |
| Movie | 64 | 1 / 3 / 5 | 1.7 / 1.6 / 1.8 | 1164.3 / 1189.2 / 1168.3 | 1.3 / 0.7 / 0.7 | 0.0 / 0.2 / 0.4 | 0.0 / 6.2 / 16.8 | 853.3 / 853.3 / 853.3 | 470.2 / 424.2 / 1091.1 | 2497.4 / 2481.7 / 3138.6 | 0.734 / 0.671 / 0.678 | 0.728 / 0.663 / 0.670 |
| Movie | 128 | 1 / 3 / 5 | 1.4 / 1.3 / 1.4 | 2524.8 / 2505.7 / 2463.1 | 1.9 / 0.8 / 0.9 | 0.0 / 0.2 / 0.3 | 0.0 / 6.2 / 13.5 | 1706.7 / 1706.7 / 1706.7 | 457.8 / 375.8 / 982.4 | 4698.6 / 4600.9 / 5174.6 | 0.757 / 0.706 / 0.722 | 0.755 / 0.709 / 0.719 |
| Movie | 512 | 1 / 3 / 5 | 1.5 / 1.6 / 1.6 | 15867.4 / 15845.8 / 15754.6 | 5.5 / 1.6 / 1.8 | 0.0 / 0.2 / 0.4 | 0.0 / 7.2 / 14.6 | 6826.7 / 6826.7 / 6826.7 | 1669.1 / 616.6 / 1486.8 | 24375.5 / 23306.0 / 24093.2 | 0.689 / 0.769 / 0.760 | 0.690 / 0.783 / 0.774 |
| Movie | 1,024 | 1 / 3 / 5 | 2.2 / 1.9 / 1.8 | 59491.0 / 60887.2 / 91585.9 | 10.9 / 4.5 / 3.6 | 0.0 / 0.2 / 0.5 | 0.0 / 8.4 / 14.0 | 13653.3 / 13653.3 / 13653.3 | 1240.2 / 808.5 / 1276.6 | 74412.8 / 75373.6 / 106544.9 | 0.845 / 0.796 / 0.798 | 0.860 / 0.841 / 0.843 |

## Comparison with the copied-vector baseline

At K=64, averaging the three round counts against the original report:

| Dataset | Loader: copied vectors | Loader: borrowed views | Reduction | Old k-means | New k-means | Old est. total | New est. total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Movie | 306.0 | 1.0 | 99.7% | 1153.4 | 1165.5 | 2325.8 | 2034.1 |
| FEVER | 310.2 | 1.0 | 99.7% | 1151.4 | 1162.9 | 2326.1 | 2032.5 |

Borrowing removes about 305 ms from each cluster run and lowers the K=64
end-to-end estimate by about 12.5%. It does not make k-means faster; that is
now the dominant local cost. Propagation is separate because it expands the
training set and, on these slices, is slower and less accurate than direct
cluster labels.
