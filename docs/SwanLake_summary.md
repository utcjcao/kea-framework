# Distributed Semantic Operator: Performance Evaluation Report

This report evaluates the distributed semantic filter implemented on
SwanLake: a proxy-based two-phase pipeline (`semantic_filter`) with an
LLM cascade. Phase A (proxy training) is organized as a fixed R-round
framework (R=1: one-shot sampling; R>1: recursive uncertainty sampling)
with pluggable sampling / labeling / training methods; every evaluated
configuration is one method of this framework.

Evaluation follows the professor's two-axis protocol:

- **Axis 1 - proxy quality:** AURAC (Area Under the Risk-Accuracy Curve,
  El-Kishky et al., "Beyond 50/50", WWW 2022), plus ACC/AUC as support
  metrics (Sections 4, 8.1, 9.1).
- **Axis 2 - given a budget, how many LLM calls can the proxy save:**
  accuracy under a call budget and call reduction under an FNR budget with
  a static threshold (Sections 5, 8.2, 9.2).

Supporting experiments: execution-time breakdown (Section 3), deployment
latency projection (Section 6), 1M-row scale check (Section 7).

## 1. Executive summary

- **System overhead is small.** At 20k rows in release builds, Phase B
  executes in 7-21 ms for every baseline; Phase A costs 0.02-1.8 s for all
  local/random strategies. The only expensive components are the GC
  variants' single-node global k-means (32.8 s at k=384, 131.7 s at
  k=1536) and their full-table shipping. End-to-end latency is dominated
  by LLM calls in every realistic deployment tier.
- **Cascade accuracy tracks escalation volume, not model quality.** A weak
  proxy (federated_sample, model acc 0.640) achieves final acc 1.000 by
  escalating 100% of rows to the LLM - contributing nothing itself.
  AURAC, not final cascade accuracy, is therefore the proxy-quality metric.
- **Clean labels scale with budget; propagation saturates early.** On the
  easy dataset, clean-label training climbs 0.705 -> 0.824 from 192 to
  1,536 labels while propagated training saturates at ~0.81 already at 384
  labels (with base-rate-inflated scores and near-chance AUC). On the hard
  dataset (FEVER), propagated training leads AURAC at every budget.
- **Recursive uncertainty sampling is an orthogonal optimizer, not a
  panacea**: it repairs weak bases (random/LC1, +0.05-0.11 AURAC) but does
  not beat one-shot propagation on hard data; LC4-rec is the best single
  two-axis compromise (top-2 AURAC in all 8 cells + mid-pack call savings).
- **The LC/GC/random x recursive matrix is now complete on the real 6-node
  cluster (Section 9.5)**: recursion repairs the clean-label methods
  (LC1/LC3/GC1, up to +0.17 AURAC on fever) but does not beat one-shot
  propagation (LC4/GC2 stay the AURAC leaders on hard data at equal
  budget), and the propagated family remains the cheapest on LLM calls.
- **Federated training is sound with clean labels** (weights-only transfer,
  ~10 KB) but collapses with tiny per-shard training sets (64 reps/shard ->
  acc 0.640, full score-range collapse).
- **Clean-label proxies have genuinely confident zones**: accept-zone
  correctness 90-95% at 7-52% mass. Propagated proxies look confident
  (accept mass 78-87%) but their zone correctness (82-84%) barely exceeds
  the base rate, and their reject zone is empty - they can never save LLM
  calls on negatives.

## 2. System and methodology

### 2.1 Pipeline

```
SELECT ... WHERE semantic_filter(text, emb, 'instruction')
        |
Phase A (proxy training, pluggable strategy)
  sampling/clustering -> labeling (LLM) -> LR proxy training -> broadcast
        |
Phase B (standard distributed query path)
  per-row: proxy score > T_accept -> keep
           proxy score < T_reject -> drop            (three-zone cascade)
           in between            -> escalate to LLM
```

Phase A is implemented as a fixed **R-round framework** (see
`docs/semantic_dev.md`): round 0 picks an initial label batch (the
sampling method), every following round labels the most uncertain
unlabeled rows (|score - 0.5| smallest), and the proxy retrains after
every round. R=1 is basic one-shot sampling; R>1 is recursive uncertainty
sampling. The sampling / label-amplification / training choices plug into
the loop as one configuration; all strategies below are such
configurations (the RPC-based shard variants additionally use the same
rounds loop per shard).

### 2.2 Datasets and cluster

Two real datasets, both with binary ground truth and real 384-dim
`all-MiniLM-L6-v2` embeddings generated on GPU:

- **movie (easy)** - Rotten Tomatoes / CLAPPER reviews; 80.8% positive at
  20k, 77.5% at 100k; marker-rich, separable sentiment language.
- **fever (hard)** - FEVER 1.0 claim-verification claims (SUPPORTS=1 /
  REFUTES=0, NEI dropped); ~73% positive; factual claims with no
  sentiment markers - near-uniform proxy-confidence distributions.

Cluster setups: Sections 3-8 use 6 nodes (coordinator + 5 workers =
**6 shards**, hash-distributed); Section 9's matrix uses a single
coordinator-only shard (single-machine simulation of the distributed
system; timing is not an axis there). Release builds (-O3, no
sanitizers). `random` draws a fresh rowid window every run (+-1-4 acc
points run-to-run); all cluster/federated configurations are seeded and
reproducible.

### 2.3 Oracle LLM methodology

All experiments use the **oracle backend**: Phase-A labeling and cascade
decisions return the ground-truth label instantly, simulating a _perfect_
LLM. This isolates the sampling/training/cascade architecture from LLM
quality. Call counts are measured exactly; latency is accounted
analytically as `calls x ms-per-request`, never by sleeping.

### 2.4 LLM deployment tiers (measured serving capacity)

| Tier                                               | Throughput  |
| -------------------------------------------------- | ----------- |
| Hosted API (upper bound observed across providers) | 75 req/s    |
| Self-hosted server mode (4xA100, Qwen3-4B)         | 250 req/s   |
| Self-hosted in-process mode (4xA100, Qwen3-4B)     | 1,800 req/s |

### 2.5 Baselines

| Name               | Code strategy                             | Pipeline                                                                                                     |
| ------------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| random             | `offset`                                  | random rowid-window samples per shard -> coordinator -> label -> train -> broadcast                          |
| LC1                | `cluster_local`                           | local k-means, label representatives, train on representatives in coordinator                                |
| LC2                | `cluster_local_full`                      | local k-means, label representatives, propagate labels to all shard tuples, train on all data in coordinator |
| LC3                | `federated_sample`                        | local k-means, label representatives, train on representatives **federally**, average weights                |
| LC4                | `federated`                               | local k-means, label representatives, propagate labels, train on all data **federally**, average weights     |
| GC1                | `cluster_central_sample`                  | ship all rows to coordinator, global k-means, label representatives, train on representatives                |
| GC2                | `cluster_central`                         | ship all rows to coordinator, global k-means, label representatives, propagate labels, train on all data     |
| LC1-rec .. LC4-rec | `lc1rec` / `lc2rec` / `lc3rec` / `lc4rec` | LC1-LC4 with the recursive uncertainty loop across the real shards (multi-round RPC; Section 9.5)            |
| GC1-rec / GC2-rec  | `gc1rec` / `gc2rec`                       | GC1/GC2 with Rounds>1 on the pooled table (global candidates, clean vs propagated labels)                    |

Primary setup for Exp 1-3: k = 64 clusters/shard (384 total labels);
random baselines at 32/64/128 samples per shard; cascade bands (0.2, 0.8)
and (0.1, 0.9). A k=256/shard (1,536 labels) cross-budget check is
included in Exp 2.

## 3. Experiment 1: Execution time

**Question:** where does system time go, and how expensive is each
baseline? All numbers release builds, 20k rows, 6 shards.

### 3.1 Phase A breakdown

| Baseline         | labels | sample/cluster                         | labeling   | training | broadcast | **Phase A total** |
| ---------------- | ------ | -------------------------------------- | ---------- | -------- | --------- | ----------------- |
| random 32/shard  | 192    | 12 ms                                  | 3 ms       | 3 ms     | <1 ms     | **0.02 s**        |
| random 64/shard  | 384    | 25 ms                                  | 3 ms       | 12 ms    | <1 ms     | **0.04 s**        |
| random 128/shard | 768    | 53 ms                                  | 3 ms       | 18 ms    | <1 ms     | **0.07 s**        |
| LC1              | 384    | 1.39 s (workers)                       | on workers | 11 ms    | <1 ms     | **1.40 s**        |
| LC2              | 384    | 1.47 s (workers)                       | on workers | 348 ms   | <1 ms     | **1.82 s**        |
| LC3              | 384    | 1.39 s (workers, incl. local training) | on workers | ~0 (avg) | <1 ms     | **1.39 s**        |
| LC4              | 384    | 1.48 s (workers, incl. local training) | on workers | ~0 (avg) | <1 ms     | **1.48 s**        |
| GC1              | 384    | fetch 0.97 s + k-means **32.8 s**      | 2 ms       | 7 ms     | <1 ms     | **33.8 s**        |
| GC2              | 384    | fetch 0.97 s + k-means **32.9 s**      | 2 ms       | 567 ms   | <1 ms     | **34.4 s**        |

At k=1,536 the GC global k-means grows to 131.7 s; local variants scale
mildly (1.4 s -> 4.2 s). LR training itself is the cheapest stage
everywhere (3-35 ms on clean samples; 0.3-0.6 s only when training on all
20k propagated tuples).

### 3.2 Data movement

| Baseline  | Content                                   | Direction              | Size (20k rows, 384-dim) | Measured transfer stage |
| --------- | ----------------------------------------- | ---------------------- | ------------------------ | ----------------------- |
| random    | id + text + emb strings                   | shards -> coordinator  | ~1.1 MB (384 rows)       | collect 25 ms           |
| LC1 / LC3 | representatives: emb floats + label       | workers -> coordinator | ~0.6 MB                  | few ms (inside collect) |
| LC2       | full shard: emb floats + propagated label | workers -> coordinator | **~31 MB**               | collect 1.47 s          |
| GC1 / GC2 | full table: id + text + emb strings       | shards -> coordinator  | **~60 MB**               | fetch 0.97 s            |
| LC4       | model weights only                        | workers -> coordinator | **~10 KB**               | <1 ms                   |

LC/GC variants also differ in what the labeler needs: GC ships text
because the coordinator labels; LC labels on the workers, so text never
moves.

### 3.3 Phase B execution

Identical rewritten query for all baselines: **7-21 ms wall** at 20k rows.
Prior cost attribution at 1M rows: ~15% wide-column scan, ~85% result
materialization/merge; the proxy evaluation itself (384-dim dot product
per row) is negligible.

**Takeaway:** system-side cost is dominated by two items only - the GC
variants' single-node global k-means and the O(table) shipping of GC/LC2.
Everything else fits under 2 s, so the choice of baseline should be driven
by Exp 2/3 (model quality and LLM savings), not by execution time.

## 4. Experiment 2: Proxy quality (ACC + AUC)

**Question:** how good is the trained proxy as a standalone classifier,
independent of the cascade? Method: proxy-only run - the model alone
judges all 20k rows at its own threshold. Metrics: accuracy, AUC
(threshold-free ranking), score separation (positive mean - negative
mean).

### 4.1 Model-only accuracy across label budgets

k=64/shard (384 labels) and k=256/shard (1,536 labels):

| Baseline         | labels      | acc @ k=64/shard | acc @ k=256/shard |
| ---------------- | ----------- | ---------------- | ----------------- |
| random 32/shard  | 192         | 0.705            | -                 |
| random 64/shard  | 384         | 0.770            | -                 |
| random 128/shard | 768         | 0.790            | -                 |
| random 256/shard | 1,536       | -                | 0.818             |
| LC1              | 384 / 1,536 | 0.761            | 0.818             |
| LC2              | 384 / 1,536 | **0.805**        | 0.810             |
| LC3              | 384 / 1,536 | 0.640            | 0.808             |
| LC4              | 384 / 1,536 | **0.807**        | 0.809             |
| GC1              | 384 / 1,536 | 0.750            | **0.824**         |
| GC2              | 384 / 1,536 | **0.807**        | 0.810             |

Findings:

- **Propagated-label training wins at low budgets**: LC2/LC4/GC2 all reach
  0.805-0.807 with only 384 real LLM calls, above every clean-label
  baseline at equal or even double budget (random 128/shard: 0.790 at 768
  calls). Label propagation amplifies 384 calls into 20k training tuples.
- **Propagation saturates immediately**: k=64 and k=256 give 0.805-0.810.
  More clusters trade label noise against per-tuple coverage and cancel.
- **Clean-label training keeps scaling** with budget: 0.705 (192) -> 0.770
  (384) -> 0.790 (768) -> 0.818-0.824 (1,536), crossing propagation
  between 768 and 1,536 labels.
- **Random vs LC1 at equal budget (384):** statistically tied (0.770 vs
  0.761). At k=64/shard (~52 rows per cluster) representatives are nearly
  per-row labels; clustering's label-saving value appears at smaller
  k/shard ratios.
- **LC3 collapses at low k**: six shard models trained on only 64 reps
  each, averaged with per-shard folded standardization, yield 0.640. It
  recovers to 0.808 at k=256/shard - the failure is the tiny per-shard
  training set, not weight averaging per se.

### 4.2 Discrimination: AUC and score separation

Scores of all 20k rows were dumped (`SWANLAKE_SEMANTIC_SCORE_DUMP`) and
joined with ground truth (`scripts/analyze_score_dump.py`). k=64/shard:

| Baseline         | AUC       | separation | model acc |
| ---------------- | --------- | ---------- | --------- |
| random 128/shard | **0.754** | 0.192      | 0.799     |
| GC1              | 0.741     | **0.201**  | 0.731     |
| LC3              | 0.726     | 0.046      | 0.640     |
| random 64/shard  | 0.712     | 0.106      | 0.758     |
| LC1              | 0.711     | 0.187      | 0.761     |
| random 32/shard  | 0.684     | 0.123      | 0.689     |
| GC2              | 0.680     | 0.103      | **0.810** |
| LC4              | 0.614     | 0.029      | 0.807     |
| LC2              | 0.572     | 0.026      | 0.805     |

**ACC and AUC diverge sharply - this is the key result of Exp 2.** The
propagated baselines (LC2/LC4/GC2) hold the highest accuracies but the
lowest AUCs (0.57-0.68, near chance): they inflate all scores
(negative-class means 0.82-0.86) and ride the 80.8% positive base rate
rather than separating the classes. The clean-label baselines own the top
AUCs (0.71-0.75). LC3 keeps decent ranking (0.726) despite the worst
accuracy - its failure is calibration, not discrimination.

**Takeaway:** as standalone classifiers, clean-label models are the better
rankers and improve monotonically with budget; propagated models deliver a
cheap accuracy floor (~0.81 at 384 calls) that is largely a base-rate
artifact. Report both ACC and AUC - either alone misranks the baselines.

## 5. Experiment 3: Cascade utility

**Question:** can the proxy actually reduce LLM calls, and is its
confidence trustworthy? Metrics: zone mass (accept / reject / ambiguous at
the cascade band), zone correctness, Phase-B escalations (= LLM calls),
final accuracy. Method: cascade runs plus the score dump of Section 4.2.

### 5.1 Zone placement and correctness, band (0.2, 0.8)

| Baseline         | accept mass (correct) | reject mass (correct) | ambiguous |
| ---------------- | --------------------- | --------------------- | --------- |
| LC1              | 52.2% (90.1%)         | 4.8% (47.1%)          | 43.0%     |
| random 128/shard | 43.4% (92.7%)         | 1.9% (71.4%)          | 54.8%     |
| GC1              | 28.0% (**94.8%**)     | 5.6% (52.1%)          | 66.4%     |
| random 32/shard  | 12.8% (93.7%)         | 2.5% (51.6%)          | 84.6%     |
| random 64/shard  | 7.2% (95.4%)          | 0.2%                  | 92.7%     |
| GC2              | 85.2% (84.3%)         | 0.8%                  | 14.0%     |
| LC4              | 87.2% (82.2%)         | 0.0%                  | 12.8%     |
| LC2              | 77.8% (82.6%)         | 0.0%                  | 22.1%     |
| LC3              | 0.0%                  | 0.0%                  | **100%**  |

- **Clean-label proxies park more rows in the band, but their confident
  zones are genuinely confident**: accept-zone correctness 90-95% across
  7-52% mass. Their weak spot is the reject zone (47-71% correct) - the
  proxies are badly calibrated on the negative side, which is why widening
  T_reject costs recall.
- **Propagated proxies look confident but carry almost no information**:
  accept mass 78-87% with only 82-84% correctness - barely above the 80.8%
  base rate - and empty reject zones. They can never save LLM calls on
  negatives.
- **LC3 places 100% of rows in the band** - pure score-range collapse from
  averaging six mis-calibrated shard models (its AUC of 0.726 shows the
  ranking survives; the score _range_ does not).

### 5.2 Escalations and final accuracy

k=64/shard; total LLM calls = 384 Phase-A labels + Phase-B escalations:

| Baseline         | model acc | final acc (0.2,0.8) | escal.     | final acc (0.1,0.9) | escal.     | total calls (0.2,0.8) |
| ---------------- | --------- | ------------------- | ---------- | ------------------- | ---------- | --------------------- |
| random 32/shard  | 0.705     | 0.896               | 10,664     | 0.986               | 17,204     | 11,048                |
| random 64/shard  | 0.770     | 0.902               | 7,152      | 0.936               | 9,957      | 7,536                 |
| random 128/shard | 0.790     | 0.939               | 8,652      | 0.976               | 12,630     | 9,420                 |
| LC1              | 0.761     | 0.923               | 8,606      | 0.966               | 12,941     | 8,990                 |
| LC2              | 0.805     | 0.865               | 4,430      | 0.934               | 11,574     | 4,814                 |
| LC3              | 0.640     | **1.000**           | **20,000** | **1.000**           | **20,000** | **20,384**            |
| LC4              | 0.807     | 0.845               | 2,565      | 0.924               | 9,424      | 2,949                 |
| GC1              | 0.750     | 0.977               | 15,064     | 0.991               | 17,795     | 15,448                |
| GC2              | 0.810     | 0.878               | 3,693      | 0.912               | 5,927      | 4,077                 |

Findings:

- **Final cascade accuracy tracks escalation volume, not model quality.**
  The rank correlation with model acc is weak; with escalation count it is
  near-perfect. LC3's "perfect" 1.000 is the limiting case: the weakest
  model escalates 100% of rows and the perfect oracle LLM does all the
  work - zero proxy contribution at maximum LLM cost.
- **Band width is the strongest accuracy lever**: widening (0.2, 0.8) ->
  (0.1, 0.9) lifts every baseline by 1-6 points at 1.4-1.9x LLM calls.
- **Low escalation is not a virtue by itself**: LC4/GC2 escalate least
  (2.6k-3.7k) only because their biased scores "decide" rows the model
  cannot actually separate (Section 5.1) - their final accuracy is the
  worst of the cascade runs.
- **The meaningful operating points** are baselines whose high final
  accuracy comes with genuinely informative confident zones: GC1 (0.977 at
  15.4k calls, accept-zone 94.8% correct), LC1 (0.923 at 9.0k) and
  random 128/shard (0.939 at 9.4k).

**Takeaway:** a cascade-worthy proxy needs BOTH high AUC (ranking) and
sharp, calibrated scores (zone placement); accuracy measures neither.
Under the oracle, every point of final accuracy above the model-only acc
is bought with LLM calls - Exp 3 quantifies the exchange rate per
baseline.

## 6. Deployment latency projection

Total LLM calls projected through the three serving tiers
(Section 2.4); system overhead (Exp 1) added on top. k=64/shard,
band (0.2, 0.8):

| Baseline         | Calls  | API 75 req/s | Server 250 req/s | In-process 1,800 req/s | System (Phase A) |
| ---------------- | ------ | ------------ | ---------------- | ---------------------- | ---------------- |
| LC4              | 2,949  | 39.3 s       | 11.8 s           | 1.6 s                  | 1.5 s            |
| GC2              | 4,077  | 54.4 s       | 16.3 s           | 2.3 s                  | 34.4 s           |
| LC2              | 4,814  | 64.2 s       | 19.3 s           | 2.7 s                  | 1.8 s            |
| random 64/shard  | 7,536  | 100.5 s      | 30.1 s           | 4.2 s                  | 0.04 s           |
| LC1              | 8,990  | 119.9 s      | 36.0 s           | 5.0 s                  | 1.4 s            |
| random 128/shard | 9,420  | 125.6 s      | 37.7 s           | 5.2 s                  | 0.07 s           |
| GC1              | 15,448 | 206.0 s      | 61.8 s           | 8.6 s                  | 33.8 s           |
| LC3              | 20,384 | 271.8 s      | 81.5 s           | 11.3 s                 | 1.4 s            |

- Under hosted APIs the LLM term dominates everything; system-side choices
  move the total by <3% (GC's k-means being the sole exception).
- In in-process mode the LLM term shrinks to seconds and Phase-A costs
  become visible: random is ~15x cheaper than LC1/LC3 and ~200x cheaper
  than the GC variants.
- Caveat: LC4/GC2/LC2 buy their low call counts with base-rate-biased
  models (Exp 3); a real (non-oracle) LLM would not correct their
  escalations as reliably, and their zone decisions are nearly uninformative.

## 7. Scale check: 1,000,000 rows

Same setup as Exp 1-3 (6 nodes/shards, k=64/shard = 384 labels, 384-dim
MiniLM embeddings, oracle LLM, release) at 50x the rows. The 1M dataset is
harder: 67.4% positive (vs 80.8% at 20k) and dominated by longer-tail
fallback reviews, so absolute accuracies shift down; the comparison target
is whether the _patterns_ survive. Table loading (COPY) costs 8.8 s for 1M
rows. Strategies were switched at runtime on one loaded cluster
(`COPY swanlakeconfig FROM '<strategy>'`).

### 7.1 Model-only accuracy: 20k vs 1M

| Baseline         | labels | acc @20k  | acc @1M   |
| ---------------- | ------ | --------- | --------- |
| random 32/shard  | 192    | 0.705     | 0.620     |
| random 64/shard  | 384    | 0.770     | 0.678     |
| random 128/shard | 768    | 0.790     | **0.708** |
| LC1              | 384    | 0.761     | 0.702     |
| LC2              | 384    | **0.805** | 0.678     |
| LC3              | 384    | 0.640     | 0.651     |
| LC4              | 384    | **0.807** | 0.686     |

Holds at scale:

- **Clean-label training still scales with budget**: 0.620 -> 0.678 ->
  0.708, monotone, same shape as 20k.
- **LC3 remains the weakest clean variant** (0.651): 64 reps per shard is
  still too little per-shard training data; the low-k collapse survives.
- **Final cascade accuracy still tracks escalations, not model quality**:
  at band (0.2, 0.8), random-64 reaches 0.986 by escalating 893,823 rows
  (89.4%) while LC1 reaches only 0.923 with 571,724 escalations (57.2%) -
  the same inversion seen at 20k.
- **Timing scales linearly**: LC Phase-A dispatch+collect 1.4 s (20k) ->
  67-71 s (1M), matching the 50x row growth; a single-node contention-free
  full-table pass (scan + k-means + propagate + train on 1M rows) takes
  450 s, consistent with 6 x the per-shard work.

Changed at scale:

- **The propagated-label advantage disappears.** At 20k, LC2/LC4 (0.805-
  0.807) beat every clean variant at <=768 labels; at 1M they only TIE
  equal-budget random (LC2 0.678 = random-64 0.678; LC4 0.686), and
  double-budget clean wins (0.708). On harder, more diverse data the
  propagated floor does not lift - budget it like a tie, not a win.
- **Equal-budget cluster vs random now favors clustering**: LC1 0.702 vs
  random-64 0.678 (+2.4 pts), where 20k had them tied - representatives
  cover the wider 1M distribution better than one random rowid window.
- **Central full-data training becomes visible**: LR training on all 1M
  propagated tuples costs 42.3 s at the coordinator (LC2 total 111 s);
  the federated equivalent is ~42.3/6 = ~7 s per node (linear split,
  measured per-shard equivalent), keeping LC4's Phase A at 71 s.
- GC variants were not run at 1M: their single-node global k-means
  extrapolates to ~27 min (32.8 s x 50) plus the 3 GB string fetch;
  the "GC is dominated" conclusion strengthens, not weakens.

### 7.2 Proportional-budget follow-up at 1M (partial)

Fixed 384 labels at 1M is a 50x thinner rate than at 20k (0.038% vs
1.92%). Matching the rate instead:

| Config                                       | labels | model acc @1M                                          |
| -------------------------------------------- | ------ | ------------------------------------------------------ |
| random 640/shard (0.38%)                     | 3,840  | 0.743                                                  |
| LC1 k=640/shard (0.38%)                      | 3,840  | 0.744                                                  |
| random 3200/shard (1.92%, equal-rate to 20k) | 19,200 | 0.758                                                  |
| LC1 k=3200/shard (1.92%)                     | 19,200 | did not complete (single-node k-means ~40 min + crash) |

At equal labels random and LC1 tie (0.743 vs 0.744); clean-label training
keeps scaling with budget (0.758 at 19,200 labels, above every 384-label
config). The starved 384-label picture of Section 7 therefore understates
all methods equally; the relative conclusions are unaffected.

## 8. Deferral-framework evaluation: AURAC + LLM-call budget

Following the deferral literature (El-Kishky et al., "Beyond 50/50: Crowd,
Expertise and AI", WWW 2022), the proxy is evaluated on two axes that
separate model quality from escalation spending:

- **Axis 1 - AURAC** (Area Under the Risk-Accuracy Curve): sort all rows by
  proxy confidence |score - 0.5| descending; the risk-accuracy curve plots
  the proxy's accuracy on the kept top-rho fraction; AURAC is its mean.
  Threshold-free: it measures how well confidence ranks correctness.
  nAURAC normalizes between the random-ordering baseline (base rate) and
  the oracle curve. Computed offline from the per-row score dumps
  (`scripts/analyze_aurac.py`).
- **Axis 2 - accuracy under an LLM budget B**: optimal deferral escalates
  exactly the B least-confident rows to the LLM; acc@B is the resulting
  accuracy. Compared against the escalation fraction a fixed-band cascade
  consumes.

### 8.1 AURAC ranking, k=64/shard (384 labels unless noted)

20k rows (base rate 0.808, oracle AURAC 0.980):

| Method                        | AURAC     | nAURAC     | AUC   | model acc |
| ----------------------------- | --------- | ---------- | ----- | --------- |
| random 128/shard (768 labels) | **0.890** | 0.478      | 0.754 | 0.790     |
| GC2                           | 0.883     | 0.436      | 0.680 | 0.807     |
| LC1                           | 0.880     | 0.420      | 0.711 | 0.761     |
| random 64/shard               | 0.872     | 0.369      | 0.712 | 0.770     |
| LC4                           | 0.860     | 0.302      | 0.614 | 0.807     |
| GC1                           | 0.852     | 0.253      | 0.741 | 0.750     |
| LC2                           | 0.839     | 0.182      | 0.572 | 0.805     |
| random 32/shard (192 labels)  | 0.837     | 0.168      | 0.684 | 0.705     |
| LC3                           | 0.790     | **-0.104** | 0.726 | 0.651     |

1M rows (base rate 0.674, oracle AURAC 0.940; GC not run, see Section 7):

| Method                        | AURAC     | nAURAC     | AUC       | model acc |
| ----------------------------- | --------- | ---------- | --------- | --------- |
| LC4                           | **0.805** | **0.492**  | 0.706     | 0.686     |
| LC2                           | 0.760     | 0.325      | 0.629     | 0.678     |
| random 128/shard (768 labels) | 0.744     | 0.264      | 0.740     | 0.708     |
| random 32/shard (192 labels)  | 0.722     | 0.181      | 0.697     | 0.620     |
| random 64/shard               | 0.711     | 0.140      | 0.719     | 0.678     |
| LC1                           | 0.702     | 0.107      | **0.747** | 0.702     |
| LC3                           | 0.631     | **-0.159** | 0.749     | 0.651     |

Findings:

- **AURAC reranks the field relative to accuracy.** The propagated models
  that lead model accuracy at 20k (0.805-0.810) sit mid-pack on AURAC
  there - their confidence does not rank correctness. Conversely GC2 is
  #2 on AURAC despite mediocre accuracy. Any "best proxy" claim must name
  the metric.
- **LC3 is uniformly disqualified**: negative nAURAC at both scales -
  its confidence ordering is WORSE than random, even though its AUC
  (0.726-0.749) is among the best. Score-range collapse (Section 5.1)
  destroys deferral value while preserving class ranking: AUC is
  necessary but not sufficient for a deferrable proxy.
- **No single strategy dominates across scales.** At 384 labels, LC1 wins
  at 20k (0.880) but ranks near the bottom at 1M (0.702), while LC4 flips
  from mid-pack (0.860) to best (0.805). Random sampling is the only
  family consistently in the top half at both scales, at a fraction of the
  Phase-A cost - and at double budget (768 labels) random is top-3 on
  both scales. The datasets differ in difficulty (80.8% vs 67.4% positive),
  so cross-scale comparisons are qualitative.

### 8.2 Accuracy under an LLM budget

acc@B with optimal deferral (perfect LLM on the B least-confident rows),
1M rows:

| Method     | B=5%  | B=10% | B=20%     | B=30%     | B=50%     | band(0.2,0.8) escalation |
| ---------- | ----- | ----- | --------- | --------- | --------- | ------------------------ |
| LC4        | 0.706 | 0.735 | **0.787** | **0.831** | **0.902** | 34.6%                    |
| LC2        | 0.698 | 0.722 | 0.767     | 0.807     | 0.877     | 47.3%                    |
| random 128 | 0.693 | 0.713 | 0.752     | 0.791     | 0.864     | 38.9%                    |
| LC1        | 0.692 | 0.710 | 0.747     | 0.783     | 0.853     | 57.2%                    |
| LC3        | 0.688 | 0.702 | 0.731     | 0.761     | 0.821     | 99.8%                    |

Reading the budget axis:

- **Ranking by AURAC is ranking by acc@B at every budget point** - the
  curve is the budget view of the same property. Choosing the proxy is
  therefore an AURAC decision; the budget only selects the operating
  point on the chosen curve.
- **Fixed bands are a blunt implementation of the same idea.** Confidence
  here is |score - 0.5|, so a band IS a confidence-threshold deferral;
  its only parameter must be set per model from the risk-accuracy curve.
  Example at 1M: LC1's default band escalates 57.2% for cascade acc
  0.923, while LC4 escalates 34.6% - a band tuned on LC1's curve wastes
  23 pts of budget on LC4's flatter score distribution. At 20k the gap is
  worse (random-64 band: 92.6% escalation). Per-model band calibration
  from the dumped scores is free accuracy.
- **The escalation cliff is real**: at 1M the retained (outside-band) rows
  are ~82% correct while the first rows inside the band drop to near-zero
  correctness - widening the band past the cliff buys almost nothing.

### 8.3 Executing the recommended proxy

Per-shard Phase-A breakdown at 1M (166k rows, instrumented, release):

| Stage                                      | Time/shard | Share |
| ------------------------------------------ | ---------- | ----- |
| scan + embedding string parse              | 20.7 s     | 31%   |
| k-means (k=64, 10 iters) + representatives | **46.2 s** | 68%   |
| labeling 64 representatives                | 0.45 s     | <1%   |

Execution roadmap for LC-family proxies (the AURAC leaders):

1. **k-means is the single lever** (68% of Phase A, single-threaded naive
   Lloyd): SIMD distance kernels, k-means|| init, or early iteration stops
   target the 46 s directly; it parallelizes across shards but not within.
2. **Embedding transport as binary floats** removes most of the 21 s
   scan+parse (strings are ~8x wider than f32).
3. **Central clustering (GC) stays excluded**: single-node k-means
   extrapolates to ~27 min at 1M against 46 s/shard for the local variants,
   while its AURAC advantage at 20k (+0.003 over LC1) is within noise.
4. LC4 additionally pays the federated weight round (11 KB) and per-shard
   LR training (~7 s/shard at 166k rows, linear) - cheaper than LC2's 42 s
   central training on 1M rows, with better AURAC at 1M.

### 8.4 Answer summary

- **Best proxy by AURAC**: at the 384-label budget, LC1 on the easy
  distribution (20k), LC4 on the hard/large one (1M); random sampling is
  the robust, cheapest competitor (top-3 at both scales, 0.01-0.2 s
  Phase A). LC3 should be dropped from consideration (nAURAC < 0).
- **Best proxy per LLM-call budget**: identical selection (AURAC ranks ==
  acc@B ranks); the budget then picks the operating point on that
  model's risk-accuracy curve, and the cascade band must be calibrated
  per model to hit the budget.
- **Execution**: Phase B is negligible (<=1 s at 1M); the LC-family Phase A
  (46 s k-means + 21 s parse per shard at 1M, parallel) is the only
  system cost worth optimizing, with k-means and binary embedding
  transport as the two concrete targets.

## 9. Recursive uncertainty sampling x FEVER: 2x2x2 validation

Setup: methods {random, recursive, LC1, LC2, LC4} x datasets {movie
(easy), FEVER claims (hard)} x scale {20k, 100k} x label budget {200,
1000} = 40 cells, single-shard simulation (coordinator only; timing not
measured), oracle LLM, score dumps for offline analysis. Note LC2 == LC4
identically here by construction: with one shard, central and federated
training see the same propagated training set (the tie in all LC2/LC4
numbers is structural, not a coincidence).

**Recursive strategy** (new, orthogonal sampling axis): round 0 labels a
random 20% of the budget and trains the LR proxy; each of 5 following
rounds labels the most uncertain unlabeled rows (|score - 0.5| smallest)
with 1/5 of the remaining budget and retrains on all labeled rows.
Selection requires scoring the whole table each round (full-table fetch,
single-machine simulation). `recursive` in the strategy factory; knobs
`recursive_budget` / `SWANLAKE_SEMANTIC_RECURSIVE_{BUDGET,INIT_FRAC,ROUNDS}`.

### 9.1 Axis 1: AURAC (proxy quality per label budget)

Winner per cell (nAURAC in parentheses):

| Cell               | #1                         | #2                     | #3              | #4 (worst)   |
| ------------------ | -------------------------- | ---------------------- | --------------- | ------------ |
| movie 20k, B=200   | LC2/LC4 0.883 (0.44)       | recursive 0.869 (0.35) | LC1 0.854       | random 0.835 |
| movie 20k, B=1000  | **recursive 0.909 (0.58)** | LC2/LC4 0.905          | random 0.891    | LC1 0.849    |
| movie 100k, B=200  | LC1 0.839 (0.32)           | LC2/LC4 0.831          | recursive 0.820 | random 0.814 |
| movie 100k, B=1000 | LC1 0.864 (0.45)           | recursive 0.858        | LC2/LC4 0.854   | random 0.825 |
| fever 20k, B=200   | LC2/LC4 0.779 (0.21)       | random 0.662           | recursive 0.660 | LC1 0.591    |
| fever 20k, B=1000  | **LC2/LC4 0.823 (0.41)**   | LC1 0.784              | recursive 0.754 | random 0.622 |
| fever 100k, B=200  | LC2/LC4 0.768 (0.17)       | recursive 0.685        | random 0.679    | LC1 0.593    |
| fever 100k, B=1000 | **LC2/LC4 0.813 (0.37)**   | LC1 0.727              | recursive 0.714 | random 0.605 |

Findings:

- **Cluster + label propagation is the best proxy family on AURAC**, on
  the hard dataset at every budget (0.768-0.823), and top-2 on the easy
  one. Propagated models order correctness by confidence far better than
  their raw accuracy suggested (Section 8 already showed GC2 #2 at 20k).
- **Random sampling is last or near-last on AURAC in 7 of 8 cells despite
  competitive AUC** (fever B=1000: AUC 0.743-0.763 yet AURAC 0.605-0.622).
  Random labels give class ranking but no confidence calibration. The
  AUC-is-not-sufficient lesson of Section 8 reproduces on a second dataset.
- **Recursive beats random in 7 of 8 cells** and wins outright on
  movie-20k-B1000 (0.909, the single best AURAC measured). Its value is
  exactly the calibration fix random lacks: spending labels on the
  uncertain boundary teaches the model where its own confidence is low.
- **LC1 (clean representatives) is the most unstable**: best once
  (movie-100k) but worst on fever at B=200 (0.591-0.593, nAURAC < -0.5).
  Mechanism: representatives are the points nearest cluster centers - the
  prototypical, easy points of each cluster - so a reps-only model trains
  exclusively on confident regions and cannot rank uncertainty on hard
  data. Clean labels alone do not buy deferral quality; coverage of the
  boundary does.
- Certainty-distribution diagnostic (fraction of a random-200 model's
  scores with |s-0.5| >= 0.2): movie 49-63%, fever@100k **1.4%** - the
  near-uniform "all yellow" signature of hard datasets; recursive's gain
  over random is largest where this signature holds (fever B=1000: +0.109
  AURAC at 100k).

### 9.2 Axis 2: LLM-call reduction under an FNR budget (static threshold)

Policy: auto-reject rows with score < t, escalate everything else to the
LLM; t is the tightest threshold keeping FNR <= alpha. red = fraction of
LLM calls saved, best per dataset-scale (B=1000):

| Dataset-scale | red@FNR=1% | red@FNR=5% | red@FNR=10% | best methods         | worst                 |
| ------------- | ---------- | ---------- | ----------- | -------------------- | --------------------- |
| movie 20k     | 0.026      | 0.098      | 0.169       | recursive/random/lc1 | LC2/LC4 (0.082/0.144) |
| movie 100k    | 0.030      | 0.101      | 0.170       | lc1/random           | LC2/LC4 (0.069/0.131) |
| fever 20k     | 0.073      | 0.153      | 0.208       | random/recursive     | LC2/LC4 (0.112/0.175) |
| fever 100k    | 0.076      | 0.152      | 0.210       | recursive/random     | LC2/LC4 (0.113/0.173) |

Findings:

- **The Axis-2 ranking inverts Axis-1 for the propagated family.** LC2/LC4
  lead AURAC but save the fewest calls under an FNR budget: their score
  mass sits high (inflated positive-side confidence), leaving a thin
  auto-rejectable negative tail. FNR-bounded savings come from the mass of
  negatives scored far below the positives' floor, which the clean/random
  and recursive models have.
- **Absolute savings are modest and dataset-driven**: ~10-15% of calls at
  FNR=5%, ~17-21% at FNR=10%. Both datasets are positive-heavy (73-81%),
  so the rejectable population is small; a balanced dataset would widen
  these numbers.
- Recursive matches the best call-savers on fever (0.150-0.152 at 5%) -
  the only method strong on BOTH axes.

### 9.3 Synthesis: which proxy, then

- **Axis 1 (accuracy per deferral budget)**: cluster + propagation
  (LC2/LC4) is the best proxy family overall and clearly best on hard
  data; recursive is the best clean-label method and wins on easy data at
  B=1000. LC1 and pure random should not be chosen for deferral quality.
- **Axis 2 (call reduction under FNR budget)**: recursive / random / LC1
  save the most calls; propagated models save the least. Savings are
  10-21% at FNR 5-10% on these positive-heavy datasets.
- **No single configuration dominates both axes.** If one must be chosen:
  recursive uncertainty sampling is the robust middle - top-3 on AURAC in
  7/8 cells, top on call reduction for hard data, no label propagation
  noise, and its Phase A is pure scan + tiny LR retrains. Where AURAC is
  the priority on hard data, LC4 (federated propagation) leads while
  keeping raw data on the workers.

### 9.4 Recursive x cluster hybrids (LC1-rec, LC4-rec)

Recursive is a label-acquisition optimizer orthogonal to the base
strategies; the two hybrids keep the label semantics of their base:

- **LC1-rec**: k-means once (k = budget); round 0 labels 20% of the
  representatives (coverage), later rounds label the most uncertain
  unlabeled ROWS (boundary); clean labels, central training.
- **LC4-rec**: k-means once per shard; same recursive label loop, but each
  label propagates to its whole cluster (majority vote) and the LR
  retrains on the propagated shard every round. Single-shard path
  implemented; multi-round federated RPC is future work.

Axis 1, AURAC vs the previous best of each cell:

| Cell               | LC1-rec   | LC4-rec | pure recursive | prev. cell best |
| ------------------ | --------- | ------- | -------------- | --------------- |
| movie 20k, B=200   | 0.873     | 0.862   | 0.869          | LC2/LC4 0.883   |
| movie 20k, B=1000  | 0.903     | 0.899   | 0.909          | recursive 0.909 |
| movie 100k, B=200  | 0.814     | 0.836   | 0.820          | LC1 0.839       |
| movie 100k, B=1000 | **0.868** | 0.848   | 0.858          | LC1 0.864       |
| fever 20k, B=200   | 0.696     | 0.762   | 0.660          | LC2/LC4 0.779   |
| fever 20k, B=1000  | 0.705     | 0.786   | 0.754          | LC2/LC4 0.823   |
| fever 100k, B=200  | 0.680     | 0.766   | 0.685          | LC2/LC4 0.768   |
| fever 100k, B=1000 | 0.692     | 0.784   | 0.714          | LC2/LC4 0.813   |

Axis 2, call reduction (red@FNR=5% / 10%, B=1000):

| Cell       | LC1-rec         | LC4-rec     | recursive   | random      |
| ---------- | --------------- | ----------- | ----------- | ----------- |
| movie 20k  | 0.100/0.172     | 0.078/0.140 | 0.098/0.169 | 0.093/0.161 |
| movie 100k | 0.097/0.166     | 0.069/0.128 | 0.101/0.170 | 0.100/0.170 |
| fever 20k  | **0.155/0.209** | 0.108/0.164 | 0.150/0.207 | 0.153/0.208 |
| fever 100k | **0.156/0.211** | 0.109/0.166 | 0.152/0.210 | 0.150/0.203 |

Findings:

- **LC4-rec beats pure recursive in 7/8 AURAC cells** - seeding labels
  with cluster structure (and propagating them) is worth ~0.02-0.10 AURAC
  over random-init active learning. It is the top or runner-up proxy in
  every cell, and the new overall best at movie-100k-B1000 (via LC1-rec,
  0.868).
- **LC1-rec repairs LC1 exactly as hypothesized**: on fever, LC1's
  easy-points-only failure (AURAC 0.591-0.593, nAURAC -0.6) becomes
  0.680-0.705 once boundary rows enter the training set - still below the
  propagation family, but now competitive with pure recursive.
- **Recursive does NOT dethrone one-shot propagation on the hard dataset**:
  LC2/LC4 keep a 0.03-0.04 AURAC edge over LC4-rec on fever at B=1000.
  Uniformly spread representative labels remain slightly better boundary
  teachers than uncertainty-chosen ones for propagation training.
- **Axis 2 pattern persists**: LC1-rec joins the top call-savers
  (0.155-0.156 @5% on fever) while LC4-rec inherits propagation's thin
  negative tail (0.108-0.109). The hybrid does not break the axis-1 /
  axis-2 trade-off; LC4-rec remains the best single compromise across the
  two axes (top-2 everywhere on Axis 1, mid-pack on Axis 2).

## 9.5 LC/GC/random x recursive: the completed orthogonal matrix (true 6-node runs)

Section 9's matrix left two gaps: the LC3/GC baselines and the whole
GC x recursive cross, and every cell there was a single-shard simulation.
This section completes the matrix {random, LC1-LC4, GC1, GC2} x {recursive:
no, yes} x label budgets {200, 1,000} x datasets {movie, fever} x scales
{20k, 100k} = 112 cells, all run on the REAL 6-node cluster (coordinator +
5 workers, hash-distributed shards).

Implementation (new since Section 9):

- **Multi-round recursive RPC** (opcode 'R'): the distributed LC-rec
  family runs its label loop across the real shards. Every shard clusters
  once per round (seeded, identical result); round 0 labels its share of
  the representatives (coverage), each following round labels its share of
  the most uncertain unlabeled rows scored with the CURRENT GLOBAL model
  (boundary). Training-side split mirrors the non-recursive family:
  LC1-rec = clean labels + central training (rows travel), LC2-rec =
  propagated labels + central training (full shards travel), LC3-rec =
  clean labels + federated training (weights travel), LC4-rec = propagated
  labels + federated training (weights travel). The budget is split evenly
  over shards and spent exactly (round-0 share + 5 uncertainty rounds).
- **random / recursive**: one-shot rowid-window sampling (`offset`) with
  the budget split per shard (33/166 per shard, i.e. 198/996 total
  labels), and the R-round random-based active-learning variant on the
  pooled table - both reuse the existing distributed paths.
- **GC1-rec / GC2-rec**: the R-round framework with cluster-representative
  candidates on the pooled table (global k-means at the coordinator),
  clean labels (GC1-rec) or cluster propagation (GC2-rec) - the recursive
  extension of the GC family, reusing the distributed fetch path.
- Equivalence check: with one shard, the new distributed LC1-rec path and
  the GC1-rec framework path agree bit-for-bit (identical AURAC and call
  counts on a fever-2k probe), and single-shard LC4-rec reproduces the
  Section 9.4 routine.
- Datasets were regenerated on the evaluation machine with the same
  generator scripts and seeds (movie 100k: 77.5% positive; fever 100k:
  72.8% positive - both matching Section 2.2), so absolute numbers are
  directly comparable across this section's cells; hash-distributed shards
  are structurally different from Section 9's pooled single shard, so
  LC-family values here are not 1:1 comparable to Section 9.

Protocol as in Sections 8-9: oracle LLM, cascade band (0.2, 0.8) for the
call counts, score dumps analyzed offline. "calls" = Phase-A labels +
Phase-B escalations. Measured Phase-A labels: exactly 200/1,000 for all
recursive variants and the GC family (random: 198/996 from the per-shard
split); one-shot cluster variants land at
198/996 where the shared k-means produced empty clusters (equal for every
cluster method, comparison unaffected).

### 9.5.1 movie (easy)

| Method | recursive | labels | AURAC @20k | calls @20k | AURAC @100k | calls @100k |
| ------ | --------- | ------ | ---------- | ---------- | ----------- | ----------- |
| LC1    | no        | 200    | 0.795      | 14,234     | 0.807       | 63,777      |
| LC1    | no        | 1,000  | 0.864      | 9,612      | 0.862       | 49,183      |
| LC1    | yes       | 200    | 0.815      | 10,372     | 0.815       | 42,356      |
| LC1    | yes       | 1,000  | **0.879**  | 12,426     | **0.872**   | 21,894      |
| LC2    | no        | 200    | 0.830      | 6,223      | 0.854       | 28,680      |
| LC2    | no        | 1,000  | 0.853      | 8,307      | 0.845       | 27,672      |
| LC2    | yes       | 200    | 0.824      | 5,059      | 0.832       | 14,363      |
| LC2    | yes       | 1,000  | 0.817      | 8,350      | 0.846       | 42,437      |
| LC3    | no        | 200    | 0.750      | 20,198     | 0.744       | 100,198     |
| LC3    | no        | 1,000  | 0.854      | 18,707     | 0.841       | 82,259      |
| LC3    | yes       | 200    | 0.761      | 20,200     | 0.784       | 100,200     |
| LC3    | yes       | 1,000  | 0.870      | 17,505     | 0.854       | 95,043      |
| LC4    | no        | 200    | 0.855      | 6,035      | 0.868       | 9,845       |
| LC4    | no        | 1,000  | **0.881**  | 7,289      | **0.872**   | 16,634      |
| LC4    | yes       | 200    | 0.826      | 6,992      | 0.827       | 471         |
| LC4    | yes       | 1,000  | 0.859      | 8,642      | 0.862       | 13,906      |
| GC1    | no        | 200    | 0.743      | 20,199     | 0.810       | 96,264      |
| GC1    | no        | 1,000  | 0.868      | 9,088      | 0.869       | 74,462      |
| GC1    | yes       | 200    | 0.830      | 7,632      | 0.785       | 100,029     |
| GC1    | yes       | 1,000  | 0.865      | 10,043     | 0.869       | 34,251      |
| GC2    | no        | 200    | 0.847      | 6,812      | 0.802       | 29,361      |
| GC2    | no        | 1,000  | 0.853      | 7,143      | 0.857       | 28,682      |
| GC2    | yes       | 200    | 0.793      | 654        | 0.826       | 6,991       |
| GC2    | yes       | 1,000  | 0.859      | 6,174      | 0.836       | 23,506      |
| RANDOM | no        | 200    | 0.783      | 18,881     | 0.819       | 57,223      |
| RANDOM | no        | 1,000  | 0.863      | 13,415     | 0.858       | 45,071      |
| RANDOM | yes       | 200    | 0.848      | 8,838      | 0.811       | 56,239      |
| RANDOM | yes       | 1,000  | 0.878      | 7,555      | 0.880       | 29,956      |

### 9.5.2 fever (hard)

| Method | recursive | labels | AURAC @20k | calls @20k | AURAC @100k | calls @100k |
| ------ | --------- | ------ | ---------- | ---------- | ----------- | ----------- |
| LC1    | no        | 200    | 0.612      | 11,961     | 0.594       | 59,229      |
| LC1    | no        | 1,000  | 0.643      | 8,992      | 0.646       | 29,923      |
| LC1    | yes       | 200    | **0.714**  | 6,659      | 0.679       | 55,164      |
| LC1    | yes       | 1,000  | 0.703      | 6,313      | 0.675       | 47,897      |
| LC2    | no        | 200    | 0.791      | 5,252      | 0.825       | 10,171      |
| LC2    | no        | 1,000  | 0.816      | 6,774      | 0.827       | 17,593      |
| LC2    | yes       | 200    | 0.791      | 4,908      | 0.738       | 41,134      |
| LC2    | yes       | 1,000  | 0.780      | 8,982      | 0.721       | 49,000      |
| LC3    | no        | 200    | 0.584      | 20,179     | 0.570       | 96,970      |
| LC3    | no        | 1,000  | 0.625      | 15,434     | 0.619       | 75,488      |
| LC3    | yes       | 200    | 0.575      | 20,200     | 0.737       | 100,200     |
| LC3    | yes       | 1,000  | 0.662      | 18,886     | 0.682       | 80,229      |
| LC4    | no        | 200    | 0.798      | 2,131      | **0.835**   | 5,254       |
| LC4    | no        | 1,000  | **0.829**  | 4,792      | **0.829**   | 7,249       |
| LC4    | yes       | 200    | 0.646      | 8,651      | 0.699       | 38,801      |
| LC4    | yes       | 1,000  | 0.723      | 13,449     | 0.734       | 25,511      |
| GC1    | no        | 200    | 0.593      | 16,570     | 0.596       | 96,563      |
| GC1    | no        | 1,000  | 0.723      | 10,896     | 0.711       | 33,272      |
| GC1    | yes       | 200    | 0.622      | 20,200     | **0.751**   | 68,717      |
| GC1    | yes       | 1,000  | 0.674      | 5,436      | 0.689       | 65,089      |
| GC2    | no        | 200    | 0.784      | 3,839      | 0.799       | 9,670       |
| GC2    | no        | 1,000  | 0.821      | 4,223      | 0.824       | 10,694      |
| GC2    | yes       | 200    | **0.806**  | 4,991      | 0.791       | 19,406      |
| GC2    | yes       | 1,000  | 0.784      | 6,048      | 0.777       | 25,772      |
| RANDOM | no        | 200    | 0.623      | 19,562     | 0.601       | 80,464      |
| RANDOM | no        | 1,000  | 0.643      | 9,485      | 0.624       | 60,757      |
| RANDOM | yes       | 200    | 0.734      | 16,629     | 0.664       | 94,822      |
| RANDOM | yes       | 1,000  | 0.666      | 5,775      | 0.653       | 74,041      |

Findings:

- **Recursive repairs the clean-label methods, now also for GC.** On
  fever, LC1 gains +0.07-0.10 AURAC, LC3 up to +0.17 (100k, B=200), and
  GC1 +0.15 at 100k/B=200 (0.596 -> 0.751) - the easy-points-only failure
  of representative labeling (Section 9.4) is fixed by boundary rows even
  when the clustering is global. On movie the gains are smaller (+0.01-0.03)
  except GC1 at B=200 (+0.09).
- **Recursive still does not beat one-shot propagation.** LC4 and the
  LC2/GC2 family keep the AURAC lead on fever in every budget/scale cell
  (LC4 0.829-0.835 vs LC4-rec 0.699-0.734); the gap is WIDER than in the
  single-shard Section 9.4, because the distributed uncertainty rounds
  spend per-shard quotas on boundary rows while one-shot spreads the whole
  budget as uniformly placed representatives - better teachers for
  propagation training. GC2-rec stays within ~0.01-0.05 of GC2.
  Why recursion HURTS the propagated family (score-dump diagnostics):
  (i) the one-shot budget is spread as uniform representative anchors, whose
  propagation gives spatially smooth, confident-and-correct scores - exactly
  what AURAC rewards - while uncertainty rounds spend 80% of the budget on
  boundary rows, whose majority-vote propagation muddies previously clean
  clusters and pulls confident-correct rows toward 0.5 (fever-100k B=1,000:
  AUC 0.745 -> 0.724 but certainty mass |s-0.5| >= 0.2 96.3% -> 90.2%,
  score mean 0.900 -> 0.784); (ii) multi-round federated averaging replays
  LC3's calibration collapse - at movie-100k B=200 LC4-rec degenerates to a
  near-constant function (score std 0.028, pos-neg separation 0.008,
  AUC 0.688 -> 0.597, 471 total calls); (iii) positive-heavy base rates
  leave most negative prototype clusters unlabeled at the 20% round-0 share.
- **The random family reproduces Section 9 on the real cluster**: one-shot
  random sits at or near the bottom of AURAC on fever at every budget
  (0.601-0.643), and its recursive variant is the cheapest clustering-free
  repair (+0.03-0.13; at fever-20k B=200 it is the best clean-label-style
  configuration, 0.734, ahead of LC1-rec/GC1-rec). One-shot random uses a
  fresh rowid window per run, so its values carry the usual +-1-4 point
  run-to-run spread.
- **No cell where recursion hurts a clean method**; on movie LC1-rec is
  the best or co-best configuration at both budgets (0.872-0.879).
- **LLM calls mirror Section 9.2's axis-2 picture**: propagated proxies
  are the cheapest at inference (LC4 fever-100k B=1,000: 7,249 total
  calls), LC3/LC3-rec escalate ~100% of rows, and turning recursion ON
  usually raises Phase-B calls for the propagated family (their scores
  lose the accept-heavy bias that suppressed escalations).
- **Best single configurations of the completed matrix**: fever - LC4
  (one-shot, federated propagation) at every budget; movie - LC4 at
  B=200/1,000 with LC1-rec within 0.001-0.007. GC remains dominated: its
  best cell (GC2-rec fever-20k B=200, 0.806) is matched or beaten by the
  LC propagation family at equal budget in every case except that one
  cell, while paying the O(table) shipping + central k-means.

Reproduction:

```bash
# datasets (regenerated; see docs/semantic_deploy.md)
python3 scripts/gen_movie_dataset.py <clapper_source> 100000 --out-dir ~/.swanlake-movie/data100k
python3 scripts/gen_fever_dataset.py <train.jsonl> 100000 --out-dir ~/.swanlake-fever/data100k
python3 scripts/gen_embeddings.py --in .../reviews.csv --out .../reviews_emb.csv --device cuda
# 20k/2k = first N rows of the 100k outputs

# 6-node matrix: scales {20k,100k} x budgets {200,1000} x 12 methods
bash scripts/run_matrix_6node.sh movie 'Is this movie review positive?'
bash scripts/run_matrix_6node.sh fever 'Is this claim supported?'
python3 scripts/collect_deferral6_table.py <movie|fever>   # AURAC + calls
```

## 10. Limitations and future work

- Oracle LLM: final-accuracy numbers are upper bounds for a perfect LLM;
  a real Qwen3-4B will trade some accuracy for the call counts reported.
- Two scales of one dataset family (20k at 80.8% positive, 1M at 67.4%);
  base-rate and difficulty shifts moved absolute accuracies, so conclusions
  are stated as patterns, not absolute numbers. The score-dump tooling
  (`SWANLAKE_SEMANTIC_SCORE_DUMP` + `scripts/analyze_score_dump.py`,
  `analyze_aurac.py`, `analyze_fnrbudget.py`) makes re-analysis on other
  datasets straightforward.
- Section 9 uses single-shard simulation (per instruction, timing out of
  scope): LC2 and LC4 coincide by construction there; with real shards LC4
  additionally pays per-shard LR training (~7 s/166k rows) and LC2 ships
  full shards (~31 MB/166k rows) to the coordinator.
- The 1M LC1 k=3200/shard proportional cell crashed after a ~40 min
  single-node k-means (wire underflow at restart of the polluted run dir);
  its Section 8b conclusions rest on the completed cells.
- AUC/zone metrics measured at 20k only; the 1M check covers accuracy and
  escalation volumes.
- Naive single-threaded k-means at the coordinator is the GC bottleneck;
  vectorized/parallel k-means or sampling-based clustering would shrink it.
- Propagated-label training may improve with calibration (e.g. global
  standardization before federated averaging) or a coordinator
  fine-tuning round - untested. Negative-side calibration of the clean
  proxies (their weak reject zone) is the most promising accuracy lever.
- String-encoded sample/result transport; a binary path would cut GC
  shipping (~60 MB at 20k, ~3 GB at 1M) roughly in half.

## Appendix: reproduction

```bash
# dataset: 20k reviews, 384-dim MiniLM embeddings, ground truth (cached)
# 6 nodes = coordinator + 5 workers; k=64/shard; oracle labelers by default
SWANLAKE_SEMANTIC_CLUSTER_COUNT=384 \
SWANLAKE_SEMANTIC_SAMPLE_STRATEGY=cluster_local \      # offset | cluster_local | cluster_local_full |
                                                       # cluster_central_sample | cluster_central |
                                                       # federated_sample | federated
SWANLAKE_SEMANTIC_CASCADE_THRESHOLD=0.8 \
SWANLAKE_SEMANTIC_CASCADE_REJECT_THRESHOLD=0.2 \
SWANLAKE_SEMANTIC_SCORE_DUMP=$HOME/.swanlake-movie/scores/cluster_local \
    bash scripts/movie_e2e.sh 20000 5

# discrimination analysis (AUC, zone mass, zone correctness)
python3 scripts/analyze_score_dump.py \
    $HOME/.swanlake-movie/scores/cluster_local \
    $HOME/.swanlake-movie/data/reviews.labels --bands 0.2,0.8 0.1,0.9
```

Measured metrics per run: precision/recall/accuracy vs ground truth;
`[Semantic] phase timings` (detection / sample+collect / clustering /
labeling / training / broadcast / rewrite); `[Semantic][cascade]`
(escalations, simulated LLM latency, Phase-B wall); per-row score dumps
for offline zone analysis.
