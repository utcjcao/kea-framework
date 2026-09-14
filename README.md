Kea trains a logistic-regression proxy from labeled examples selected by an
extensible sampler. A single `ProxyTrainingRunner` owns the training loop;
single-machine execution is the currently implemented deployment mode.

There are two developer-provided extension points:

- `ISampler`: how to choose the initial points used to train the model;
- `ILabeler`: how selected candidates receive binary labels.

`RunConfig` specifies shared run parameters such as the dataset, total label
budget, number of rounds, random seed, label semantics, and eventual
execution/training placement.

## Configuring a training run

The caller configures the source table, creates a sampler with any
sampler-specific options, and sets the number of total training rounds.

```cpp
#include "kea/function_labeler.h"
#include "kea/proxy_training.h"
#include "kea/run_config.h"
#include "kea/samplers/random_sampler.h"

duckdb::Connection connection(database);

kea::RunConfig config;
config.dataset.table_name = "documents";
config.dataset.id_column = "id";
config.dataset.text_column = "text";
config.dataset.embedding_column = "embedding";
config.rounds = 5;                 // 1 = one-shot; >1 = recursive uncertainty rounds
config.label_budget = 150;         // total labels across every round
config.initial_label_fraction = 0.2;
config.seed = 42;

kea::RandomSampler sampler;
kea::FunctionLabeler labeler([](const kea::Candidate& candidate) {
  return /* return the binary label for candidate */ 0;
});

kea::ProxyTrainingRunner runner(connection);
kea::ProxyModel model = runner.Run(config, sampler, labeler);
// model.weights and model.intercept
```

To use centralized cluster-representative sampling instead, construct a
`ClusterSampler` and pass it to the same `Run()` call:

```cpp
#include "kea/samplers/cluster_sampler.h"

kea::ClusterSamplingOptions cluster_options;
cluster_options.cluster_count = 50;
cluster_options.max_iterations = 20;
kea::ClusterSampler sampler(cluster_options);
```

The combinations map to the supported strategies as follows:

| Initial sampler | `rounds` | Behavior |
| --- | ---: | --- |
| `RandomSampler` | `1` | one-shot random sampling (`offset`) |
| `RandomSampler` | `> 1` | recursive uncertainty sampling (`recursive`) |
| `ClusterSampler` | `1` | centralized cluster representatives (`cluster_central_sample` / GC1) |
| `ClusterSampler` | `> 1` | cluster representatives followed by uncertainty sampling (`gc1rec`) |

The public C++ API is declared in `include/kea/proxy_training.h` and
`include/kea/run_config.h`. It keeps DuckDB as the data store, exposes initial
sampling and labeling extension points, and leaves recursive uncertainty
sampling under the runner's control.

`include/kea/samplers/random_sampler.h` provides the first sampler implementation. It
requests a seed-stable random initial batch; the DuckDB sampling context owns
the hash-order query used to perform it.

`src/logistic_regression_trainer.cc` adapts mlpack's binary, L2-regularized
logistic regression and returns the learned intercept and weights. Its C++17
unit test is in `tests/logistic_regression_trainer_test.cc`.

`ProxyTrainingRunner` implements the complete single-machine MVP flow:
initial sampling, labeling, training, recursive uncertainty sampling, and
retraining. It returns the final `ProxyModel` directly. The in-memory DuckDB
end-to-end test is in `tests/proxy_training_runner_test.cc`. Samplers request
operations through a framework-created sampling context; only the internal
DuckDB data-access layer constructs SQL.

The transport-neutral execution backend and shard-worker protocol are internal
implementation details. `ProxyTrainingRunner` uses the single-machine backend
to dispatch to one internal DuckDB worker. A real multi-machine backend
remains pending.

## Not yet supported

The centralized clean-label strategies are supported: random one-shot
(`offset`), random recursive (`recursive`), centralized cluster
representatives (`cluster_central_sample` / GC1), and its recursive form
(`gc1rec`).

The remaining strategies are intentionally not implemented yet:

- `cluster_central` (GC2) and `gc2rec` need a label-amplification policy that
  propagates a representative's label to its cluster members. This is a
  separate responsibility from sampling and should be introduced as its own
  extension point.
- `cluster_local`, `cluster_local_full`, `federated_sample`, `federated`, and
  `lc1rec` through `lc4rec` require multiple real shards. They depend on
  worker dispatch/RPC, shard-local data ownership, and—in federated modes—
  exchanging and aggregating independently trained model weights.

There is an internal in-process multi-shard LC1 correctness test: independent
DuckDB shards perform local cluster-representative sampling, their clean labels
are pooled for central training, and the resulting model is broadcast after
each round. It is not a networked distributed runtime.

For this MVP, the configured DuckDB table must provide a `VARCHAR` ID column,
a `VARCHAR` text column, and a `FLOAT[]` embedding column. `FunctionLabeler`
is the included callback-backed labeler for tests or existing labeling systems.

Build dependencies are DuckDB, mlpack, Armadillo, cereal, and ensmallen. Once
installed, configure and test with `cmake -S . -B build && cmake --build build
&& ctest --test-dir build`.
