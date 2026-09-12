This is the single machine implementation of choosing a model and the number of recursive rounds.

there are two main parameters that the user can change:

- the sampling type: which method we use to select points to train our model with
- the training type: the number of rounds we train our model with

## Configuring a training run

The caller configures the source table, creates a sampler with any
sampler-specific options, and sets the number of total training rounds.

```cpp
#include "kea/function_labeler.h"
#include "kea/proxy_training.h"
#include "kea/samplers/random_sampler.h"

duckdb::Connection connection(database);

kea::TrainingDataset dataset;
dataset.table_name = "documents";
dataset.id_column = "id";
dataset.text_column = "text";
dataset.embedding_column = "embedding";

kea::TrainingConfig config;
config.rounds = 5;                 // 1 = one-shot; >1 = recursive uncertainty rounds
config.initial_batch_size = 50;    // labels acquired before the first model fit
config.batch_size_per_round = 25;  // labels acquired in each later round
config.seed = 42;

kea::RandomSampler sampler;
kea::FunctionLabeler labeler([](const kea::Candidate& candidate) {
  return /* return the binary label for candidate */ 0;
});

kea::ProxyTrainingRunner runner(connection);
kea::TrainingResult result = runner.Run(dataset, config, sampler, labeler);
// result.final_model.weights and result.final_model.intercept
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

The initial C++ API is declared in `include/kea/proxy_training.h`. It keeps
DuckDB as the data store, exposes an initial-sampling and labeling extension
point, and leaves recursive uncertainty sampling under the runner's control.

`include/kea/samplers/random_sampler.h` provides the first sampler implementation. It
requests a seed-stable random initial batch; the DuckDB sampling context owns
the hash-order query used to perform it.

`src/logistic_regression_trainer.cc` adapts mlpack's binary, L2-regularized
logistic regression and returns the learned intercept and weights. Its C++17
unit test is in `tests/logistic_regression_trainer_test.cc`.

`ProxyTrainingRunner` implements the complete MVP flow: initial sampling,
labeling, training, recursive uncertainty sampling, and retraining. The
in-memory DuckDB end-to-end test is in `tests/proxy_training_runner_test.cc`.
Samplers request operations through a framework-created sampling context; only
the internal DuckDB data-access layer constructs SQL.

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
  exchanging and aggregating independently trained model weights. A threaded
  single-machine imitation would not establish those future contracts.

For this MVP, the configured DuckDB table must provide a `VARCHAR` ID column,
a `VARCHAR` text column, and a `FLOAT[]` embedding column. `FunctionLabeler`
is the included callback-backed labeler for tests or existing labeling systems.

Build dependencies are DuckDB, mlpack, Armadillo, cereal, and ensmallen. Once
installed, configure and test with `cmake -S . -B build && cmake --build build
&& ctest --test-dir build`.
