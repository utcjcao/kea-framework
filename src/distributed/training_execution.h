#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "distributed/execution_backend.h"
#include "kea/run_config.h"

namespace kea::distributed::detail {

struct TrainingRoundTiming {
  std::size_t round_index = 0;
  std::size_t requested_labels = 0;
  std::size_t acquired_labels = 0;
  std::uint64_t sampling_us = 0;
  std::uint64_t fetching_us = 0;
  std::uint64_t acquisition_us = 0;
  std::uint64_t training_us = 0;
};

// Optional instrumentation for benchmarks. Acquisition includes sampling,
// DuckDB row access, and labeling; callers that need finer attribution may
// time their sampler and labeler implementations independently.
struct TrainingExecutionTiming {
  std::vector<TrainingRoundTiming> rounds;
  std::uint64_t total_us = 0;
};

// Shared central-training rounds loop. For clean labels it trains on direct
// labels; for propagated labels it trains on each worker's current expanded
// cluster-labeled dataset.
[[nodiscard]] ProxyModel RunCentralTraining(
    const RunConfig& config,
    ITrainingExecutionBackend& backend,
    const LogisticRegressionTrainer& trainer,
    TrainingExecutionTiming* timing = nullptr);

}  // namespace kea::distributed::detail
