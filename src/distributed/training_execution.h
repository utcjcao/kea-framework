#pragma once

#include "distributed/execution_backend.h"
#include "kea/run_config.h"

namespace kea::distributed::detail {

// Shared clean-label, central-training rounds loop. Both the one-shard local
// backend and the multi-shard test backend use this implementation.
[[nodiscard]] ProxyModel RunCleanCentralTraining(
    const RunConfig& config,
    ITrainingExecutionBackend& backend,
    const LogisticRegressionTrainer& trainer);

}  // namespace kea::distributed::detail
