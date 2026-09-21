#include "kea/proxy_training.h"

#include <memory>

#include "duckdb.hpp"

#include "distributed/duckdb_shard_worker.h"
#include "distributed/execution_backend.h"
#include "distributed/training_execution.h"
#include "kea/run_config.h"

namespace kea {
ProxyTrainingRunner::ProxyTrainingRunner(duckdb::Connection& connection)
    : connection_(connection) {}

ProxyModel ProxyTrainingRunner::Run(
    const RunConfig& config,
    ISampler& sampler,
    ILabeler& labeler) {
  if (config.execution_mode != ExecutionMode::SingleMachine) {
    throw std::logic_error(
        "Distributed execution is not connected to ProxyTrainingRunner yet");
  }
  auto worker = std::make_shared<distributed::detail::DuckDbShardWorker>(
      "local", connection_, sampler, labeler);
  distributed::SingleMachineBackend backend(worker);
  return distributed::detail::RunCentralTraining(config, backend, trainer_);
}

}  // namespace kea
