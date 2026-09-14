#include "kea/run_config.h"

#include <stdexcept>
#include <unordered_set>

namespace kea {

void ValidateRunConfig(const RunConfig& config) {
  if (config.dataset.table_name.empty()) {
    throw std::invalid_argument("RunConfig requires a dataset table name");
  }
  if (config.rounds == 0 || config.label_budget == 0) {
    throw std::invalid_argument("RunConfig rounds and label_budget must be positive");
  }
  if (config.initial_label_fraction <= 0.0 || config.initial_label_fraction > 1.0) {
    throw std::invalid_argument("RunConfig initial_label_fraction must be in (0, 1]");
  }

  const bool distributed = config.execution_mode == ExecutionMode::Distributed;
  if (!distributed && !config.workers.empty()) {
    throw std::invalid_argument("Single-machine runs must not specify workers");
  }
  if (distributed && config.workers.empty()) {
    throw std::invalid_argument("Distributed runs require at least one worker");
  }
  if (!distributed && config.training_placement == TrainingPlacement::Federated) {
    throw std::invalid_argument("Federated training requires distributed execution");
  }

  std::unordered_set<std::string> worker_ids;
  for (const WorkerEndpoint& worker : config.workers) {
    if (worker.worker_id.empty() || worker.address.empty()) {
      throw std::invalid_argument("Each worker requires an ID and address");
    }
    if (!worker_ids.insert(worker.worker_id).second) {
      throw std::invalid_argument("Worker IDs must be unique");
    }
  }
}

}  // namespace kea
