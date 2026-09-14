#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kea/proxy_training.h"

namespace kea {

enum class ExecutionMode {
  SingleMachine,
  Distributed,
};

enum class LabelMode {
  Clean,
  Propagated,
};

enum class TrainingPlacement {
  Central,
  Federated,
};

struct WorkerEndpoint {
  std::string worker_id;
  std::string address;
};

struct RunConfig {
  ExecutionMode execution_mode = ExecutionMode::SingleMachine;
  LabelMode label_mode = LabelMode::Clean;
  TrainingPlacement training_placement = TrainingPlacement::Central;

  TrainingDataset dataset;
  std::size_t rounds = 1;
  std::size_t label_budget = 200;
  double initial_label_fraction = 0.2;
  std::uint64_t seed = 42;

  std::vector<WorkerEndpoint> workers;
};

void ValidateRunConfig(const RunConfig& config);

}  // namespace kea
