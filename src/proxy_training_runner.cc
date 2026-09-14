#include "kea/proxy_training.h"

#include <algorithm>
#include <stdexcept>
#include <memory>
#include <utility>
#include <vector>

#include "duckdb.hpp"

#include "distributed/duckdb_shard_worker.h"
#include "distributed/execution_backend.h"
#include "kea/run_config.h"

namespace kea {
ProxyTrainingRunner::ProxyTrainingRunner(duckdb::Connection& connection)
    : connection_(connection) {}

ProxyModel ProxyTrainingRunner::Run(
    const RunConfig& config,
    ISampler& sampler,
    ILabeler& labeler) {
  ValidateRunConfig(config);
  if (config.execution_mode != ExecutionMode::SingleMachine) {
    throw std::logic_error(
        "Distributed execution is not connected to ProxyTrainingRunner yet");
  }
  if (config.label_mode != LabelMode::Clean) {
    throw std::logic_error("Propagated labels are not implemented yet");
  }
  if (config.training_placement != TrainingPlacement::Central) {
    throw std::logic_error("Federated training is not implemented yet");
  }

  const std::size_t initial_batch_size =
      config.rounds == 1
          ? config.label_budget
          : std::max<std::size_t>(
                1, std::min(config.label_budget, static_cast<std::size_t>(
                    static_cast<double>(config.label_budget) * config.initial_label_fraction)));
  std::size_t remaining_budget = config.label_budget - initial_batch_size;
  auto worker = std::make_shared<distributed::detail::DuckDbShardWorker>(
      connection_, sampler, labeler);
  distributed::SingleMachineBackend backend(worker);

  distributed::InitialSamplingRequest initial_request;
  initial_request.dataset = config.dataset;
  initial_request.label_mode = config.label_mode;
  initial_request.label_budget = initial_batch_size;
  initial_request.seed = config.seed;

  const auto initial_batches = backend.AcquireInitialLabels(initial_request);
  std::vector<LabeledExample> all_labeled_examples = initial_batches.front().examples;
  ProxyModel model = trainer_.Train(all_labeled_examples);
  backend.BroadcastModel({model, 0});

  for (std::size_t round = 1; round < config.rounds && remaining_budget > 0; ++round) {
    const std::size_t rounds_remaining = config.rounds - round;
    const std::size_t batch_size =
        (remaining_budget + rounds_remaining - 1) / rounds_remaining;
    distributed::RecursiveSamplingRequest recursive_request;
    recursive_request.dataset = config.dataset;
    recursive_request.current_model = model;
    recursive_request.label_budget = batch_size;
    recursive_request.round_index = round;

    const auto batches = backend.AcquireUncertainLabels(recursive_request);
    const std::vector<LabeledExample>& labels = batches.front().examples;
    if (labels.empty()) {
      break;
    }
    all_labeled_examples.insert(
        all_labeled_examples.end(), labels.begin(), labels.end());
    model = trainer_.Train(all_labeled_examples);
    backend.BroadcastModel({model, round});
    remaining_budget -= labels.size();
  }

  return model;
}

}  // namespace kea
