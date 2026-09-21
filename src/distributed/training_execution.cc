#include "distributed/training_execution.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

namespace kea::distributed::detail {
namespace {

std::vector<LabeledExample> FlattenDirectLabels(
    const std::vector<LabeledExampleBatch>& batches) {
  std::vector<LabeledExample> examples;
  for (const auto& batch : batches) {
    examples.insert(examples.end(), batch.examples.begin(), batch.examples.end());
  }
  return examples;
}

std::vector<LabeledExample> FlattenPropagatedTrainingExamples(
    const std::vector<LabeledExampleBatch>& batches) {
  std::vector<LabeledExample> examples;
  for (const auto& batch : batches) {
    examples.insert(examples.end(),
                    batch.propagated_examples.begin(), batch.propagated_examples.end());
  }
  return examples;
}

std::uint64_t SumSamplingUs(const std::vector<LabeledExampleBatch>& batches) {
  std::uint64_t total = 0;
  for (const auto& batch : batches) {
    total += batch.sampling_us;
  }
  return total;
}

std::uint64_t SumFetchingUs(const std::vector<LabeledExampleBatch>& batches) {
  std::uint64_t total = 0;
  for (const auto& batch : batches) {
    total += batch.fetching_us;
  }
  return total;
}

std::uint64_t ElapsedUs(const std::chrono::steady_clock::time_point& start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

}  // namespace

ProxyModel RunCentralTraining(
    const RunConfig& config,
    ITrainingExecutionBackend& backend,
    const LogisticRegressionTrainer& trainer,
    TrainingExecutionTiming* timing) {
  ValidateRunConfig(config);
  if (config.training_placement != TrainingPlacement::Central) {
    throw std::logic_error("Federated training is not implemented yet");
  }
  if (timing != nullptr) {
    *timing = {};
  }
  const auto total_start = std::chrono::steady_clock::now();

  const std::size_t initial_budget =
      config.rounds == 1
          ? config.label_budget
          : std::max<std::size_t>(
                1, std::min(config.label_budget, static_cast<std::size_t>(
                    static_cast<double>(config.label_budget) * config.initial_label_fraction)));
  std::size_t remaining_budget = config.label_budget - initial_budget;

  InitialSamplingRequest initial_request;
  initial_request.dataset = config.dataset;
  initial_request.label_mode = config.label_mode;
  initial_request.label_budget = initial_budget;
  initial_request.seed = config.seed;

  const auto initial_acquisition_start = std::chrono::steady_clock::now();
  const std::vector<LabeledExampleBatch> initial_batches =
      backend.AcquireInitialLabels(initial_request);
  std::vector<LabeledExample> all_direct_labels = FlattenDirectLabels(initial_batches);
  const std::uint64_t initial_acquisition_us = ElapsedUs(initial_acquisition_start);
  if (all_direct_labels.empty()) {
    throw std::runtime_error("Initial sampling produced no labeled examples");
  }
  std::vector<LabeledExample> training_examples =
      config.label_mode == LabelMode::Clean
          ? all_direct_labels
          : FlattenPropagatedTrainingExamples(initial_batches);
  if (training_examples.empty()) {
    throw std::runtime_error("Propagated initial sampling produced no training examples");
  }
  const auto initial_training_start = std::chrono::steady_clock::now();
  ProxyModel model = trainer.Train(training_examples);
  const std::uint64_t initial_training_us = ElapsedUs(initial_training_start);
  if (timing != nullptr) {
    timing->rounds.push_back({0, initial_budget, all_direct_labels.size(),
                              SumSamplingUs(initial_batches), SumFetchingUs(initial_batches),
                              initial_acquisition_us, initial_training_us});
  }
  backend.BroadcastModel({model, 0});

  for (std::size_t round = 1; round < config.rounds && remaining_budget > 0; ++round) {
    const std::size_t rounds_remaining = config.rounds - round;
    const std::size_t budget = (remaining_budget + rounds_remaining - 1) / rounds_remaining;
    RecursiveSamplingRequest request;
    request.dataset = config.dataset;
    request.current_model = model;
    request.label_mode = config.label_mode;
    request.label_budget = budget;
    request.round_index = round;

    const auto acquisition_start = std::chrono::steady_clock::now();
    const std::vector<LabeledExampleBatch> batches = backend.AcquireUncertainLabels(request);
    const std::vector<LabeledExample> labels = FlattenDirectLabels(batches);
    const std::uint64_t acquisition_us = ElapsedUs(acquisition_start);
    if (labels.empty()) {
      if (timing != nullptr) {
        timing->rounds.push_back({round, budget, 0,
                                  SumSamplingUs(batches), SumFetchingUs(batches),
                                  acquisition_us, 0});
      }
      break;
    }
    all_direct_labels.insert(all_direct_labels.end(), labels.begin(), labels.end());
    training_examples = config.label_mode == LabelMode::Clean
        ? all_direct_labels
        : FlattenPropagatedTrainingExamples(batches);
    if (training_examples.empty()) {
      throw std::runtime_error("Propagated recursive sampling produced no training examples");
    }
    const auto training_start = std::chrono::steady_clock::now();
    model = trainer.Train(training_examples);
    const std::uint64_t training_us = ElapsedUs(training_start);
    if (timing != nullptr) {
      timing->rounds.push_back({round, budget, labels.size(),
                                SumSamplingUs(batches), SumFetchingUs(batches),
                                acquisition_us, training_us});
    }
    backend.BroadcastModel({model, round});
    remaining_budget -= labels.size();
  }
  if (timing != nullptr) {
    timing->total_us = ElapsedUs(total_start);
  }
  return model;
}

}  // namespace kea::distributed::detail
