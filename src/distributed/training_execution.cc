#include "distributed/training_execution.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace kea::distributed::detail {
namespace {

std::vector<LabeledExample> Flatten(const std::vector<LabeledExampleBatch>& batches) {
  std::vector<LabeledExample> examples;
  for (const auto& batch : batches) {
    examples.insert(examples.end(), batch.examples.begin(), batch.examples.end());
  }
  return examples;
}

}  // namespace

ProxyModel RunCleanCentralTraining(
    const RunConfig& config,
    ITrainingExecutionBackend& backend,
    const LogisticRegressionTrainer& trainer) {
  ValidateRunConfig(config);
  if (config.label_mode != LabelMode::Clean) {
    throw std::logic_error("Propagated labels are not implemented yet");
  }
  if (config.training_placement != TrainingPlacement::Central) {
    throw std::logic_error("Federated training is not implemented yet");
  }

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

  std::vector<LabeledExample> all_examples = Flatten(backend.AcquireInitialLabels(initial_request));
  if (all_examples.empty()) {
    throw std::runtime_error("Initial sampling produced no labeled examples");
  }
  ProxyModel model = trainer.Train(all_examples);
  backend.BroadcastModel({model, 0});

  for (std::size_t round = 1; round < config.rounds && remaining_budget > 0; ++round) {
    const std::size_t rounds_remaining = config.rounds - round;
    const std::size_t budget = (remaining_budget + rounds_remaining - 1) / rounds_remaining;
    RecursiveSamplingRequest request;
    request.dataset = config.dataset;
    request.current_model = model;
    request.label_budget = budget;
    request.round_index = round;

    const std::vector<LabeledExample> labels = Flatten(backend.AcquireUncertainLabels(request));
    if (labels.empty()) {
      break;
    }
    all_examples.insert(all_examples.end(), labels.begin(), labels.end());
    model = trainer.Train(all_examples);
    backend.BroadcastModel({model, round});
    remaining_budget -= labels.size();
  }
  return model;
}

}  // namespace kea::distributed::detail
