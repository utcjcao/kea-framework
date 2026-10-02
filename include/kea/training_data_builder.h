#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "kea/proxy_training.h"

namespace kea {

// Inputs available after an ILabeler has produced direct ground-truth labels.
// A builder may use the optional full candidate pool and cluster assignments to
// derive additional training examples, but only direct_labels count against the
// LLM-label budget.
struct TrainingDataBuildInput {
  const std::vector<LabeledExample>& direct_labels;
  const std::vector<Candidate>* candidate_pool = nullptr;
  const std::unordered_map<RowId, std::size_t>* cluster_assignments = nullptr;
};

// Fourth extension point in the training pipeline:
// sampler -> labeler -> training-data builder -> trainer.
//
// The labeler obtains ground truth for the selected rows. A builder decides
// which examples and labels the trainer receives, including any pseudo-label
// expansion over unlabeled rows. Implementations run on each worker, so the
// same abstraction also applies when execution becomes distributed.
class ITrainingDataBuilder {
 public:
  virtual ~ITrainingDataBuilder() = default;

  // Lets the framework prepare only the additional state this builder needs.
  [[nodiscard]] virtual bool RequiresCandidatePool() const { return false; }
  [[nodiscard]] virtual bool RequiresClusterAssignments() const { return false; }

  [[nodiscard]] virtual std::vector<LabeledExample> Build(
      const TrainingDataBuildInput& input) = 0;
};

// Uses only the directly labeled examples, preserving the previous default
// clean-label behavior.
class DirectLabelTrainingDataBuilder final : public ITrainingDataBuilder {
 public:
  [[nodiscard]] std::vector<LabeledExample> Build(
      const TrainingDataBuildInput& input) override;
};

// Assigns every row in a cluster the majority direct label observed for that
// cluster. Ties are positive and clusters without a direct label are omitted.
class ClusterPropagationTrainingDataBuilder final : public ITrainingDataBuilder {
 public:
  [[nodiscard]] bool RequiresCandidatePool() const override { return true; }
  [[nodiscard]] bool RequiresClusterAssignments() const override { return true; }

  [[nodiscard]] std::vector<LabeledExample> Build(
      const TrainingDataBuildInput& input) override;
};

}  // namespace kea
