#include "kea/training_data_builder.h"

#include <stdexcept>
#include <unordered_map>

namespace kea {

std::vector<LabeledExample> DirectLabelTrainingDataBuilder::Build(
    const TrainingDataBuildInput& input) {
  return input.direct_labels;
}

std::vector<LabeledExample> ClusterPropagationTrainingDataBuilder::Build(
    const TrainingDataBuildInput& input) {
  if (input.candidate_pool == nullptr || input.cluster_assignments == nullptr) {
    throw std::invalid_argument(
        "Cluster propagation requires a candidate pool and cluster assignments");
  }

  struct Votes {
    std::size_t positives = 0;
    std::size_t negatives = 0;
  };
  std::unordered_map<std::size_t, Votes> votes;
  for (const LabeledExample& direct : input.direct_labels) {
    const auto assignment = input.cluster_assignments->find(direct.candidate.id);
    if (assignment == input.cluster_assignments->end()) {
      continue;
    }
    Votes& cluster_votes = votes[assignment->second];
    if (direct.label == 1) {
      ++cluster_votes.positives;
    } else {
      ++cluster_votes.negatives;
    }
  }

  std::vector<LabeledExample> expanded;
  expanded.reserve(input.candidate_pool->size());
  for (const Candidate& candidate : *input.candidate_pool) {
    const auto assignment = input.cluster_assignments->find(candidate.id);
    if (assignment == input.cluster_assignments->end()) {
      continue;
    }
    const auto cluster_votes = votes.find(assignment->second);
    if (cluster_votes == votes.end()) {
      continue;
    }
    const int pseudo_label =
        cluster_votes->second.positives >= cluster_votes->second.negatives ? 1 : 0;
    expanded.push_back({candidate, pseudo_label});
  }
  return expanded;
}

}  // namespace kea
