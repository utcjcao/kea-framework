#include "detail/proxy_training_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace kea::detail {

void ValidateTrainingConfig(const TrainingConfig& config) {
  if (config.rounds == 0) {
    throw std::invalid_argument("TrainingConfig.rounds must be at least one");
  }
  if (config.initial_batch_size == 0) {
    throw std::invalid_argument("TrainingConfig.initial_batch_size must be positive");
  }
  if (config.rounds > 1 && config.batch_size_per_round == 0) {
    throw std::invalid_argument("TrainingConfig.batch_size_per_round must be positive");
  }
}

void ValidateSelectedIds(const std::vector<RowId>& ids, std::size_t budget) {
  if (ids.empty()) {
    throw std::runtime_error("Sampler selected no initial candidates");
  }
  if (ids.size() > budget) {
    throw std::runtime_error("Sampler selected more candidates than its budget");
  }
  const std::unordered_set<RowId> unique_ids(ids.begin(), ids.end());
  if (unique_ids.size() != ids.size()) {
    throw std::runtime_error("Sampler selected duplicate candidate IDs");
  }
}

void ValidateLabels(const std::vector<LabeledExample>& examples,
                    const std::vector<RowId>& selected_ids) {
  if (examples.size() != selected_ids.size()) {
    throw std::runtime_error("Labeler returned an unexpected number of labels");
  }
  std::unordered_set<RowId> expected_ids(selected_ids.begin(), selected_ids.end());
  for (const LabeledExample& example : examples) {
    if (expected_ids.erase(example.candidate.id) != 1) {
      throw std::runtime_error("Labeler returned an unexpected or duplicate candidate ID");
    }
  }
  if (!expected_ids.empty()) {
    throw std::runtime_error("Labeler did not return labels for all selected IDs");
  }
}

std::vector<RowId> SelectMostUncertainIds(
    const std::vector<Candidate>& candidates,
    const ProxyModel& model,
    std::size_t budget) {
  std::vector<std::pair<float, RowId>> scores;
  scores.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    const float uncertainty = std::abs(model.PredictProbability(candidate.embedding) - 0.5F);
    scores.emplace_back(uncertainty, candidate.id);
  }
  std::sort(scores.begin(), scores.end(), [](const auto& left, const auto& right) {
    return left.first == right.first ? left.second < right.second : left.first < right.first;
  });

  const std::size_t count = std::min(budget, scores.size());
  std::vector<RowId> selected_ids;
  selected_ids.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    selected_ids.push_back(scores[index].second);
  }
  return selected_ids;
}

}  // namespace kea::detail
