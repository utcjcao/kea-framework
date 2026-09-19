#include "detail/proxy_training_helpers.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace kea::detail {

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

std::vector<Candidate> SelectMostUncertainCandidates(
    const std::vector<Candidate>& candidates,
    const ProxyModel& model,
    std::size_t budget,
    UncertaintySelectionTiming* timing) {
  if (timing != nullptr) {
    *timing = {};
  }
  const auto scoring_start = std::chrono::steady_clock::now();
  std::vector<std::pair<float, std::size_t>> scores;
  scores.reserve(candidates.size());
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const Candidate& candidate = candidates[index];
    const float uncertainty = std::abs(model.PredictProbability(candidate.embedding) - 0.5F);
    scores.emplace_back(uncertainty, index);
  }
  if (timing != nullptr) {
    timing->scoring_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - scoring_start).count());
  }
  const auto selection_start = std::chrono::steady_clock::now();
  std::sort(scores.begin(), scores.end(), [&candidates](const auto& left, const auto& right) {
    return left.first == right.first
        ? candidates[left.second].id < candidates[right.second].id
        : left.first < right.first;
  });

  const std::size_t count = std::min(budget, scores.size());
  std::vector<Candidate> selected;
  selected.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    selected.push_back(candidates[scores[index].second]);
  }
  if (timing != nullptr) {
    timing->sorting_and_copying_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - selection_start).count());
  }
  return selected;
}

}  // namespace kea::detail
