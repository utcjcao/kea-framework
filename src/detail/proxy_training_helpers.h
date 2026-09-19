#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "kea/proxy_training.h"

namespace kea::detail {

struct UncertaintySelectionTiming {
  std::uint64_t scoring_us = 0;
  std::uint64_t sorting_and_copying_us = 0;
};

void ValidateSelectedIds(
    const std::vector<RowId>& ids,
    std::size_t budget);

void ValidateLabels(
    const std::vector<LabeledExample>& examples,
    const std::vector<RowId>& selected_ids);

[[nodiscard]] std::vector<Candidate> SelectMostUncertainCandidates(
    const std::vector<Candidate>& candidates,
    const ProxyModel& model,
    std::size_t budget,
    UncertaintySelectionTiming* timing = nullptr);

}  // namespace kea::detail
