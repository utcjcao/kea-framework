#pragma once

#include <cstddef>
#include <vector>

#include "kea/proxy_training.h"

namespace kea::detail {

void ValidateTrainingConfig(const TrainingConfig& config);

void ValidateSelectedIds(
    const std::vector<RowId>& ids,
    std::size_t budget);

void ValidateLabels(
    const std::vector<LabeledExample>& examples,
    const std::vector<RowId>& selected_ids);

[[nodiscard]] std::vector<RowId> SelectMostUncertainIds(
    const std::vector<Candidate>& candidates,
    const ProxyModel& model,
    std::size_t budget);

}  // namespace kea::detail
