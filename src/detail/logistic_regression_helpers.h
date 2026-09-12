#pragma once

#include "kea/proxy_training.h"

namespace kea::detail {

[[nodiscard]] float Sigmoid(float value);

void ValidateLogisticRegressionOptions(const LogisticRegressionOptions& options);
void ValidateLabeledExamples(const std::vector<LabeledExample>& examples);

}  // namespace kea::detail
