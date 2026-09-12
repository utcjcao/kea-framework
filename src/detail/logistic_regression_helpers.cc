#include "detail/logistic_regression_helpers.h"

#include <cmath>
#include <stdexcept>

namespace kea::detail {

float Sigmoid(float value) {
  // This equivalent form avoids overflow for strongly negative logits.
  if (value >= 0.0F) {
    return 1.0F / (1.0F + std::exp(-value));
  }
  const float exponential = std::exp(value);
  return exponential / (1.0F + exponential);
}

void ValidateLogisticRegressionOptions(const LogisticRegressionOptions& options) {
  if (!std::isfinite(options.l2_regularization) || options.l2_regularization < 0.0F) {
    throw std::invalid_argument("Logistic regression l2_regularization must be non-negative");
  }
}

void ValidateLabeledExamples(const std::vector<LabeledExample>& examples) {
  if (examples.empty()) {
    throw std::invalid_argument("Logistic regression requires at least one labeled example");
  }

  const std::size_t dimension = examples.front().candidate.embedding.size();
  if (dimension == 0) {
    throw std::invalid_argument("Logistic regression embeddings must not be empty");
  }

  for (const LabeledExample& example : examples) {
    if (example.label != 0 && example.label != 1) {
      throw std::invalid_argument("Logistic regression labels must be binary (0 or 1)");
    }
    if (example.candidate.embedding.size() != dimension) {
      throw std::invalid_argument("Logistic regression embeddings must have one shared dimension");
    }
    for (const float feature : example.candidate.embedding) {
      if (!std::isfinite(feature)) {
        throw std::invalid_argument("Logistic regression embeddings must be finite");
      }
    }
  }
}

}  // namespace kea::detail
