#pragma once

#include <functional>

#include "kea/proxy_training.h"

namespace kea {

// Minimal concrete labeler for tests and integrations that already have a
// labeling function. Production labelers can implement ILabeler directly.
class FunctionLabeler final : public ILabeler {
 public:
  using LabelFunction = std::function<int(const Candidate&)>;

  explicit FunctionLabeler(LabelFunction label_function);

  [[nodiscard]] std::vector<LabeledExample> Label(
      const std::vector<Candidate>& candidates) override;

 private:
  LabelFunction label_function_;
};

}  // namespace kea
