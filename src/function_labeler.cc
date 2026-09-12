#include "kea/function_labeler.h"

#include <stdexcept>
#include <utility>

namespace kea {

FunctionLabeler::FunctionLabeler(LabelFunction label_function)
    : label_function_(std::move(label_function)) {
  if (!label_function_) {
    throw std::invalid_argument("FunctionLabeler requires a label function");
  }
}

std::vector<LabeledExample> FunctionLabeler::Label(
    const std::vector<Candidate>& candidates) {
  std::vector<LabeledExample> labeled_examples;
  labeled_examples.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    labeled_examples.push_back({candidate, label_function_(candidate)});
  }
  return labeled_examples;
}

}  // namespace kea
