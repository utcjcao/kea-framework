#include <cassert>
#include <unordered_map>
#include <vector>

#include "kea/training_data_builder.h"

int main() {
  const std::vector<kea::Candidate> candidates = {
      {"a", {}, {1.0F}}, {"b", {}, {2.0F}}, {"c", {}, {3.0F}}, {"d", {}, {4.0F}}};
  const std::vector<kea::LabeledExample> direct_labels = {
      {candidates[0], 0}, {candidates[1], 1}, {candidates[2], 1}};
  const std::unordered_map<kea::RowId, std::size_t> assignments = {
      {"a", 0}, {"b", 0}, {"c", 1}, {"d", 2}};

  kea::ClusterPropagationTrainingDataBuilder propagation;
  const std::vector<kea::LabeledExample> expanded = propagation.Build(
      {direct_labels, &candidates, &assignments});

  // Cluster 0 is tied and therefore positive. Cluster 1 is positive; cluster
  // 2 has no direct label and is intentionally omitted.
  assert(expanded.size() == 3);
  assert(expanded[0].candidate.id == "a" && expanded[0].label == 1);
  assert(expanded[1].candidate.id == "b" && expanded[1].label == 1);
  assert(expanded[2].candidate.id == "c" && expanded[2].label == 1);

  kea::DirectLabelTrainingDataBuilder direct;
  const std::vector<kea::LabeledExample> clean = direct.Build({direct_labels});
  assert(clean.size() == direct_labels.size());
  assert(clean[0].candidate.id == "a" && clean[0].label == 0);
}
