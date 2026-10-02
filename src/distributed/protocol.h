#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kea/run_config.h"

namespace kea::distributed {

using ShardId = std::string;

struct InitialSamplingRequest {
  TrainingDataset dataset;
  std::size_t label_budget = 0;
  std::uint64_t seed = 42;
};

struct RecursiveSamplingRequest {
  TrainingDataset dataset;
  ProxyModel current_model;
  std::size_t label_budget = 0;
  std::size_t round_index = 0;
  float uncertainty_center_1 = 0.5F;
  float uncertainty_center_2 = 0.5F;
};

struct LocalTrainingRequest {
  std::size_t round_index = 0;
  bool use_propagated_labels = false;
};

struct LabeledExampleBatch {
  ShardId shard_id;
  // Direct oracle/LLM labels. These, not propagated examples, consume the
  // configured label budget and are retained across recursive rounds.
  std::vector<LabeledExample> direct_labels;
  // Current shard-local training set after the configured training-data
  // builder has transformed the accumulated direct labels. It may contain
  // only direct labels or also pseudo-labeled rows.
  std::vector<LabeledExample> training_examples;
  std::uint64_t sampling_us = 0;
  std::uint64_t fetching_us = 0;
};

struct LocalModelResult {
  ShardId shard_id;
  ProxyModel model;
  std::size_t training_row_count = 0;
};

struct ModelBroadcast {
  ProxyModel model;
  std::size_t round_index = 0;
};

}  // namespace kea::distributed
