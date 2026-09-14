#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "kea/run_config.h"

namespace kea::distributed {

using ShardId = std::string;

struct InitialSamplingRequest {
  TrainingDataset dataset;
  LabelMode label_mode = LabelMode::Clean;
  std::size_t label_budget = 0;
  std::uint64_t seed = 42;
};

struct RecursiveSamplingRequest {
  TrainingDataset dataset;
  ProxyModel current_model;
  std::size_t label_budget = 0;
  std::size_t round_index = 0;
};

struct LocalTrainingRequest {
  std::size_t round_index = 0;
  bool use_propagated_labels = false;
};

struct LabeledExampleBatch {
  ShardId shard_id;
  std::vector<LabeledExample> examples;
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
