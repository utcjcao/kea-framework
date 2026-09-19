#pragma once

#include <cstdint>
#include <unordered_set>

#include "detail/training_data_access.h"
#include "distributed/execution_backend.h"

namespace duckdb {
class Connection;
}  // namespace duckdb

namespace kea {
class ILabeler;
class ISampler;
}  // namespace kea

namespace kea::distributed::detail {

struct ShardWorkerTiming {
  kea::detail::InitialSamplingTiming initial_sampling;
  kea::detail::CandidateFetchTiming initial_selected_fetch;
  kea::detail::CandidateFetchTiming recursive_unlabeled_fetch;
  std::uint64_t recursive_scoring_us = 0;
  std::uint64_t recursive_sort_and_copy_us = 0;
};

// The local execution implementation. It is intentionally an internal class:
// callers interact with the runner, sampler, labeler, and RunConfig instead.
class DuckDbShardWorker final : public IShardWorker {
 public:
  DuckDbShardWorker(
      ShardId id, duckdb::Connection& connection, ISampler& sampler, ILabeler& labeler);
  ~DuckDbShardWorker() override;

  [[nodiscard]] ShardId Id() const override;
  [[nodiscard]] LabeledExampleBatch AcquireInitialLabels(
      const InitialSamplingRequest& request) override;
  [[nodiscard]] LabeledExampleBatch AcquireUncertainLabels(
      const RecursiveSamplingRequest& request) override;
  [[nodiscard]] LocalModelResult TrainLocalModel(
      const LocalTrainingRequest& request) override;
  void ReceiveModel(const ModelBroadcast& broadcast) override;

  [[nodiscard]] const ShardWorkerTiming& timing() const { return timing_; }

 private:
  duckdb::Connection& connection_;
  ShardId id_;
  ISampler& sampler_;
  ILabeler& labeler_;
  ShardWorkerTiming timing_;
  kea::detail::EmbeddingCache embedding_cache_;
  std::unordered_set<RowId> labeled_ids_;
};

}  // namespace kea::distributed::detail
