#pragma once

#include "distributed/execution_backend.h"

namespace duckdb {
class Connection;
}  // namespace duckdb

namespace kea {
class ILabeler;
class ISampler;
}  // namespace kea

namespace kea::distributed::detail {

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

 private:
  duckdb::Connection& connection_;
  ShardId id_;
  ISampler& sampler_;
  ILabeler& labeler_;
};

}  // namespace kea::distributed::detail
