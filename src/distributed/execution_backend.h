#pragma once

#include <memory>
#include <vector>

#include "distributed/protocol.h"

namespace kea::distributed {

class IShardWorker {
 public:
  virtual ~IShardWorker() = default;
  [[nodiscard]] virtual ShardId Id() const = 0;
  virtual LabeledExampleBatch AcquireInitialLabels(const InitialSamplingRequest& request) = 0;
  virtual LabeledExampleBatch AcquireUncertainLabels(const RecursiveSamplingRequest& request) = 0;
  virtual LocalModelResult TrainLocalModel(const LocalTrainingRequest& request) = 0;
  virtual void ReceiveModel(const ModelBroadcast& broadcast) = 0;
};

// Communication-framework integration point: a future gRPC, MPI, or TCP
// backend implements this interface, serializes these requests, and dispatches
// them to remote shard workers. Keep transport-specific types out of the
// coordinator loop and DuckDB shard worker.
class ITrainingExecutionBackend {
 public:
  virtual ~ITrainingExecutionBackend() = default;
  [[nodiscard]] virtual std::vector<LabeledExampleBatch> AcquireInitialLabels(
      const InitialSamplingRequest& request) = 0;
  [[nodiscard]] virtual std::vector<LabeledExampleBatch> AcquireUncertainLabels(
      const RecursiveSamplingRequest& request) = 0;
  [[nodiscard]] virtual std::vector<LocalModelResult> TrainLocalModels(
      const LocalTrainingRequest& request) = 0;
  virtual void BroadcastModel(const ModelBroadcast& broadcast) = 0;
};

class SingleMachineBackend final : public ITrainingExecutionBackend {
 public:
  explicit SingleMachineBackend(std::shared_ptr<IShardWorker> worker);
  [[nodiscard]] std::vector<LabeledExampleBatch> AcquireInitialLabels(
      const InitialSamplingRequest& request) override;
  [[nodiscard]] std::vector<LabeledExampleBatch> AcquireUncertainLabels(
      const RecursiveSamplingRequest& request) override;
  [[nodiscard]] std::vector<LocalModelResult> TrainLocalModels(
      const LocalTrainingRequest& request) override;
  void BroadcastModel(const ModelBroadcast& broadcast) override;

 private:
  std::shared_ptr<IShardWorker> worker_;
};

}  // namespace kea::distributed
