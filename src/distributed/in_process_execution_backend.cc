#include "distributed/execution_backend.h"

#include <stdexcept>
#include <unordered_set>
namespace kea::distributed {
namespace {

std::size_t ShardBudget(std::size_t total, std::size_t index, std::size_t shards) {
  return total / shards + (index < total % shards ? 1 : 0);
}

}  // namespace

SingleMachineBackend::SingleMachineBackend(std::shared_ptr<IShardWorker> worker)
    : worker_(std::move(worker)) {
  if (!worker_ || worker_->Id().empty()) {
    throw std::invalid_argument("Single-machine backend requires one worker with a non-empty ID");
  }
}

std::vector<LabeledExampleBatch> SingleMachineBackend::AcquireInitialLabels(
    const InitialSamplingRequest& request) {
  return {worker_->AcquireInitialLabels(request)};
}

std::vector<LabeledExampleBatch> SingleMachineBackend::AcquireUncertainLabels(
    const RecursiveSamplingRequest& request) {
  return {worker_->AcquireUncertainLabels(request)};
}

std::vector<LocalModelResult> SingleMachineBackend::TrainLocalModels(
    const LocalTrainingRequest& request) {
  return {worker_->TrainLocalModel(request)};
}

void SingleMachineBackend::BroadcastModel(const ModelBroadcast& broadcast) {
  worker_->ReceiveModel(broadcast);
}

InProcessMultiShardBackend::InProcessMultiShardBackend(
    std::vector<std::shared_ptr<IShardWorker>> workers)
    : workers_(std::move(workers)) {
  if (workers_.empty()) {
    throw std::invalid_argument("Multi-shard backend requires at least one worker");
  }
  std::unordered_set<ShardId> ids;
  for (const auto& worker : workers_) {
    if (!worker || worker->Id().empty() || !ids.insert(worker->Id()).second) {
      throw std::invalid_argument("Multi-shard workers require unique non-empty IDs");
    }
  }
}

std::vector<LabeledExampleBatch> InProcessMultiShardBackend::AcquireInitialLabels(
    const InitialSamplingRequest& request) {
  std::vector<LabeledExampleBatch> batches;
  batches.reserve(workers_.size());
  for (std::size_t index = 0; index < workers_.size(); ++index) {
    InitialSamplingRequest shard_request = request;
    shard_request.label_budget = ShardBudget(request.label_budget, index, workers_.size());
    batches.push_back(workers_[index]->AcquireInitialLabels(shard_request));
  }
  return batches;
}

std::vector<LabeledExampleBatch> InProcessMultiShardBackend::AcquireUncertainLabels(
    const RecursiveSamplingRequest& request) {
  std::vector<LabeledExampleBatch> batches;
  batches.reserve(workers_.size());
  for (std::size_t index = 0; index < workers_.size(); ++index) {
    RecursiveSamplingRequest shard_request = request;
    shard_request.label_budget = ShardBudget(request.label_budget, index, workers_.size());
    batches.push_back(workers_[index]->AcquireUncertainLabels(shard_request));
  }
  return batches;
}

std::vector<LocalModelResult> InProcessMultiShardBackend::TrainLocalModels(
    const LocalTrainingRequest& request) {
  std::vector<LocalModelResult> models;
  models.reserve(workers_.size());
  for (const auto& worker : workers_) {
    models.push_back(worker->TrainLocalModel(request));
  }
  return models;
}

void InProcessMultiShardBackend::BroadcastModel(const ModelBroadcast& broadcast) {
  for (const auto& worker : workers_) {
    worker->ReceiveModel(broadcast);
  }
}

}  // namespace kea::distributed
