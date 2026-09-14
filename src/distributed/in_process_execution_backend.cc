#include "distributed/execution_backend.h"

#include <stdexcept>
namespace kea::distributed {

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

}  // namespace kea::distributed
