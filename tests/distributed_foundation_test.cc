#include "distributed/execution_backend.h"
#include "kea/run_config.h"

#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

class TestWorker final : public kea::distributed::IShardWorker {
 public:
  explicit TestWorker(kea::distributed::ShardId id) : id_(std::move(id)) {}

  kea::distributed::ShardId Id() const override { return id_; }

  kea::distributed::LabeledExampleBatch AcquireInitialLabels(
      const kea::distributed::InitialSamplingRequest&) override {
    return {id_, {{{id_ + "-row", "", {1.0F}}, 1}}};
  }

  kea::distributed::LabeledExampleBatch AcquireUncertainLabels(
      const kea::distributed::RecursiveSamplingRequest& request) override {
    assert(request.current_model.weights.size() == 1);
    return {id_, {{{id_ + "-uncertain", "", {1.0F}}, 1}}};
  }

  kea::distributed::LocalModelResult TrainLocalModel(
      const kea::distributed::LocalTrainingRequest&) override {
    return {id_, {{1.0F}, 0.0F}, 1};
  }

  void ReceiveModel(const kea::distributed::ModelBroadcast& broadcast) override {
    last_broadcast_ = broadcast;
  }

  std::optional<kea::distributed::ModelBroadcast> last_broadcast_;

 private:
  kea::distributed::ShardId id_;
};

bool ThrowsInvalidConfig(const kea::RunConfig& config) {
  try {
    kea::ValidateRunConfig(config);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  kea::RunConfig distributed_config;
  distributed_config.execution_mode = kea::ExecutionMode::Distributed;
  distributed_config.dataset.table_name = "documents";
  distributed_config.workers = {{"worker-a", "in-process://a"}, {"worker-b", "in-process://b"}};
  kea::ValidateRunConfig(distributed_config);

  kea::RunConfig invalid_config = distributed_config;
  invalid_config.execution_mode = kea::ExecutionMode::SingleMachine;
  assert(ThrowsInvalidConfig(invalid_config));

  auto worker_a = std::make_shared<TestWorker>("worker-a");
  kea::distributed::SingleMachineBackend backend(worker_a);

  kea::distributed::InitialSamplingRequest initial;
  initial.dataset.table_name = "documents";
  const auto labeled = backend.AcquireInitialLabels(initial);
  assert(labeled.size() == 1);
  assert(labeled[0].examples.size() == 1);

  kea::distributed::ModelBroadcast broadcast{{{2.0F}, -1.0F}, 1};
  backend.BroadcastModel(broadcast);
  assert(worker_a->last_broadcast_->round_index == 1);

  kea::distributed::RecursiveSamplingRequest recursive;
  recursive.current_model = broadcast.model;
  const auto uncertain = backend.AcquireUncertainLabels(recursive);
  assert(uncertain.size() == 1);
  assert(uncertain[0].examples[0].candidate.id == "worker-a-uncertain");

  const auto models = backend.TrainLocalModels({});
  assert(models.size() == 1);
  assert(models[0].training_row_count == 1);
}
