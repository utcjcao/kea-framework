#include "distributed/duckdb_shard_worker.h"
#include "distributed/execution_backend.h"
#include "distributed/training_execution.h"
#include "kea/function_labeler.h"
#include "kea/run_config.h"
#include "kea/samplers/cluster_sampler.h"

#include <cassert>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "duckdb.hpp"

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

void CreateShard(duckdb::Connection& connection, const std::vector<float>& values) {
  auto create = connection.Query(
      "CREATE TABLE examples (id VARCHAR PRIMARY KEY, text VARCHAR, embedding FLOAT[])");
  assert(!create->HasError());
  for (std::size_t index = 0; index < values.size(); ++index) {
    const std::string id = "row-" + std::to_string(index) + "-" + std::to_string(values[index]);
    const std::string text = values[index] > 0.0F ? "positive" : "negative";
    const auto insert = connection.Query(
        "INSERT INTO examples VALUES ('" + id + "', '" + text + "', [" +
        std::to_string(values[index]) + "])" );
    assert(!insert->HasError());
  }
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

  // LC1-style algorithm test: local k-means representatives on two separate
  // DuckDB shards, clean labels pooled for central training, and a broadcast
  // after each of two rounds.
  duckdb::DuckDB shard_a_database(nullptr);
  duckdb::DuckDB shard_b_database(nullptr);
  duckdb::Connection shard_a_connection(shard_a_database);
  duckdb::Connection shard_b_connection(shard_b_database);
  CreateShard(shard_a_connection, {-6.0F, -5.0F, -4.0F, 4.0F, 5.0F, 6.0F});
  CreateShard(shard_b_connection, {-3.0F, -2.0F, -1.0F, 1.0F, 2.0F, 3.0F});

  std::vector<kea::RowId> labeled_ids;
  kea::FunctionLabeler labeler([&labeled_ids](const kea::Candidate& candidate) {
    labeled_ids.push_back(candidate.id);
    return candidate.embedding[0] > 0.0F ? 1 : 0;
  });
  kea::ClusterSamplingOptions options;
  options.cluster_count = 2;
  kea::ClusterSampler sampler(options);
  auto shard_a_worker = std::make_shared<kea::distributed::detail::DuckDbShardWorker>(
      "shard-a", shard_a_connection, sampler, labeler);
  auto shard_b_worker = std::make_shared<kea::distributed::detail::DuckDbShardWorker>(
      "shard-b", shard_b_connection, sampler, labeler);
  kea::distributed::InProcessMultiShardBackend multi_backend(
      {shard_a_worker, shard_b_worker});

  kea::RunConfig lc1_config;
  lc1_config.execution_mode = kea::ExecutionMode::Distributed;
  lc1_config.dataset.table_name = "examples";
  lc1_config.rounds = 2;
  lc1_config.label_budget = 8;
  lc1_config.initial_label_fraction = 0.5;
  lc1_config.seed = 42;
  lc1_config.workers = {{"shard-a", "in-process://a"}, {"shard-b", "in-process://b"}};

  const kea::LogisticRegressionTrainer trainer;
  const kea::ProxyModel lc1_model = kea::distributed::detail::RunCleanCentralTraining(
      lc1_config, multi_backend, trainer);
  assert(labeled_ids.size() == 8);
  assert(std::unordered_set<kea::RowId>(labeled_ids.begin(), labeled_ids.end()).size() == 8);
  assert(lc1_model.PredictProbability({-2.0F}) < 0.5F);
  assert(lc1_model.PredictProbability({2.0F}) > 0.5F);
}
