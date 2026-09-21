#include "distributed/duckdb_shard_worker.h"
#include "distributed/execution_backend.h"
#include "distributed/training_execution.h"
#include "kea/function_labeler.h"
#include "kea/run_config.h"
#include "kea/samplers/cluster_sampler.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "duckdb.hpp"

#if defined(KEA_HAS_SEMBENCH)
#include "sembench_test_data.h"
#endif

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

#if defined(KEA_HAS_SEMBENCH)

void RunSemBenchDistributedFlow() {
  duckdb::DuckDB loader_database(nullptr);
  duckdb::Connection loader_connection(loader_database);
  const auto examples = kea::test::LoadSemBenchExamples(
      loader_connection,
      kea::test::FindSemBenchDataset(
          std::filesystem::path(KEA_SEMBENCH_DIR), kea::test::SemBenchDataset::Movie),
      /*maximum_rows=*/768);
  assert(examples.size() == 768);

  std::vector<kea::test::SemBenchExample> shard_a_examples;
  std::vector<kea::test::SemBenchExample> shard_b_examples;
  shard_a_examples.reserve(examples.size() / 2);
  shard_b_examples.reserve(examples.size() / 2);
  for (std::size_t index = 0; index < examples.size(); ++index) {
    (index % 2 == 0 ? shard_a_examples : shard_b_examples).push_back(examples[index]);
  }

  duckdb::DuckDB shard_a_database(nullptr);
  duckdb::DuckDB shard_b_database(nullptr);
  duckdb::Connection shard_a_connection(shard_a_database);
  duckdb::Connection shard_b_connection(shard_b_database);
  kea::test::CreateExamplesTable(shard_a_connection, shard_a_examples);
  kea::test::CreateExamplesTable(shard_b_connection, shard_b_examples);

  const auto labels = kea::test::LabelsById(examples);
  std::vector<kea::RowId> labeled_ids;
  kea::FunctionLabeler labeler([&labels, &labeled_ids](const kea::Candidate& candidate) {
    labeled_ids.push_back(candidate.id);
    return labels.at(candidate.id);
  });
  kea::ClusterSamplingOptions options;
  options.cluster_count = 12;
  options.max_iterations = 5;
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
  lc1_config.label_budget = 48;
  lc1_config.initial_label_fraction = 0.5;
  lc1_config.seed = 42;
  lc1_config.workers = {{"shard-a", "in-process://a"}, {"shard-b", "in-process://b"}};

  const kea::LogisticRegressionTrainer trainer;
  const kea::ProxyModel lc1_model = kea::distributed::detail::RunCentralTraining(
      lc1_config, multi_backend, trainer);
  assert(labeled_ids.size() == 48);
  assert(std::unordered_set<kea::RowId>(labeled_ids.begin(), labeled_ids.end()).size() == 48);
  assert(lc1_model.weights.size() == 1024);
  const float probability = lc1_model.PredictProbability(examples.front().candidate.embedding);
  assert(std::isfinite(probability));
  assert(probability >= 0.0F && probability <= 1.0F);
}

#endif

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

#if defined(KEA_HAS_SEMBENCH)
  // LC1-style flow: local clustering on two independent SemBench Movie
  // shards, clean labels pooled at the coordinator, and a model broadcast
  // after every round.
  RunSemBenchDistributedFlow();
#endif
}
