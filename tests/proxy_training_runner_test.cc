#include "kea/function_labeler.h"
#include "kea/proxy_training.h"
#include "kea/samplers/cluster_sampler.h"
#include "kea/samplers/random_sampler.h"
#include "kea/run_config.h"

#include <cassert>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "duckdb.hpp"

namespace {

void RequireSuccess(const duckdb::MaterializedQueryResult& result) {
  assert(!result.HasError());
}

void CreateTestDataset(duckdb::Connection& connection) {
  auto result = connection.Query(R"(
    CREATE TABLE examples (
      id VARCHAR PRIMARY KEY,
      text VARCHAR NOT NULL,
      embedding FLOAT[] NOT NULL
    );
    INSERT INTO examples VALUES
      ('row-01', 'negative', [-5.0]),
      ('row-02', 'positive', [5.0]),
      ('row-03', 'negative', [-4.0]),
      ('row-04', 'positive', [4.0]),
      ('row-05', 'negative', [-3.0]),
      ('row-06', 'positive', [3.0]),
      ('row-07', 'negative', [-2.0]),
      ('row-08', 'positive', [2.0]),
      ('row-09', 'negative', [-1.0]),
      ('row-10', 'positive', [1.0]),
      ('row-11', 'negative', [-0.5]),
      ('row-12', 'positive', [0.5]);
  )");
  RequireSuccess(*result);
}

kea::FunctionLabeler MakeOracleLabeler(std::vector<kea::RowId>* labeled_ids) {
  const std::unordered_map<kea::RowId, int> labels = {
      {"row-01", 0}, {"row-02", 1}, {"row-03", 0}, {"row-04", 1},
      {"row-05", 0}, {"row-06", 1}, {"row-07", 0}, {"row-08", 1},
      {"row-09", 0}, {"row-10", 1}, {"row-11", 0}, {"row-12", 1},
  };
  return kea::FunctionLabeler([labels, labeled_ids](const kea::Candidate& candidate) {
    labeled_ids->push_back(candidate.id);
    return labels.at(candidate.id);
  });
}

}  // namespace

int main() {
  duckdb::DuckDB database(nullptr);
  duckdb::Connection connection(database);
  CreateTestDataset(connection);

  kea::RandomSampler sampler;
  kea::ProxyTrainingRunner runner(connection);

  std::vector<kea::RowId> one_round_ids;
  auto one_round_labeler = MakeOracleLabeler(&one_round_ids);
  kea::RunConfig one_round_config;
  one_round_config.dataset.table_name = "examples";
  one_round_config.rounds = 1;
  one_round_config.label_budget = 6;
  one_round_config.seed = 42;
  const kea::ProxyModel one_round_model =
      runner.Run(one_round_config, sampler, one_round_labeler);
  assert(one_round_ids.size() == 6);
  assert(std::unordered_set<kea::RowId>(one_round_ids.begin(), one_round_ids.end()).size() == 6);
  assert(one_round_model.PredictProbability({-4.0F}) < 0.5F);
  assert(one_round_model.PredictProbability({4.0F}) > 0.5F);

  std::vector<kea::RowId> recursive_ids;
  auto recursive_labeler = MakeOracleLabeler(&recursive_ids);
  kea::RunConfig recursive_config;
  recursive_config.dataset.table_name = "examples";
  recursive_config.rounds = 3;
  recursive_config.label_budget = 10;
  recursive_config.initial_label_fraction = 0.6;
  recursive_config.seed = 42;
  const kea::ProxyModel recursive_model =
      runner.Run(recursive_config, sampler, recursive_labeler);
  assert(recursive_ids.size() == 10);
  assert(std::unordered_set<kea::RowId>(recursive_ids.begin(), recursive_ids.end()).size() == 10);
  assert(recursive_model.PredictProbability({-4.0F}) < 0.5F);
  assert(recursive_model.PredictProbability({4.0F}) > 0.5F);

  std::vector<kea::RowId> cluster_ids;
  auto cluster_labeler = MakeOracleLabeler(&cluster_ids);
  kea::ClusterSamplingOptions cluster_options;
  cluster_options.cluster_count = 2;
  cluster_options.max_iterations = 20;
  kea::ClusterSampler cluster_sampler(cluster_options);
  kea::RunConfig cluster_config;
  cluster_config.dataset.table_name = "examples";
  cluster_config.rounds = 1;
  cluster_config.label_budget = 2;
  cluster_config.seed = 42;
  const kea::ProxyModel cluster_model =
      runner.Run(cluster_config, cluster_sampler, cluster_labeler);
  assert(cluster_ids.size() == 2);
  assert(cluster_model.PredictProbability({-4.0F}) < 0.5F);
  assert(cluster_model.PredictProbability({4.0F}) > 0.5F);
}
