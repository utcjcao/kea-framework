#if defined(KEA_HAS_SEMBENCH)

#include "sembench_test_data.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

#include "duckdb.hpp"

#include "kea/function_labeler.h"
#include "kea/proxy_training.h"
#include "kea/run_config.h"
#include "kea/samplers/random_sampler.h"

namespace {

constexpr std::size_t kRowsPerDataset = 768;

void RunEndToEndCase(
    const kea::test::SemBenchDatasetPaths& dataset,
    std::size_t rounds,
    std::size_t label_budget) {
  assert(std::filesystem::is_regular_file(dataset.csv));
  assert(std::filesystem::is_regular_file(dataset.embeddings));

  duckdb::DuckDB database(nullptr);
  duckdb::Connection connection(database);
  const auto examples = kea::test::LoadSemBenchExamples(
      connection, dataset, kRowsPerDataset);
  assert(examples.size() == kRowsPerDataset);
  assert(examples.front().candidate.embedding.size() == 1024);
  kea::test::CreateExamplesTable(connection, examples);

  const auto labels = kea::test::LabelsById(examples);
  std::vector<kea::RowId> labeled_ids;
  kea::FunctionLabeler oracle([&labels, &labeled_ids](const kea::Candidate& candidate) {
    labeled_ids.push_back(candidate.id);
    return labels.at(candidate.id);
  });
  kea::RandomSampler sampler;
  kea::ProxyTrainingRunner runner(connection);
  kea::RunConfig config;
  config.dataset.table_name = "examples";
  config.rounds = rounds;
  config.label_budget = label_budget;
  config.initial_label_fraction = 0.5;
  config.seed = 42;

  const kea::ProxyModel model = runner.Run(config, sampler, oracle);
  assert(labeled_ids.size() == label_budget);
  assert(std::unordered_set<kea::RowId>(labeled_ids.begin(), labeled_ids.end()).size() == label_budget);
  assert(model.weights.size() == 1024);
  for (const float weight : model.weights) {
    assert(std::isfinite(weight));
  }
  assert(std::isfinite(model.intercept));
  const float probability = model.PredictProbability(examples.front().candidate.embedding);
  assert(std::isfinite(probability));
  assert(probability >= 0.0F && probability <= 1.0F);
}

}  // namespace

int main() {
  const std::filesystem::path sembench_root = KEA_SEMBENCH_DIR;
  // Movie exercises one-shot random sampling with real review sentiment.
  RunEndToEndCase(kea::test::FindSemBenchDataset(
      sembench_root, kea::test::SemBenchDataset::Movie),
      /*rounds=*/1, /*label_budget=*/64);
  // FEVER exercises recursive uncertainty sampling with real claim labels.
  RunEndToEndCase(kea::test::FindSemBenchDataset(
      sembench_root, kea::test::SemBenchDataset::Fever),
      /*rounds=*/3, /*label_budget=*/96);
}

#else

int main() {
  return 0;
}

#endif
