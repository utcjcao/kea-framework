#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace duckdb {
class Connection;
}  // namespace duckdb

namespace kea {

// Row IDs are kept independent of a DuckDB row position so that selections
// remain meaningful if storage changes later.
using RowId = std::string;
using Embedding = std::vector<float>;

// Identifies the DuckDB table used by one training run. The runner owns the
// queries; this type does not itself provide data access. For this MVP, the
// referenced columns must be VARCHAR ID, VARCHAR text, and FLOAT[] embedding.
struct TrainingDataset {
  std::string table_name;
  std::string id_column = "id";
  std::string text_column = "text";
  std::string embedding_column = "embedding";
};

// `rounds` is the total number of train operations.  One round means initial
// sampling, labeling, and training only.  Later rounds add uncertainty-picked
// rows and retrain on every labeled example accumulated so far.
struct TrainingConfig {
  std::size_t rounds = 1;
  std::size_t initial_batch_size = 50;
  std::size_t batch_size_per_round = 50;
  std::uint64_t seed = 42;
};

struct Candidate {
  RowId id;
  std::string text;
  Embedding embedding;
};

struct LabeledExample {
  Candidate candidate;
  int label = 0;
};

// The persisted output of logistic-regression training.
struct ProxyModel {
  std::vector<float> weights;
  float intercept = 0.0F;

  [[nodiscard]] float PredictProbability(const Embedding& embedding) const;
};

struct TrainingResult {
  ProxyModel final_model;
};

// Logistic-regression settings for the MVP. They are constructor options
// rather than top-level run settings because logistic regression is not yet a
// user-selectable training model.
struct LogisticRegressionOptions {
  float l2_regularization = 0.0F;
};

struct ClusterSamplingOptions {
  std::size_t cluster_count = 50;
  std::size_t max_iterations = 10;
};

// Framework-created access to supported initial-selection operations. Sampler
// implementations request an operation; they never construct database queries
// or manage dataset state directly.
class IInitialSamplingContext {
 public:
  virtual ~IInitialSamplingContext() = default;

  virtual std::vector<RowId> SelectRandomIds(
      std::size_t budget,
      std::uint64_t seed) = 0;

  virtual std::vector<RowId> SelectClusterRepresentativeIds(
      const ClusterSamplingOptions& options,
      std::uint64_t seed) = 0;
};

// Extension point for choosing the first labeled examples. Recursive
// uncertainty selection is deliberately owned by ProxyTrainingRunner, not by
// a sampler, in this MVP.
class ISampler {
 public:
  virtual ~ISampler() = default;

  // Returns at most `budget` stable IDs through the provided context.
  virtual std::vector<RowId> SelectInitial(
      IInitialSamplingContext& context,
      std::size_t budget,
      std::uint64_t seed) = 0;
};

// Extension point for converting selected source rows into training labels.
class ILabeler {
 public:
  virtual ~ILabeler() = default;

  virtual std::vector<LabeledExample> Label(
      const std::vector<Candidate>& candidates) = 0;
};

// Logistic regression is fixed for the MVP, so this is a concrete component,
// not a second extension interface. Its implementation delegates fitting to
// mlpack and returns only the portable weight/intercept artifact.
class LogisticRegressionTrainer {
 public:
  explicit LogisticRegressionTrainer(
      LogisticRegressionOptions options = {});

  [[nodiscard]] ProxyModel Train(
      const std::vector<LabeledExample>& examples) const;

 private:
  LogisticRegressionOptions options_;
};

// Owns the complete single-machine lifecycle:
// initial sample -> label -> train -> recursive uncertainty sample -> retrain.
// The DuckDB connection must outlive the runner.
class ProxyTrainingRunner {
 public:
  explicit ProxyTrainingRunner(duckdb::Connection& connection);

  [[nodiscard]] TrainingResult Run(
      const TrainingDataset& dataset,
      const TrainingConfig& config,
      ISampler& sampler,
      ILabeler& labeler);

 private:
  duckdb::Connection& connection_;
  LogisticRegressionTrainer trainer_;
};

}  // namespace kea
