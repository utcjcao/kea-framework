#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace duckdb {
class Connection;
}  // namespace duckdb

namespace kea {

struct RunConfig;

// Row IDs are kept independent of a DuckDB row position so that selections
// remain meaningful if storage changes later.
using RowId = std::string;
// A read-only embedding that either owns its values or borrows a contiguous
// FLOAT buffer from a retained DuckDB result. Copies of an owned embedding
// share its backing vector; copies of a borrowed embedding remain views. This
// keeps Candidate and LabeledExample inexpensive to pass through the training
// pipeline without allowing a borrowed DuckDB buffer to escape its owner.
class Embedding {
 public:
  Embedding() = default;
  Embedding(std::initializer_list<float> values)
      : Embedding(std::vector<float>(values)) {}
  Embedding(std::vector<float> values)
      : owned_values_(std::make_shared<std::vector<float>>(std::move(values))) {
    RefreshOwnedView();
  }

  [[nodiscard]] static Embedding Borrow(const float* values, std::size_t size) {
    if (values == nullptr && size != 0) {
      throw std::invalid_argument("Cannot borrow a non-empty embedding from a null pointer");
    }
    Embedding embedding;
    embedding.values_ = values;
    embedding.size_ = size;
    return embedding;
  }

  [[nodiscard]] std::size_t size() const { return size_; }
  [[nodiscard]] bool empty() const { return size_ == 0; }
  [[nodiscard]] const float* data() const { return values_; }
  [[nodiscard]] const float& operator[](std::size_t index) const { return values_[index]; }
  [[nodiscard]] const float* begin() const { return values_; }
  [[nodiscard]] const float* end() const { return values_ + size_; }
  [[nodiscard]] bool IsBorrowed() const { return !owned_values_ && values_ != nullptr; }

  // Mutating operations are only needed by fixture/data-loading code. A
  // borrowed DuckDB view is deliberately immutable.
  void resize(std::size_t size) {
    EnsureOwned();
    owned_values_->resize(size);
    RefreshOwnedView();
  }
  [[nodiscard]] float* data() {
    EnsureOwned();
    RefreshOwnedView();
    return const_cast<float*>(values_);
  }

 private:
  void EnsureOwned() {
    if (!owned_values_) {
      if (values_ != nullptr) {
        throw std::logic_error("Cannot mutate a borrowed embedding view");
      }
      owned_values_ = std::make_shared<std::vector<float>>();
    }
  }
  void RefreshOwnedView() {
    values_ = owned_values_->data();
    size_ = owned_values_->size();
  }

  std::shared_ptr<std::vector<float>> owned_values_;
  const float* values_ = nullptr;
  std::size_t size_ = 0;
};

// Identifies the DuckDB table used by one training run. The runner owns the
// queries; this type does not itself provide data access. For this MVP, the
// referenced columns must be VARCHAR ID, VARCHAR text, and a fixed-width
// FLOAT[dimensions] embedding array.
struct TrainingDataset {
  std::string table_name;
  std::string id_column = "id";
  std::string text_column = "text";
  std::string embedding_column = "embedding";
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

class ITrainingDataBuilder;

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

// Owns the complete training lifecycle: initial sampling -> label -> train ->
// recursive confidence-center sampling -> retrain. The caller supplies
// extensible sampling, labeling, and training-data-building components;
// deployment configuration, including the recursive selection centers, comes
// from RunConfig. The DuckDB connection must outlive the runner.
class ProxyTrainingRunner {
 public:
  explicit ProxyTrainingRunner(duckdb::Connection& connection);

  [[nodiscard]] ProxyModel Run(
      const RunConfig& config,
      ISampler& sampler,
      ILabeler& labeler,
      ITrainingDataBuilder& training_data_builder);

 private:
  duckdb::Connection& connection_;
  LogisticRegressionTrainer trainer_;
};

}  // namespace kea
