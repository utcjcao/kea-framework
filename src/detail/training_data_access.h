#pragma once

#include <cstdint>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "kea/proxy_training.h"

namespace duckdb {
class Connection;
}  // namespace duckdb

namespace kea::detail {

inline constexpr std::string_view kLabeledIdsTable = "kea_labeled_ids";

// Fine-grained local timing used only by the benchmark harness.
struct CandidateFetchTiming {
  std::uint64_t query_us = 0;
  std::uint64_t materialization_us = 0;
};

struct InitialSamplingTiming {
  std::uint64_t query_us = 0;
  std::uint64_t materialization_us = 0;
  std::uint64_t matrix_copy_us = 0;
  std::uint64_t kmeans_us = 0;
  std::uint64_t representative_selection_us = 0;
};

// Per-run cache: the operations that train and score a proxy require an ID
// and embedding, while text is only needed for the small batch sent to a
// labeler. It is intentionally internal to the DuckDB worker.
struct CachedEmbedding {
  RowId id;
  Embedding embedding;
};

struct EmbeddingCache {
  std::vector<CachedEmbedding> rows;
  bool loaded = false;
};

void LoadEmbeddingCache(
    duckdb::Connection& connection,
    const TrainingDataset& dataset,
    EmbeddingCache& cache,
    CandidateFetchTiming* timing = nullptr);

[[nodiscard]] std::vector<Candidate> CachedUnlabeledCandidates(
    const EmbeddingCache& cache,
    const std::unordered_set<RowId>& labeled_ids);

class DuckDbInitialSamplingContext final : public IInitialSamplingContext {
 public:
  DuckDbInitialSamplingContext(
      duckdb::Connection& connection,
      const TrainingDataset& dataset,
      EmbeddingCache* cache = nullptr)
      : connection_(connection), dataset_(dataset), cache_(cache) {}

  [[nodiscard]] std::vector<RowId> SelectRandomIds(
      std::size_t budget,
      std::uint64_t seed) override;

  [[nodiscard]] std::vector<RowId> SelectClusterRepresentativeIds(
      const ClusterSamplingOptions& options,
      std::uint64_t seed) override;

  [[nodiscard]] const InitialSamplingTiming& timing() const { return timing_; }

 private:
  duckdb::Connection& connection_;
  const TrainingDataset& dataset_;
  EmbeddingCache* cache_;
  InitialSamplingTiming timing_;
};

void CreateLabeledIdsTable(duckdb::Connection& connection);
void DropLabeledIdsTable(duckdb::Connection& connection);

[[nodiscard]] std::vector<Candidate> FetchCandidatesByIds(
    duckdb::Connection& connection,
    const TrainingDataset& dataset,
    const std::vector<RowId>& ids,
    CandidateFetchTiming* timing = nullptr,
    const EmbeddingCache* cache = nullptr);

[[nodiscard]] std::vector<Candidate> FetchUnlabeledCandidates(
    duckdb::Connection& connection,
    const TrainingDataset& dataset,
    CandidateFetchTiming* timing = nullptr);

void RecordLabeledIds(
    duckdb::Connection& connection,
    const std::vector<RowId>& ids);

}  // namespace kea::detail
