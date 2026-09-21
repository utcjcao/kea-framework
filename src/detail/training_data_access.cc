#include "detail/training_data_access.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <armadillo>
#include "duckdb.hpp"
#include <mlpack/core/math/random.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>

#include "detail/duckdb_sql_helpers.h"

namespace kea::detail {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t ElapsedUs(const Clock::time_point& start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
}

void ThrowIfFailed(const duckdb::MaterializedQueryResult& result,
                   std::string_view operation) {
  if (result.HasError()) {
    throw std::runtime_error(std::string(operation) + ": " + result.GetError());
  }
}

Candidate CandidateFromResult(duckdb::MaterializedQueryResult& result,
                              duckdb::idx_t row) {
  const duckdb::Value embedding_value = result.GetValue(2, row);
  if (embedding_value.IsNull()) {
    throw std::runtime_error("Candidate embedding must not be NULL");
  }

  Candidate candidate;
  candidate.id = result.GetValue(0, row).ToString();
  candidate.text = result.GetValue(1, row).ToString();
  for (const duckdb::Value& value : duckdb::ListValue::GetChildren(embedding_value)) {
    if (value.IsNull()) {
      throw std::runtime_error("Candidate embedding values must not be NULL");
    }
    candidate.embedding.push_back(value.GetValue<float>());
  }
  return candidate;
}

Embedding EmbeddingFromValue(const duckdb::Value& embedding_value) {
  if (embedding_value.IsNull()) {
    throw std::runtime_error("Candidate embedding must not be NULL");
  }
  Embedding embedding;
  for (const duckdb::Value& value : duckdb::ListValue::GetChildren(embedding_value)) {
    if (value.IsNull()) {
      throw std::runtime_error("Candidate embedding values must not be NULL");
    }
    embedding.push_back(value.GetValue<float>());
  }
  return embedding;
}

std::string CandidateColumns(const TrainingDataset& dataset) {
  return QuoteDuckDbIdentifier(dataset.id_column) + ", " +
      QuoteDuckDbIdentifier(dataset.text_column) + ", " +
      QuoteDuckDbIdentifier(dataset.embedding_column);
}

std::string SourceCandidateColumns(const TrainingDataset& dataset) {
  return "source." + QuoteDuckDbIdentifier(dataset.id_column) + ", source." +
      QuoteDuckDbIdentifier(dataset.text_column) + ", source." +
      QuoteDuckDbIdentifier(dataset.embedding_column);
}

std::string IdAndEmbeddingColumns(const TrainingDataset& dataset) {
  return QuoteDuckDbIdentifier(dataset.id_column) + ", " +
      QuoteDuckDbIdentifier(dataset.embedding_column);
}

std::string SqlStringList(const std::vector<RowId>& ids) {
  std::ostringstream values;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index != 0) {
      values << ", ";
    }
    values << QuoteDuckDbStringLiteral(ids[index]);
  }
  return values.str();
}

}  // namespace

void LoadEmbeddingCache(
    duckdb::Connection& connection,
    const TrainingDataset& dataset,
    EmbeddingCache& cache,
    CandidateFetchTiming* timing) {
  if (timing != nullptr) {
    *timing = {};
  }
  if (cache.loaded) {
    return;
  }

  const std::string query = "SELECT " + IdAndEmbeddingColumns(dataset) + " FROM " +
      QuoteDuckDbQualifiedIdentifier(dataset.table_name);
  const auto query_start = Clock::now();
  auto result = connection.Query(query);
  if (timing != nullptr) {
    timing->query_us = ElapsedUs(query_start);
  }
  ThrowIfFailed(*result, "Unable to load embedding cache");

  const auto materialization_start = Clock::now();
  cache.rows.clear();
  cache.rows.reserve(result->RowCount());
  for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
    cache.rows.push_back({result->GetValue(0, row).ToString(),
                          EmbeddingFromValue(result->GetValue(1, row))});
  }
  cache.loaded = true;
  if (timing != nullptr) {
    timing->materialization_us = ElapsedUs(materialization_start);
  }
}

std::vector<Candidate> CachedUnlabeledCandidates(
    const EmbeddingCache& cache,
    const std::unordered_set<RowId>& labeled_ids) {
  std::vector<Candidate> candidates;
  candidates.reserve(cache.rows.size() - std::min(cache.rows.size(), labeled_ids.size()));
  for (const CachedEmbedding& row : cache.rows) {
    if (labeled_ids.find(row.id) == labeled_ids.end()) {
      candidates.push_back({row.id, {}, row.embedding});
    }
  }
  return candidates;
}

std::vector<RowId> DuckDbInitialSamplingContext::SelectRandomIds(
    std::size_t budget,
    std::uint64_t seed) {
  if (budget == 0) {
    return {};
  }

  const std::string id_column = QuoteDuckDbIdentifier(dataset_.id_column);
  const std::string query =
      "SELECT " + id_column +
      " FROM " + QuoteDuckDbQualifiedIdentifier(dataset_.table_name) +
      " WHERE " + id_column + " IS NOT NULL"
      " ORDER BY hash(" + id_column + ", " + std::to_string(seed) + "), " + id_column +
      " LIMIT " + std::to_string(budget);
  timing_ = {};
  const auto query_start = Clock::now();
  auto result = connection_.Query(query);
  timing_.query_us = ElapsedUs(query_start);
  ThrowIfFailed(*result, "Unable to select random initial candidates");

  const auto materialization_start = Clock::now();
  std::vector<RowId> selected_ids;
  selected_ids.reserve(result->RowCount());
  for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
    selected_ids.push_back(result->GetValue(0, row).ToString());
  }
  timing_.materialization_us = ElapsedUs(materialization_start);
  return selected_ids;
}

std::vector<RowId> DuckDbInitialSamplingContext::SelectClusterRepresentativeIds(
    const ClusterSamplingOptions& options,
    std::uint64_t seed) {
  if (options.cluster_count == 0 || options.max_iterations == 0) {
    throw std::invalid_argument("Cluster sampling requires positive options");
  }

  timing_ = {};
  cluster_partition_.reset();
  EmbeddingCache local_cache;
  EmbeddingCache& cache = cache_ != nullptr ? *cache_ : local_cache;
  CandidateFetchTiming cache_timing;
  LoadEmbeddingCache(connection_, dataset_, cache, &cache_timing);
  timing_.query_us = cache_timing.query_us;
  timing_.materialization_us = cache_timing.materialization_us;
  if (cache.rows.empty()) {
    return {};
  }

  const std::size_t dimensions = cache.rows.front().embedding.size();
  if (dimensions == 0) {
    throw std::runtime_error("Cannot cluster empty embeddings");
  }
  const auto matrix_copy_start = Clock::now();
  arma::mat features(dimensions, cache.rows.size());
  for (std::size_t column = 0; column < cache.rows.size(); ++column) {
    if (cache.rows[column].embedding.size() != dimensions) {
      throw std::runtime_error("Cannot cluster embeddings with different dimensions");
    }
    for (std::size_t row = 0; row < dimensions; ++row) {
      features(row, column) = cache.rows[column].embedding[row];
    }
  }
  timing_.matrix_copy_us = ElapsedUs(matrix_copy_start);

  const std::size_t cluster_count = std::min(options.cluster_count, cache.rows.size());
  mlpack::RandomSeed(seed);
  mlpack::KMeans<> kmeans(options.max_iterations);
  arma::Row<std::size_t> assignments;
  arma::mat centroids;
  const auto kmeans_start = Clock::now();
  kmeans.Cluster(features, cluster_count, assignments, centroids);
  timing_.kmeans_us = ElapsedUs(kmeans_start);

  ClusterPartition partition;
  partition.cluster_for_id.reserve(cache.rows.size());
  for (std::size_t index = 0; index < cache.rows.size(); ++index) {
    partition.cluster_for_id.emplace(cache.rows[index].id, assignments[index]);
  }
  cluster_partition_ = std::move(partition);

  const auto representatives_start = Clock::now();
  std::vector<std::size_t> representative_indices(cluster_count, cache.rows.size());
  std::vector<double> representative_distances(cluster_count, 0.0);
  for (std::size_t index = 0; index < cache.rows.size(); ++index) {
    const std::size_t cluster = assignments[index];
    const double distance = arma::accu(arma::square(features.col(index) - centroids.col(cluster)));
    const std::size_t current = representative_indices[cluster];
    if (current == cache.rows.size() || distance < representative_distances[cluster] ||
        (distance == representative_distances[cluster] && cache.rows[index].id < cache.rows[current].id)) {
      representative_indices[cluster] = index;
      representative_distances[cluster] = distance;
    }
  }

  std::vector<RowId> representative_ids;
  representative_ids.reserve(cluster_count);
  for (const std::size_t index : representative_indices) {
    if (index != cache.rows.size()) {
      representative_ids.push_back(cache.rows[index].id);
    }
  }
  timing_.representative_selection_us = ElapsedUs(representatives_start);
  return representative_ids;
}

std::vector<LabeledExample> BuildPropagatedExamples(
    const EmbeddingCache& cache,
    const ClusterPartition& partition,
    const std::vector<LabeledExample>& direct_labels) {
  struct Votes {
    std::size_t positives = 0;
    std::size_t negatives = 0;
  };
  std::unordered_map<std::size_t, Votes> votes;
  for (const LabeledExample& direct : direct_labels) {
    const auto assignment = partition.cluster_for_id.find(direct.candidate.id);
    if (assignment == partition.cluster_for_id.end()) {
      continue;
    }
    Votes& cluster_votes = votes[assignment->second];
    if (direct.label == 1) {
      ++cluster_votes.positives;
    } else {
      ++cluster_votes.negatives;
    }
  }

  std::vector<LabeledExample> propagated;
  propagated.reserve(cache.rows.size());
  for (const CachedEmbedding& row : cache.rows) {
    const auto assignment = partition.cluster_for_id.find(row.id);
    if (assignment == partition.cluster_for_id.end()) {
      continue;
    }
    const auto cluster_votes = votes.find(assignment->second);
    if (cluster_votes == votes.end()) {
      continue;
    }
    // A tie is positive, matching SwanLake's recursive propagation rule.
    const int pseudo_label =
        cluster_votes->second.positives >= cluster_votes->second.negatives ? 1 : 0;
    propagated.push_back({{row.id, {}, row.embedding}, pseudo_label});
  }
  return propagated;
}

void CreateLabeledIdsTable(duckdb::Connection& connection) {
  auto result = connection.Query(
      "CREATE TEMP TABLE kea_labeled_ids (id VARCHAR PRIMARY KEY)");
  ThrowIfFailed(*result, "Unable to create temporary labeled-ID table");
}

void DropLabeledIdsTable(duckdb::Connection& connection) {
  auto result = connection.Query("DROP TABLE IF EXISTS kea_labeled_ids");
  ThrowIfFailed(*result, "Unable to drop temporary labeled-ID table");
}

std::vector<Candidate> FetchCandidatesByIds(
    duckdb::Connection& connection,
    const TrainingDataset& dataset,
    const std::vector<RowId>& ids,
    CandidateFetchTiming* timing,
    const EmbeddingCache* cache) {
  if (timing != nullptr) {
    *timing = {};
  }
  if (ids.empty()) {
    return {};
  }
  const std::string columns = cache == nullptr
      ? CandidateColumns(dataset)
      : QuoteDuckDbIdentifier(dataset.id_column) + ", " +
            QuoteDuckDbIdentifier(dataset.text_column);
  const std::string query = "SELECT " + columns +
      " FROM " + QuoteDuckDbQualifiedIdentifier(dataset.table_name) +
      " WHERE " + QuoteDuckDbIdentifier(dataset.id_column) +
      " IN (" + SqlStringList(ids) + ")";

  std::vector<Candidate> candidates;
  candidates.reserve(ids.size());
  const auto query_start = Clock::now();
  auto result = connection.Query(query);
  if (timing != nullptr) {
    timing->query_us = ElapsedUs(query_start);
  }
  ThrowIfFailed(*result, "Unable to fetch selected candidates");
  if (result->RowCount() != ids.size()) {
    throw std::runtime_error("Selected candidate IDs were not found exactly once");
  }
  const auto materialization_start = Clock::now();
  for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
    if (cache == nullptr) {
      candidates.push_back(CandidateFromResult(*result, row));
      continue;
    }
    const RowId id = result->GetValue(0, row).ToString();
    const auto cached = std::find_if(cache->rows.begin(), cache->rows.end(), [&id](const auto& value) {
      return value.id == id;
    });
    if (cached == cache->rows.end()) {
      throw std::runtime_error("Selected candidate is missing from the embedding cache");
    }
    candidates.push_back({id, result->GetValue(1, row).ToString(), cached->embedding});
  }
  if (timing != nullptr) {
    timing->materialization_us = ElapsedUs(materialization_start);
  }
  return candidates;
}

std::vector<Candidate> FetchUnlabeledCandidates(
    duckdb::Connection& connection,
    const TrainingDataset& dataset,
    CandidateFetchTiming* timing) {
  if (timing != nullptr) {
    *timing = {};
  }
  const std::string id_column = QuoteDuckDbIdentifier(dataset.id_column);
  const std::string query = "SELECT " + SourceCandidateColumns(dataset) + " FROM " +
      QuoteDuckDbQualifiedIdentifier(dataset.table_name) + " AS source"
      " LEFT JOIN kea_labeled_ids AS labeled ON source." + id_column + " = labeled.id"
      " WHERE labeled.id IS NULL";
  const auto query_start = Clock::now();
  auto result = connection.Query(query);
  if (timing != nullptr) {
    timing->query_us = ElapsedUs(query_start);
  }
  ThrowIfFailed(*result, "Unable to fetch unlabeled candidates");

  const auto materialization_start = Clock::now();
  std::vector<Candidate> candidates;
  candidates.reserve(result->RowCount());
  for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
    candidates.push_back(CandidateFromResult(*result, row));
  }
  if (timing != nullptr) {
    timing->materialization_us = ElapsedUs(materialization_start);
  }
  return candidates;
}

void RecordLabeledIds(
    duckdb::Connection& connection,
    const std::vector<RowId>& ids) {
  if (ids.empty()) {
    return;
  }
  std::string values;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index != 0) {
      values += ", ";
    }
    values += "(" + QuoteDuckDbStringLiteral(ids[index]) + ")";
  }
  auto result = connection.Query("INSERT INTO kea_labeled_ids VALUES " + values);
  ThrowIfFailed(*result, "Unable to record labeled candidate IDs");
}

}  // namespace kea::detail
