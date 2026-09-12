#include "detail/training_data_access.h"

#include <stdexcept>
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

}  // namespace

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
  auto result = connection_.Query(query);
  ThrowIfFailed(*result, "Unable to select random initial candidates");

  std::vector<RowId> selected_ids;
  selected_ids.reserve(result->RowCount());
  for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
    selected_ids.push_back(result->GetValue(0, row).ToString());
  }
  return selected_ids;
}

std::vector<RowId> DuckDbInitialSamplingContext::SelectClusterRepresentativeIds(
    const ClusterSamplingOptions& options,
    std::uint64_t seed) {
  if (options.cluster_count == 0 || options.max_iterations == 0) {
    throw std::invalid_argument("Cluster sampling requires positive options");
  }

  const std::string query = "SELECT " + CandidateColumns(dataset_) + " FROM " +
      QuoteDuckDbQualifiedIdentifier(dataset_.table_name);
  auto result = connection_.Query(query);
  ThrowIfFailed(*result, "Unable to fetch candidates for clustering");
  if (result->RowCount() == 0) {
    return {};
  }

  std::vector<Candidate> candidates;
  candidates.reserve(result->RowCount());
  for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
    candidates.push_back(CandidateFromResult(*result, row));
  }

  const std::size_t dimensions = candidates.front().embedding.size();
  if (dimensions == 0) {
    throw std::runtime_error("Cannot cluster empty embeddings");
  }
  arma::mat features(dimensions, candidates.size());
  for (std::size_t column = 0; column < candidates.size(); ++column) {
    if (candidates[column].embedding.size() != dimensions) {
      throw std::runtime_error("Cannot cluster embeddings with different dimensions");
    }
    for (std::size_t row = 0; row < dimensions; ++row) {
      features(row, column) = candidates[column].embedding[row];
    }
  }

  const std::size_t cluster_count = std::min(options.cluster_count, candidates.size());
  mlpack::RandomSeed(seed);
  mlpack::KMeans<> kmeans(options.max_iterations);
  arma::Row<std::size_t> assignments;
  arma::mat centroids;
  kmeans.Cluster(features, cluster_count, assignments, centroids);

  std::vector<std::size_t> representative_indices(cluster_count, candidates.size());
  std::vector<double> representative_distances(cluster_count, 0.0);
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const std::size_t cluster = assignments[index];
    const double distance = arma::accu(arma::square(features.col(index) - centroids.col(cluster)));
    const std::size_t current = representative_indices[cluster];
    if (current == candidates.size() || distance < representative_distances[cluster] ||
        (distance == representative_distances[cluster] && candidates[index].id < candidates[current].id)) {
      representative_indices[cluster] = index;
      representative_distances[cluster] = distance;
    }
  }

  std::vector<RowId> representative_ids;
  representative_ids.reserve(cluster_count);
  for (const std::size_t index : representative_indices) {
    if (index != candidates.size()) {
      representative_ids.push_back(candidates[index].id);
    }
  }
  return representative_ids;
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
    const std::vector<RowId>& ids) {
  const std::string query = "SELECT " + CandidateColumns(dataset) +
      " FROM " + QuoteDuckDbQualifiedIdentifier(dataset.table_name) +
      " WHERE " + QuoteDuckDbIdentifier(dataset.id_column) + " = ";

  std::vector<Candidate> candidates;
  candidates.reserve(ids.size());
  for (const RowId& id : ids) {
    auto result = connection.Query(query + QuoteDuckDbStringLiteral(id));
    ThrowIfFailed(*result, "Unable to fetch selected candidate");
    if (result->RowCount() != 1) {
      throw std::runtime_error("Selected candidate ID was not found exactly once");
    }
    candidates.push_back(CandidateFromResult(*result, 0));
  }
  return candidates;
}

std::vector<Candidate> FetchUnlabeledCandidates(
    duckdb::Connection& connection,
    const TrainingDataset& dataset) {
  const std::string id_column = QuoteDuckDbIdentifier(dataset.id_column);
  const std::string query = "SELECT " + SourceCandidateColumns(dataset) + " FROM " +
      QuoteDuckDbQualifiedIdentifier(dataset.table_name) + " AS source"
      " LEFT JOIN kea_labeled_ids AS labeled ON source." + id_column + " = labeled.id"
      " WHERE labeled.id IS NULL";
  auto result = connection.Query(query);
  ThrowIfFailed(*result, "Unable to fetch unlabeled candidates");

  std::vector<Candidate> candidates;
  candidates.reserve(result->RowCount());
  for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
    candidates.push_back(CandidateFromResult(*result, row));
  }
  return candidates;
}

void RecordLabeledIds(
    duckdb::Connection& connection,
    const std::vector<RowId>& ids) {
  for (const RowId& id : ids) {
    auto result = connection.Query(
        "INSERT INTO kea_labeled_ids VALUES (" + QuoteDuckDbStringLiteral(id) + ")");
    ThrowIfFailed(*result, "Unable to record labeled candidate ID");
  }
}

}  // namespace kea::detail
