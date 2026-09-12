#pragma once

#include <string_view>
#include <vector>

#include "kea/proxy_training.h"

namespace duckdb {
class Connection;
}  // namespace duckdb

namespace kea::detail {

inline constexpr std::string_view kLabeledIdsTable = "kea_labeled_ids";

class DuckDbInitialSamplingContext final : public IInitialSamplingContext {
 public:
  DuckDbInitialSamplingContext(
      duckdb::Connection& connection,
      const TrainingDataset& dataset)
      : connection_(connection), dataset_(dataset) {}

  [[nodiscard]] std::vector<RowId> SelectRandomIds(
      std::size_t budget,
      std::uint64_t seed) override;

  [[nodiscard]] std::vector<RowId> SelectClusterRepresentativeIds(
      const ClusterSamplingOptions& options,
      std::uint64_t seed) override;

 private:
  duckdb::Connection& connection_;
  const TrainingDataset& dataset_;
};

void CreateLabeledIdsTable(duckdb::Connection& connection);
void DropLabeledIdsTable(duckdb::Connection& connection);

[[nodiscard]] std::vector<Candidate> FetchCandidatesByIds(
    duckdb::Connection& connection,
    const TrainingDataset& dataset,
    const std::vector<RowId>& ids);

[[nodiscard]] std::vector<Candidate> FetchUnlabeledCandidates(
    duckdb::Connection& connection,
    const TrainingDataset& dataset);

void RecordLabeledIds(
    duckdb::Connection& connection,
    const std::vector<RowId>& ids);

}  // namespace kea::detail
