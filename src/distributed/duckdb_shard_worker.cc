#include "distributed/duckdb_shard_worker.h"

#include <stdexcept>
#include <utility>

#include "duckdb.hpp"

#include "detail/proxy_training_helpers.h"
#include "detail/training_data_access.h"
#include "kea/proxy_training.h"

namespace kea::distributed::detail {
namespace {

LabeledExampleBatch LabelAndRecord(
    duckdb::Connection& connection,
    const TrainingDataset& dataset,
    ILabeler& labeler,
    const std::vector<RowId>& ids) {
  const std::vector<Candidate> candidates =
      kea::detail::FetchCandidatesByIds(connection, dataset, ids);
  std::vector<LabeledExample> examples = labeler.Label(candidates);
  kea::detail::ValidateLabels(examples, ids);
  kea::detail::RecordLabeledIds(connection, ids);
  return {"local", std::move(examples)};
}

}  // namespace

DuckDbShardWorker::DuckDbShardWorker(
    duckdb::Connection& connection,
    ISampler& sampler,
    ILabeler& labeler)
    : connection_(connection), sampler_(sampler), labeler_(labeler) {}

DuckDbShardWorker::~DuckDbShardWorker() {
  try {
    kea::detail::DropLabeledIdsTable(connection_);
  } catch (...) {
    // Destructors must not throw; an earlier training error remains primary.
  }
}

ShardId DuckDbShardWorker::Id() const {
  return "local";
}

LabeledExampleBatch DuckDbShardWorker::AcquireInitialLabels(
    const InitialSamplingRequest& request) {
  kea::detail::DropLabeledIdsTable(connection_);
  kea::detail::CreateLabeledIdsTable(connection_);
  try {
    kea::detail::DuckDbInitialSamplingContext sampling_context(connection_, request.dataset);
    const std::vector<RowId> ids = sampler_.SelectInitial(
        sampling_context, request.label_budget, request.seed);
    kea::detail::ValidateSelectedIds(ids, request.label_budget);
    return LabelAndRecord(connection_, request.dataset, labeler_, ids);
  } catch (...) {
    kea::detail::DropLabeledIdsTable(connection_);
    throw;
  }
}

LabeledExampleBatch DuckDbShardWorker::AcquireUncertainLabels(
    const RecursiveSamplingRequest& request) {
  const std::vector<Candidate> candidates =
      kea::detail::FetchUnlabeledCandidates(connection_, request.dataset);
  const std::vector<RowId> ids = kea::detail::SelectMostUncertainIds(
      candidates, request.current_model, request.label_budget);
  if (ids.empty()) {
    return {Id(), {}};
  }
  return LabelAndRecord(connection_, request.dataset, labeler_, ids);
}

LocalModelResult DuckDbShardWorker::TrainLocalModel(const LocalTrainingRequest&) {
  throw std::logic_error("Local federated training is not implemented");
}

void DuckDbShardWorker::ReceiveModel(const ModelBroadcast&) {
  // The local worker scores directly from the recursive request in this MVP.
}

}  // namespace kea::distributed::detail
