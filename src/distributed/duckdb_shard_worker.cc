#include "distributed/duckdb_shard_worker.h"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "duckdb.hpp"

#include "detail/proxy_training_helpers.h"
#include "detail/training_data_access.h"
#include "kea/proxy_training.h"

namespace kea::distributed::detail {
namespace {

std::uint64_t ElapsedUs(const std::chrono::steady_clock::time_point& start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

LabeledExampleBatch LabelAndRecord(
    duckdb::Connection& connection,
    ILabeler& labeler,
    const std::vector<Candidate>& candidates,
    const std::vector<RowId>& ids,
    std::uint64_t sampling_us,
    std::uint64_t fetching_us) {
  std::vector<LabeledExample> examples = labeler.Label(candidates);
  kea::detail::ValidateLabels(examples, ids);
  kea::detail::RecordLabeledIds(connection, ids);
  LabeledExampleBatch batch;
  batch.examples = std::move(examples);
  batch.sampling_us = sampling_us;
  batch.fetching_us = fetching_us;
  return batch;
}

LabeledExampleBatch FetchLabelAndRecord(
    duckdb::Connection& connection,
    const TrainingDataset& dataset,
    ILabeler& labeler,
    const std::vector<RowId>& ids,
    std::uint64_t sampling_us,
    std::uint64_t prior_fetching_us,
    kea::detail::CandidateFetchTiming* detailed_fetch_timing,
    const kea::detail::EmbeddingCache* cache) {
  const auto fetch_start = std::chrono::steady_clock::now();
  const std::vector<Candidate> candidates =
      kea::detail::FetchCandidatesByIds(connection, dataset, ids, detailed_fetch_timing, cache);
  return LabelAndRecord(connection, labeler, candidates, ids, sampling_us,
                        prior_fetching_us + ElapsedUs(fetch_start));
}

}  // namespace

DuckDbShardWorker::DuckDbShardWorker(
    ShardId id,
    duckdb::Connection& connection,
    ISampler& sampler,
    ILabeler& labeler)
    : connection_(connection), id_(std::move(id)), sampler_(sampler), labeler_(labeler) {
  if (id_.empty()) {
    throw std::invalid_argument("DuckDB shard worker requires a non-empty ID");
  }
}

DuckDbShardWorker::~DuckDbShardWorker() {
  try {
    kea::detail::DropLabeledIdsTable(connection_);
  } catch (...) {
    // Destructors must not throw; an earlier training error remains primary.
  }
}

ShardId DuckDbShardWorker::Id() const {
  return id_;
}

LabeledExampleBatch DuckDbShardWorker::AcquireInitialLabels(
    const InitialSamplingRequest& request) {
  kea::detail::DropLabeledIdsTable(connection_);
  kea::detail::CreateLabeledIdsTable(connection_);
  timing_ = {};
  embedding_cache_ = {};
  cluster_partition_.reset();
  direct_labels_.clear();
  labeled_ids_.clear();
  try {
    if (request.label_budget == 0) {
      return {Id(), {}};
    }
    kea::detail::DuckDbInitialSamplingContext sampling_context(
        connection_, request.dataset, &embedding_cache_);
    const auto sampling_start = std::chrono::steady_clock::now();
    const std::vector<RowId> ids = sampler_.SelectInitial(
        sampling_context, request.label_budget, request.seed);
    const std::uint64_t sampling_us = ElapsedUs(sampling_start);
    timing_.initial_sampling = sampling_context.timing();
    kea::detail::ValidateSelectedIds(ids, request.label_budget);
    auto batch = FetchLabelAndRecord(
        connection_, request.dataset, labeler_, ids, sampling_us, 0,
        &timing_.initial_selected_fetch,
        embedding_cache_.loaded ? &embedding_cache_ : nullptr);
    labeled_ids_.insert(ids.begin(), ids.end());
    direct_labels_ = batch.examples;
    if (request.label_mode == LabelMode::Propagated) {
      cluster_partition_ = sampling_context.TakeClusterPartition();
      if (!cluster_partition_.has_value()) {
        throw std::invalid_argument(
            "Propagated labels require a sampler that creates a cluster partition");
      }
      batch.propagated_examples = kea::detail::BuildPropagatedExamples(
          embedding_cache_, *cluster_partition_, direct_labels_);
    }
    batch.shard_id = Id();
    return batch;
  } catch (...) {
    kea::detail::DropLabeledIdsTable(connection_);
    throw;
  }
}

LabeledExampleBatch DuckDbShardWorker::AcquireUncertainLabels(
    const RecursiveSamplingRequest& request) {
  const auto unlabeled_fetch_start = std::chrono::steady_clock::now();
  kea::detail::CandidateFetchTiming fetch_timing;
  if (!embedding_cache_.loaded) {
    kea::detail::LoadEmbeddingCache(connection_, request.dataset, embedding_cache_, &fetch_timing);
  }
  const auto cache_filter_start = std::chrono::steady_clock::now();
  const std::vector<Candidate> candidates =
      kea::detail::CachedUnlabeledCandidates(embedding_cache_, labeled_ids_);
  fetch_timing.materialization_us += ElapsedUs(cache_filter_start);
  timing_.recursive_unlabeled_fetch.query_us += fetch_timing.query_us;
  timing_.recursive_unlabeled_fetch.materialization_us += fetch_timing.materialization_us;
  const std::uint64_t unlabeled_fetching_us = ElapsedUs(unlabeled_fetch_start);
  const auto sampling_start = std::chrono::steady_clock::now();
  kea::detail::UncertaintySelectionTiming selection_timing;
  const std::vector<Candidate> selected = kea::detail::SelectMostUncertainCandidates(
      candidates, request.current_model, request.label_budget, &selection_timing);
  timing_.recursive_scoring_us += selection_timing.scoring_us;
  timing_.recursive_sort_and_copy_us += selection_timing.sorting_and_copying_us;
  std::vector<RowId> ids;
  ids.reserve(selected.size());
  for (const Candidate& candidate : selected) {
    ids.push_back(candidate.id);
  }
  const std::uint64_t sampling_us = ElapsedUs(sampling_start);
  if (ids.empty()) {
    return {Id(), {}, {}, sampling_us, unlabeled_fetching_us};
  }
  auto batch = LabelAndRecord(
      connection_, labeler_, selected, ids, sampling_us, unlabeled_fetching_us);
  labeled_ids_.insert(ids.begin(), ids.end());
  direct_labels_.insert(direct_labels_.end(), batch.examples.begin(), batch.examples.end());
  if (request.label_mode == LabelMode::Propagated) {
    if (!cluster_partition_.has_value()) {
      throw std::logic_error("Propagated recursive labels require an initial cluster partition");
    }
    batch.propagated_examples = kea::detail::BuildPropagatedExamples(
        embedding_cache_, *cluster_partition_, direct_labels_);
  }
  batch.shard_id = Id();
  return batch;
}

LocalModelResult DuckDbShardWorker::TrainLocalModel(const LocalTrainingRequest&) {
  throw std::logic_error("Local federated training is not implemented");
}

void DuckDbShardWorker::ReceiveModel(const ModelBroadcast&) {
  // The local worker scores directly from the recursive request in this MVP.
}

}  // namespace kea::distributed::detail
