#include "kea/proxy_training.h"

#include <utility>
#include <vector>

#include "duckdb.hpp"

#include "detail/proxy_training_helpers.h"
#include "detail/training_data_access.h"

namespace kea {
ProxyTrainingRunner::ProxyTrainingRunner(duckdb::Connection& connection)
    : connection_(connection) {}

TrainingResult ProxyTrainingRunner::Run(
    const TrainingDataset& dataset,
    const TrainingConfig& config,
    ISampler& sampler,
    ILabeler& labeler) {
  detail::ValidateTrainingConfig(config);
  detail::DropLabeledIdsTable(connection_);
  detail::CreateLabeledIdsTable(connection_);

  try {
    std::vector<LabeledExample> all_labeled_examples;
    detail::DuckDbInitialSamplingContext sampling_context(connection_, dataset);
    const std::vector<RowId> initial_ids = sampler.SelectInitial(
        sampling_context, config.initial_batch_size, config.seed);
    detail::ValidateSelectedIds(initial_ids, config.initial_batch_size);

    const std::vector<Candidate> initial_candidates =
        detail::FetchCandidatesByIds(connection_, dataset, initial_ids);
    const std::vector<LabeledExample> initial_labels = labeler.Label(initial_candidates);
    detail::ValidateLabels(initial_labels, initial_ids);
    all_labeled_examples.insert(
        all_labeled_examples.end(), initial_labels.begin(), initial_labels.end());
    detail::RecordLabeledIds(connection_, initial_ids);
    ProxyModel model = trainer_.Train(all_labeled_examples);

    for (std::size_t round = 1; round < config.rounds; ++round) {
      const std::vector<Candidate> candidates =
          detail::FetchUnlabeledCandidates(connection_, dataset);
      const std::vector<RowId> selected_ids =
          detail::SelectMostUncertainIds(candidates, model, config.batch_size_per_round);
      if (selected_ids.empty()) {
        break;
      }

      const std::vector<Candidate> selected_candidates =
          detail::FetchCandidatesByIds(connection_, dataset, selected_ids);
      const std::vector<LabeledExample> labels = labeler.Label(selected_candidates);
      detail::ValidateLabels(labels, selected_ids);
      all_labeled_examples.insert(
          all_labeled_examples.end(), labels.begin(), labels.end());
      detail::RecordLabeledIds(connection_, selected_ids);
      model = trainer_.Train(all_labeled_examples);
    }

    detail::DropLabeledIdsTable(connection_);
    return {std::move(model)};
  } catch (...) {
    detail::DropLabeledIdsTable(connection_);
    throw;
  }
}

}  // namespace kea
