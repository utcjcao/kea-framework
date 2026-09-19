#include "distributed/duckdb_shard_worker.h"
#include "distributed/execution_backend.h"
#include "distributed/training_execution.h"
#include "sembench_test_data.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "duckdb.hpp"

#include "kea/proxy_training.h"
#include "kea/run_config.h"
#include "kea/samplers/cluster_sampler.h"
#include "kea/samplers/random_sampler.h"

namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t ElapsedUs(const Clock::time_point& start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
}

class TimedSampler final : public kea::ISampler {
 public:
  explicit TimedSampler(kea::ISampler& sampler) : sampler_(sampler) {}

  std::vector<kea::RowId> SelectInitial(
      kea::IInitialSamplingContext& context,
      std::size_t budget,
      std::uint64_t seed) override {
    const auto start = Clock::now();
    std::vector<kea::RowId> ids = sampler_.SelectInitial(context, budget, seed);
    elapsed_us_ += ElapsedUs(start);
    return ids;
  }

  [[nodiscard]] std::uint64_t elapsed_us() const { return elapsed_us_; }

 private:
  kea::ISampler& sampler_;
  std::uint64_t elapsed_us_ = 0;
};

struct LabelCallTiming {
  std::size_t labels = 0;
  std::uint64_t elapsed_us = 0;
};

class TimedOracleLabeler final : public kea::ILabeler {
 public:
  explicit TimedOracleLabeler(const std::unordered_map<kea::RowId, int>& labels)
      : labels_(labels) {}

  std::vector<kea::LabeledExample> Label(
      const std::vector<kea::Candidate>& candidates) override {
    const auto start = Clock::now();
    std::vector<kea::LabeledExample> examples;
    examples.reserve(candidates.size());
    for (const kea::Candidate& candidate : candidates) {
      examples.push_back({candidate, labels_.at(candidate.id)});
    }
    calls_.push_back({examples.size(), ElapsedUs(start)});
    return examples;
  }

  [[nodiscard]] const std::vector<LabelCallTiming>& calls() const { return calls_; }

 private:
  const std::unordered_map<kea::RowId, int>& labels_;
  std::vector<LabelCallTiming> calls_;
};

struct ScoredRow {
  std::string id;
  int label = 0;
  float score = 0.0F;
};

struct CascadeResult {
  std::size_t escalations = 0;
  double final_accuracy = 0.0;
};

struct EvaluationResult {
  std::uint64_t scoring_us = 0;
  double proxy_accuracy = 0.0;
  double aurac = 0.0;
  CascadeResult narrow_band;
  CascadeResult wide_band;
};

CascadeResult EvaluateCascade(
    const std::vector<ScoredRow>& scores,
    float reject_threshold,
    float accept_threshold) {
  std::size_t correct = 0;
  std::size_t escalations = 0;
  for (const ScoredRow& row : scores) {
    int prediction = 0;
    if (row.score <= reject_threshold) {
      prediction = 0;
    } else if (row.score >= accept_threshold) {
      prediction = 1;
    } else {
      // The SemBench ground-truth label is a perfect-oracle LLM response.
      prediction = row.label;
      ++escalations;
    }
    correct += prediction == row.label;
  }
  return {escalations, static_cast<double>(correct) / static_cast<double>(scores.size())};
}

EvaluationResult EvaluateProxy(
    const kea::ProxyModel& model,
    const std::vector<kea::test::SemBenchExample>& examples) {
  const auto scoring_start = Clock::now();
  std::vector<ScoredRow> scores;
  scores.reserve(examples.size());
  std::size_t correct = 0;
  for (const auto& example : examples) {
    const float score = model.PredictProbability(example.candidate.embedding);
    const int prediction = score >= 0.5F ? 1 : 0;
    correct += prediction == example.label;
    scores.push_back({example.candidate.id, example.label, score});
  }
  const std::uint64_t scoring_us = ElapsedUs(scoring_start);

  std::sort(scores.begin(), scores.end(), [](const ScoredRow& left, const ScoredRow& right) {
    const float left_confidence = std::abs(left.score - 0.5F);
    const float right_confidence = std::abs(right.score - 0.5F);
    return left_confidence == right_confidence ? left.id < right.id
                                               : left_confidence > right_confidence;
  });
  double accuracy_sum = 0.0;
  std::size_t kept_correct = 0;
  for (std::size_t index = 0; index < scores.size(); ++index) {
    const int prediction = scores[index].score >= 0.5F ? 1 : 0;
    kept_correct += prediction == scores[index].label;
    accuracy_sum += static_cast<double>(kept_correct) / static_cast<double>(index + 1);
  }

  EvaluationResult result;
  result.scoring_us = scoring_us;
  result.proxy_accuracy = static_cast<double>(correct) / static_cast<double>(scores.size());
  result.aurac = accuracy_sum / static_cast<double>(scores.size());
  result.narrow_band = EvaluateCascade(scores, /*reject_threshold=*/0.2F, /*accept_threshold=*/0.8F);
  result.wide_band = EvaluateCascade(scores, /*reject_threshold=*/0.1F, /*accept_threshold=*/0.9F);
  return result;
}

double UsToMs(std::uint64_t microseconds) {
  return static_cast<double>(microseconds) / 1000.0;
}

std::uint64_t SumTrainingUs(const kea::distributed::detail::TrainingExecutionTiming& timing) {
  return std::accumulate(timing.rounds.begin(), timing.rounds.end(), std::uint64_t{0},
                         [](std::uint64_t sum, const auto& round) {
                           return sum + round.training_us;
                         });
}

std::uint64_t SumSamplingUs(
    const kea::distributed::detail::TrainingExecutionTiming& timing,
    std::size_t begin) {
  return std::accumulate(timing.rounds.begin() + std::min(begin, timing.rounds.size()),
                         timing.rounds.end(), std::uint64_t{0},
                         [](std::uint64_t sum, const auto& round) {
                           return sum + round.sampling_us;
                         });
}

std::uint64_t SumFetchingUs(
    const kea::distributed::detail::TrainingExecutionTiming& timing,
    std::size_t begin) {
  return std::accumulate(timing.rounds.begin() + std::min(begin, timing.rounds.size()),
                         timing.rounds.end(), std::uint64_t{0},
                         [](std::uint64_t sum, const auto& round) {
                           return sum + round.fetching_us;
                         });
}

std::uint64_t SumLabelUs(const std::vector<LabelCallTiming>& calls, std::size_t begin) {
  return std::accumulate(calls.begin() + std::min(begin, calls.size()), calls.end(), std::uint64_t{0},
                         [](std::uint64_t sum, const auto& call) {
                           return sum + call.elapsed_us;
                         });
}

std::size_t SumAcquiredLabels(const kea::distributed::detail::TrainingExecutionTiming& timing) {
  return std::accumulate(timing.rounds.begin(), timing.rounds.end(), std::size_t{0},
                         [](std::size_t sum, const auto& round) {
                           return sum + round.acquired_labels;
                         });
}

std::string SamplerName(bool cluster) {
  return cluster ? "cluster" : "random";
}

void WriteHeader(std::ofstream& output) {
  output << "dataset,sampler,label_budget,rounds,initial_label_fraction,seed,rows,embedding_dimensions,"
         << "data_setup_ms,initial_sampling_ms,initial_fetching_ms,initial_labeling_ms,"
         << "initial_training_ms,recursive_sampling_ms,recursive_fetching_ms,recursive_labeling_ms,"
         << "recursive_training_ms,phase_a_total_ms,phase_a_labels,proxy_scoring_ms,proxy_accuracy,aurac,"
         << "initial_selection_query_ms,initial_selection_materialization_ms,initial_matrix_copy_ms,"
         << "initial_kmeans_ms,initial_representative_selection_ms,initial_selected_fetch_query_ms,"
         << "initial_selected_fetch_materialization_ms,recursive_unlabeled_query_ms,"
         << "recursive_unlabeled_materialization_ms,recursive_scoring_ms,recursive_sort_and_copy_ms,"
         << "band_02_escalations,band_02_final_accuracy,band_02_total_calls,"
         << "band_01_escalations,band_01_final_accuracy,band_01_total_calls,"
         << "band_02_hosted_75rps_seconds,band_02_server_250rps_seconds,"
         << "band_02_inprocess_1800rps_seconds\n";
}

void RunDatasetSweep(
    const std::string& dataset_name,
    kea::test::SemBenchDataset dataset_kind,
    const std::vector<std::size_t>& random_budgets,
    const std::vector<std::size_t>& cluster_budgets,
    const std::vector<std::size_t>& rounds_to_run,
    const std::vector<double>& recursive_initial_fractions,
    const std::vector<std::uint64_t>& seeds,
    std::ofstream& output) {
  duckdb::DuckDB database(nullptr);
  duckdb::Connection connection(database);
  const auto setup_start = Clock::now();
  const auto examples = kea::test::LoadSemBenchExamples(
      connection,
      kea::test::FindSemBenchDataset(std::filesystem::path(KEA_SEMBENCH_DIR), dataset_kind),
      /*maximum_rows=*/3000);
  kea::test::CreateExamplesTable(connection, examples);
  const std::uint64_t data_setup_us = ElapsedUs(setup_start);
  const auto labels = kea::test::LabelsById(examples);

  std::size_t completed = 0;
  const std::size_t configurations_per_budget = 1 +
      (rounds_to_run.size() - std::count(rounds_to_run.begin(), rounds_to_run.end(), 1)) *
          recursive_initial_fractions.size();
  const std::size_t total_runs =
      (random_budgets.size() + cluster_budgets.size()) * configurations_per_budget * seeds.size();
  for (const bool cluster : {false, true}) {
    const std::vector<std::size_t>& budgets = cluster ? cluster_budgets : random_budgets;
    for (const std::size_t budget : budgets) {
      for (const std::size_t rounds : rounds_to_run) {
        const std::vector<double> fractions =
            rounds == 1 ? std::vector<double>{1.0} : recursive_initial_fractions;
        for (const double fraction : fractions) {
          for (const std::uint64_t seed : seeds) {
            std::unique_ptr<kea::ISampler> base_sampler;
            if (cluster) {
              kea::ClusterSamplingOptions options;
              options.cluster_count = budget;
              options.max_iterations = 10;
              base_sampler = std::make_unique<kea::ClusterSampler>(options);
            } else {
              base_sampler = std::make_unique<kea::RandomSampler>();
            }
            TimedSampler sampler(*base_sampler);
            TimedOracleLabeler labeler(labels);
            auto worker = std::make_shared<kea::distributed::detail::DuckDbShardWorker>(
                "local", connection, sampler, labeler);
            kea::distributed::SingleMachineBackend backend(worker);
            kea::RunConfig config;
            config.dataset.table_name = "examples";
            config.rounds = rounds;
            config.label_budget = budget;
            config.initial_label_fraction = fraction;
            config.seed = seed;

            kea::distributed::detail::TrainingExecutionTiming timing;
            const kea::LogisticRegressionTrainer trainer;
            const kea::ProxyModel model = kea::distributed::detail::RunCleanCentralTraining(
                config, backend, trainer, &timing);
            const EvaluationResult evaluation = EvaluateProxy(model, examples);
            const auto& detailed_timing = worker->timing();

            const auto& initial_round = timing.rounds.front();
            const auto& label_calls = labeler.calls();
            const std::uint64_t initial_label_us = label_calls.empty() ? 0 : label_calls.front().elapsed_us;
            const std::uint64_t recursive_label_us = SumLabelUs(label_calls, /*begin=*/1);
            const std::uint64_t recursive_sampling_us = SumSamplingUs(timing, /*begin=*/1);
            const std::uint64_t recursive_fetching_us = SumFetchingUs(timing, /*begin=*/1);
            const std::uint64_t training_us = SumTrainingUs(timing);
            const std::uint64_t initial_training_us = initial_round.training_us;
            const std::uint64_t recursive_training_us = training_us - initial_training_us;
            const std::size_t phase_a_labels = SumAcquiredLabels(timing);
            const std::size_t narrow_total_calls = phase_a_labels + evaluation.narrow_band.escalations;
            const std::size_t wide_total_calls = phase_a_labels + evaluation.wide_band.escalations;

            output << dataset_name << ',' << SamplerName(cluster) << ',' << budget << ',' << rounds << ','
                   << std::fixed << std::setprecision(3) << fraction << ',' << seed << ',' << examples.size() << ','
                   << examples.front().candidate.embedding.size() << ','
                   << std::fixed << std::setprecision(3)
                   << UsToMs(data_setup_us) << ',' << UsToMs(initial_round.sampling_us) << ','
                   << UsToMs(initial_round.fetching_us) << ',' << UsToMs(initial_label_us) << ','
                   << UsToMs(initial_training_us) << ',' << UsToMs(recursive_sampling_us) << ','
                   << UsToMs(recursive_fetching_us) << ','
                   << UsToMs(recursive_label_us) << ',' << UsToMs(recursive_training_us) << ','
                   << UsToMs(timing.total_us) << ',' << phase_a_labels << ','
                   << UsToMs(evaluation.scoring_us) << ',' << evaluation.proxy_accuracy << ','
                   << evaluation.aurac << ','
                   << UsToMs(detailed_timing.initial_sampling.query_us) << ','
                   << UsToMs(detailed_timing.initial_sampling.materialization_us) << ','
                   << UsToMs(detailed_timing.initial_sampling.matrix_copy_us) << ','
                   << UsToMs(detailed_timing.initial_sampling.kmeans_us) << ','
                   << UsToMs(detailed_timing.initial_sampling.representative_selection_us) << ','
                   << UsToMs(detailed_timing.initial_selected_fetch.query_us) << ','
                   << UsToMs(detailed_timing.initial_selected_fetch.materialization_us) << ','
                   << UsToMs(detailed_timing.recursive_unlabeled_fetch.query_us) << ','
                   << UsToMs(detailed_timing.recursive_unlabeled_fetch.materialization_us) << ','
                   << UsToMs(detailed_timing.recursive_scoring_us) << ','
                   << UsToMs(detailed_timing.recursive_sort_and_copy_us) << ','
                   << evaluation.narrow_band.escalations << ','
                   << evaluation.narrow_band.final_accuracy << ',' << narrow_total_calls << ','
                   << evaluation.wide_band.escalations << ',' << evaluation.wide_band.final_accuracy << ','
                   << wide_total_calls << ',' << static_cast<double>(narrow_total_calls) / 75.0 << ','
                   << static_cast<double>(narrow_total_calls) / 250.0 << ','
                   << static_cast<double>(narrow_total_calls) / 1800.0 << '\n';
            output.flush();
            ++completed;
            std::cerr << '[' << dataset_name << "] completed " << completed << '/' << total_runs << '\n';
          }
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path output_path = argc > 1
      ? std::filesystem::path(argv[1])
      : std::filesystem::path("/private/tmp/kea_sembench_single_machine_sweep.csv");
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Unable to open output file " + output_path.string());
  }
  // Use equal budgets for a direct random-versus-cluster comparison.
  std::vector<std::size_t> random_budgets = {32, 64, 128};
  std::vector<std::size_t> cluster_budgets = {32, 64, 128};
  std::vector<std::size_t> rounds = {1, 3, 5};
  std::vector<double> recursive_initial_fractions = {0.2};
  std::vector<std::uint64_t> seeds = {42, 43, 44};

  // A fast representative estimate: one random and one clustered budget,
  // all requested round counts, one seed, on each dataset.
  if (argc > 2 && std::string(argv[2]) == "--sample") {
    random_budgets = {64};
    cluster_budgets = {64};
    seeds = {42};
  }
  WriteHeader(output);
  RunDatasetSweep("movie", kea::test::SemBenchDataset::Movie, random_budgets, cluster_budgets,
                  rounds, recursive_initial_fractions, seeds, output);
  RunDatasetSweep("fever", kea::test::SemBenchDataset::Fever, random_budgets, cluster_budgets,
                  rounds, recursive_initial_fractions, seeds, output);
  std::cerr << "Wrote results to " << output_path << '\n';
}
