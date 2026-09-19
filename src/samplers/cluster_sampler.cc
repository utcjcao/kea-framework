#include "kea/samplers/cluster_sampler.h"

#include <algorithm>
#include <random>
#include <stdexcept>

namespace kea {

ClusterSampler::ClusterSampler(ClusterSamplingOptions options) : options_(options) {
  if (options_.cluster_count == 0 || options_.max_iterations == 0) {
    throw std::invalid_argument("ClusterSampler requires positive cluster_count and max_iterations");
  }
}

std::vector<RowId> ClusterSampler::SelectInitial(
    IInitialSamplingContext& context,
    std::size_t budget,
    std::uint64_t seed) {
  if (budget == 0) {
    return {};
  }

  // Preserve the configured cluster count. In recursive runs this lets the
  // caller cluster at the total-budget granularity, then label only the
  // round-0 quota of representatives. That matches GC1-rec semantics.
  std::vector<RowId> representatives =
      context.SelectClusterRepresentativeIds(options_, seed);
  if (representatives.size() <= budget) {
    return representatives;
  }

  // K-means returns one representative per cluster. Choose an even,
  // reproducible subset of those representatives for the initial labels;
  // later rounds are still selected by uncertainty in the framework.
  std::sort(representatives.begin(), representatives.end());
  std::mt19937_64 generator(seed);
  std::shuffle(representatives.begin(), representatives.end(), generator);
  representatives.resize(budget);
  return representatives;
}

}  // namespace kea
