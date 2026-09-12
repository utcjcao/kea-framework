#include "kea/samplers/cluster_sampler.h"

#include <algorithm>
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
  ClusterSamplingOptions effective_options = options_;
  effective_options.cluster_count = std::min(effective_options.cluster_count, budget);
  return context.SelectClusterRepresentativeIds(effective_options, seed);
}

}  // namespace kea
