#include "kea/samplers/random_sampler.h"

namespace kea {

std::vector<RowId> RandomSampler::SelectInitial(
    IInitialSamplingContext& context,
    std::size_t budget,
    std::uint64_t seed) {
  return context.SelectRandomIds(budget, seed);
}

}  // namespace kea
