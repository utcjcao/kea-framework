#include "kea/samplers/cluster_sampler.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace {

class RecordingContext final : public kea::IInitialSamplingContext {
 public:
  std::vector<kea::RowId> SelectRandomIds(std::size_t, std::uint64_t) override {
    return {};
  }

  std::vector<kea::RowId> SelectClusterRepresentativeIds(
      const kea::ClusterSamplingOptions& options,
      std::uint64_t) override {
    received_options = options;
    return {"rep-0", "rep-1", "rep-2", "rep-3", "rep-4", "rep-5"};
  }

  kea::ClusterSamplingOptions received_options;
};

}  // namespace

int main() {
  kea::ClusterSamplingOptions options;
  options.cluster_count = 6;
  options.max_iterations = 10;
  kea::ClusterSampler sampler(options);

  RecordingContext context;
  const std::vector<kea::RowId> initial = sampler.SelectInitial(context, /*budget=*/2, /*seed=*/42);
  assert(context.received_options.cluster_count == 6);
  assert(context.received_options.max_iterations == 10);
  assert(initial.size() == 2);

  const std::vector<kea::RowId> repeated = sampler.SelectInitial(context, /*budget=*/2, /*seed=*/42);
  assert(initial == repeated);

  const std::vector<kea::RowId> all = sampler.SelectInitial(context, /*budget=*/6, /*seed=*/42);
  assert(all.size() == 6);
}
