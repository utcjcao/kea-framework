#pragma once

#include "kea/proxy_training.h"

namespace kea {

// Centralized k-means initial sampling. One candidate nearest each cluster
// centroid is selected and later rounds remain framework-owned uncertainty
// sampling.
class ClusterSampler final : public ISampler {
 public:
  explicit ClusterSampler(ClusterSamplingOptions options);

  [[nodiscard]] std::vector<RowId> SelectInitial(
      IInitialSamplingContext& context,
      std::size_t budget,
      std::uint64_t seed) override;

 private:
  ClusterSamplingOptions options_;
};

}  // namespace kea
