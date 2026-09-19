#pragma once

#include "kea/proxy_training.h"

namespace kea {

// Centralized k-means initial sampling. K-means produces one candidate per
// configured cluster; if the round-0 label quota is smaller, a deterministic
// subset of those representatives is labeled. Later rounds remain
// framework-owned uncertainty sampling.
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
