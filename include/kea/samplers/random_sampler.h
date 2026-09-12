#pragma once

#include "kea/proxy_training.h"

namespace kea {

// Selects the initial label batch through the context's deterministic random
// selection operation.
class RandomSampler final : public ISampler {
 public:
  RandomSampler() = default;

  [[nodiscard]] std::vector<RowId> SelectInitial(
      IInitialSamplingContext& context,
      std::size_t budget,
      std::uint64_t seed) override;
};

}  // namespace kea
