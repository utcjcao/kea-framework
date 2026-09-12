#include "kea/proxy_training.h"

#include <cassert>
#include <vector>

int main() {
  std::vector<kea::LabeledExample> examples = {
      { {"negative-1", "", {-2.0F}}, 0 },
      { {"negative-2", "", {-1.0F}}, 0 },
      { {"positive-1", "", {1.0F}}, 1 },
      { {"positive-2", "", {2.0F}}, 1 },
  };

  kea::LogisticRegressionOptions options;
  options.l2_regularization = 0.01F;
  kea::LogisticRegressionTrainer trainer(options);
  const kea::ProxyModel model = trainer.Train(examples);

  assert(model.PredictProbability({-1.5F}) < 0.5F);
  assert(model.PredictProbability({1.5F}) > 0.5F);
}
