#include "kea/proxy_training.h"

#include <stdexcept>

#include <armadillo>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include "detail/logistic_regression_helpers.h"

namespace kea {

float ProxyModel::PredictProbability(const Embedding& embedding) const {
  if (embedding.size() != weights.size()) {
    throw std::invalid_argument("Embedding dimension does not match model weights");
  }

  float logit = intercept;
  for (std::size_t index = 0; index < embedding.size(); ++index) {
    logit += weights[index] * embedding[index];
  }
  return detail::Sigmoid(logit);
}

LogisticRegressionTrainer::LogisticRegressionTrainer(
    LogisticRegressionOptions options)
    : options_(options) {
  detail::ValidateLogisticRegressionOptions(options_);
}

ProxyModel LogisticRegressionTrainer::Train(
    const std::vector<LabeledExample>& examples) const {
  detail::ValidateLabeledExamples(examples);

  const std::size_t dimensions = examples.front().candidate.embedding.size();
  arma::mat features(dimensions, examples.size());
  arma::Row<std::size_t> labels(examples.size());
  for (std::size_t column = 0; column < examples.size(); ++column) {
    const LabeledExample& example = examples[column];
    labels[column] = static_cast<std::size_t>(example.label);
    for (std::size_t row = 0; row < dimensions; ++row) {
      features(row, column) = static_cast<double>(example.candidate.embedding[row]);
    }
  }

  mlpack::LogisticRegression<> fitted_model(
      features, labels, static_cast<double>(options_.l2_regularization));
  const auto& parameters = fitted_model.Parameters();
  if (parameters.n_elem != dimensions + 1) {
    throw std::runtime_error("mlpack returned an unexpected logistic-regression parameter size");
  }

  ProxyModel model;
  model.intercept = static_cast<float>(parameters[0]);
  model.weights.reserve(dimensions);
  for (std::size_t index = 0; index < dimensions; ++index) {
    model.weights.push_back(static_cast<float>(parameters[index + 1]));
  }
  return model;
}

}  // namespace kea
