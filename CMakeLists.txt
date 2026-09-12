cmake_minimum_required(VERSION 3.20)

project(kea_framework LANGUAGES CXX)

option(KEA_FRAMEWORK_BUILD_TESTS "Build kea-framework tests" ON)

find_package(Armadillo REQUIRED)
find_package(cereal REQUIRED)
find_package(ensmallen REQUIRED)
find_path(MLPACK_INCLUDE_DIR
  NAMES mlpack/methods/logistic_regression/logistic_regression.hpp
  REQUIRED
)
find_path(DUCKDB_INCLUDE_DIR NAMES duckdb.hpp REQUIRED)
find_library(DUCKDB_LIBRARY NAMES duckdb REQUIRED)

add_library(kea_framework
  src/detail/duckdb_sql_helpers.cc
  src/detail/logistic_regression_helpers.cc
  src/detail/proxy_training_helpers.cc
  src/detail/training_data_access.cc
  src/function_labeler.cc
  src/logistic_regression_trainer.cc
  src/proxy_training_runner.cc
  src/samplers/cluster_sampler.cc
  src/samplers/random_sampler.cc
)

target_compile_features(kea_framework PUBLIC cxx_std_17)
target_include_directories(kea_framework
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${DUCKDB_INCLUDE_DIR}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ARMADILLO_INCLUDE_DIRS}
    ${CEREAL_INCLUDE_DIR}
    ${ENSMALLEN_INCLUDE_DIR}
    ${MLPACK_INCLUDE_DIR}
    ${DUCKDB_INCLUDE_DIR}
)
target_link_libraries(kea_framework
  PUBLIC
    ${ARMADILLO_LIBRARIES}
    ${DUCKDB_LIBRARY}
)

if(KEA_FRAMEWORK_BUILD_TESTS)
  enable_testing()
  add_executable(logistic_regression_trainer_test
    tests/logistic_regression_trainer_test.cc
  )
  target_link_libraries(logistic_regression_trainer_test PRIVATE kea_framework)
  add_test(NAME logistic_regression_trainer_test
           COMMAND logistic_regression_trainer_test)

  add_executable(proxy_training_runner_test
    tests/proxy_training_runner_test.cc
  )
  target_link_libraries(proxy_training_runner_test PRIVATE kea_framework)
  add_test(NAME proxy_training_runner_test
           COMMAND proxy_training_runner_test)
endif()
