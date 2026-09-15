cmake_minimum_required(VERSION 3.20)

project(kea_framework LANGUAGES CXX)

option(KEA_FRAMEWORK_BUILD_TESTS "Build kea-framework tests" ON)
option(KEA_FRAMEWORK_BUILD_SEMBENCH_E2E_TESTS
  "Build end-to-end tests using a local SemBench checkout" ON)
set(KEA_SEMBENCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SemBench" CACHE PATH
  "Path to a local SemBench checkout")

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
  src/distributed/duckdb_shard_worker.cc
  src/distributed/in_process_execution_backend.cc
  src/distributed/run_config.cc
  src/distributed/training_execution.cc
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

  add_executable(distributed_foundation_test
    tests/distributed_foundation_test.cc
  )
  target_link_libraries(distributed_foundation_test PRIVATE kea_framework)
  target_include_directories(distributed_foundation_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME distributed_foundation_test
           COMMAND distributed_foundation_test)

  if(KEA_FRAMEWORK_BUILD_SEMBENCH_E2E_TESTS AND EXISTS "${KEA_SEMBENCH_DIR}")
    find_package(ZLIB REQUIRED)
    add_library(kea_sembench_test_support STATIC tests/sembench_test_data.cc)
    target_compile_features(kea_sembench_test_support PRIVATE cxx_std_17)
    target_link_libraries(kea_sembench_test_support PRIVATE ZLIB::ZLIB)
    target_include_directories(kea_sembench_test_support
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${DUCKDB_INCLUDE_DIR}
    )
    foreach(test_target proxy_training_runner_test distributed_foundation_test)
      target_link_libraries(${test_target} PRIVATE kea_sembench_test_support)
      target_compile_definitions(${test_target} PRIVATE
        KEA_HAS_SEMBENCH=1
        KEA_SEMBENCH_DIR="${KEA_SEMBENCH_DIR}"
      )
    endforeach()
  elseif(KEA_FRAMEWORK_BUILD_SEMBENCH_E2E_TESTS)
    message(STATUS "SemBench not found at ${KEA_SEMBENCH_DIR}; SemBench integration checks are skipped")
  endif()
endif()
