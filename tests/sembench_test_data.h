#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "kea/proxy_training.h"

namespace duckdb {
class Connection;
}

namespace kea::test {

enum class SemBenchDataset {
  Movie,
  Fever,
};

struct SemBenchDatasetPaths {
  std::filesystem::path csv;
  std::filesystem::path embeddings;
};

struct SemBenchExample {
  Candidate candidate;
  int label = 0;
};

[[nodiscard]] SemBenchDatasetPaths FindSemBenchDataset(
    const std::filesystem::path& sembench_root,
    SemBenchDataset dataset);

// Loads a bounded prefix of a SemBench CSV/NPZ pair. The NPZ archive supplies
// embeddings and their stable source_row mapping; DuckDB parses the source
// CSV so quoted review and claim text is handled by the real storage engine.
std::vector<SemBenchExample> LoadSemBenchExamples(
    duckdb::Connection& connection,
    const SemBenchDatasetPaths& dataset,
    std::size_t maximum_rows);

void CreateExamplesTable(
    duckdb::Connection& connection,
    const std::vector<SemBenchExample>& examples);

std::unordered_map<RowId, int> LabelsById(
    const std::vector<SemBenchExample>& examples);

}  // namespace kea::test
