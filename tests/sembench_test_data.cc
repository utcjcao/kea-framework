#include "sembench_test_data.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <zlib.h>

#include "duckdb.hpp"

namespace kea::test {
namespace {

struct ZipEntry {
  std::uint32_t local_header_offset = 0;
  std::uint32_t compressed_size = 0;
  std::uint32_t uncompressed_size = 0;
};

struct NpyArray {
  std::string header;
  std::size_t data_offset = 0;
};

std::uint16_t ReadU16(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
  if (offset + 2 > bytes.size()) {
    throw std::runtime_error("Unexpected end of NPZ archive");
  }
  return static_cast<std::uint16_t>(bytes[offset]) |
      (static_cast<std::uint16_t>(bytes[offset + 1]) << 8U);
}

std::uint32_t ReadU32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
  if (offset + 4 > bytes.size()) {
    throw std::runtime_error("Unexpected end of NPZ archive");
  }
  return static_cast<std::uint32_t>(bytes[offset]) |
      (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
      (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
      (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
}

std::vector<std::uint8_t> ReadFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Unable to open " + path.string());
  }
  const std::streamsize size = input.tellg();
  if (size < 0) {
    throw std::runtime_error("Unable to measure " + path.string());
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char*>(bytes.data()), size)) {
    throw std::runtime_error("Unable to read " + path.string());
  }
  return bytes;
}

std::unordered_map<std::string, ZipEntry> ReadZipDirectory(
    const std::vector<std::uint8_t>& archive) {
  constexpr std::uint32_t kEndOfCentralDirectory = 0x06054b50;
  constexpr std::uint32_t kCentralDirectoryHeader = 0x02014b50;
  if (archive.size() < 4) {
    throw std::runtime_error("NPZ archive is too small");
  }
  const std::size_t search_start = archive.size() > 65557 ? archive.size() - 65557 : 0;
  std::size_t end = archive.size() - 4;
  while (true) {
    if (ReadU32(archive, end) == kEndOfCentralDirectory) {
      break;
    }
    if (end == search_start) {
      throw std::runtime_error("NPZ archive has no central directory");
    }
    --end;
  }

  const std::uint16_t entry_count = ReadU16(archive, end + 10);
  std::size_t offset = ReadU32(archive, end + 16);
  std::unordered_map<std::string, ZipEntry> entries;
  for (std::uint16_t entry = 0; entry < entry_count; ++entry) {
    if (ReadU32(archive, offset) != kCentralDirectoryHeader) {
      throw std::runtime_error("Malformed NPZ central-directory entry");
    }
    const std::uint16_t name_length = ReadU16(archive, offset + 28);
    const std::uint16_t extra_length = ReadU16(archive, offset + 30);
    const std::uint16_t comment_length = ReadU16(archive, offset + 32);
    const std::size_t name_offset = offset + 46;
    if (name_offset + name_length > archive.size()) {
      throw std::runtime_error("Malformed NPZ entry name");
    }
    entries.emplace(
        std::string(reinterpret_cast<const char*>(archive.data() + name_offset), name_length),
        ZipEntry{ReadU32(archive, offset + 42), ReadU32(archive, offset + 20),
                 ReadU32(archive, offset + 24)});
    offset = name_offset + name_length + extra_length + comment_length;
  }
  return entries;
}

std::vector<std::uint8_t> InflateZipEntry(
    const std::vector<std::uint8_t>& archive,
    const ZipEntry& entry) {
  constexpr std::uint32_t kLocalFileHeader = 0x04034b50;
  const std::size_t header = entry.local_header_offset;
  if (ReadU32(archive, header) != kLocalFileHeader) {
    throw std::runtime_error("Malformed NPZ local-file header");
  }
  const std::size_t data_offset = header + 30 + ReadU16(archive, header + 26) +
      ReadU16(archive, header + 28);
  if (data_offset + entry.compressed_size > archive.size()) {
    throw std::runtime_error("Malformed NPZ compressed payload");
  }

  std::vector<std::uint8_t> output(entry.uncompressed_size);
  z_stream stream{};
  stream.next_in = const_cast<Bytef*>(archive.data() + data_offset);
  stream.avail_in = entry.compressed_size;
  stream.next_out = output.data();
  stream.avail_out = entry.uncompressed_size;
  if (inflateInit2(&stream, -MAX_WBITS) != Z_OK) {
    throw std::runtime_error("Unable to initialize NPZ decompressor");
  }
  const int result = inflate(&stream, Z_FINISH);
  inflateEnd(&stream);
  if (result != Z_STREAM_END || stream.total_out != entry.uncompressed_size) {
    throw std::runtime_error("Unable to decompress NPZ entry");
  }
  return output;
}

NpyArray ParseNpyArray(const std::vector<std::uint8_t>& bytes) {
  constexpr std::array<std::uint8_t, 6> kMagic = {0x93, 'N', 'U', 'M', 'P', 'Y'};
  if (bytes.size() < 10 || !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
    throw std::runtime_error("NPZ entry is not a NPY array");
  }
  if (bytes[6] != 1 || bytes[7] != 0) {
    throw std::runtime_error("Only NPY v1.0 arrays are supported in SemBench tests");
  }
  const std::size_t header_size = ReadU16(bytes, 8);
  const std::size_t data_offset = 10 + header_size;
  if (data_offset > bytes.size()) {
    throw std::runtime_error("Malformed NPY header");
  }
  return {std::string(reinterpret_cast<const char*>(bytes.data() + 10), header_size), data_offset};
}

std::size_t ReadDimension(const std::string& header, std::size_t position) {
  const std::size_t end = header.find_first_of(",)", position);
  if (end == std::string::npos) {
    throw std::runtime_error("Malformed NPY shape");
  }
  return static_cast<std::size_t>(std::stoull(header.substr(position, end - position)));
}

std::pair<std::size_t, std::size_t> ParseFloatMatrixShape(const NpyArray& array) {
  if (array.header.find("'descr': '<f4'") == std::string::npos ||
      array.header.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error("SemBench embeddings must be C-order little-endian float32 arrays");
  }
  const std::size_t shape = array.header.find("'shape': (");
  if (shape == std::string::npos) {
    throw std::runtime_error("NPY array has no shape");
  }
  const std::size_t rows_start = shape + std::string("'shape': (").size();
  const std::size_t rows = ReadDimension(array.header, rows_start);
  const std::size_t columns_start = array.header.find(',', rows_start) + 1;
  return {rows, ReadDimension(array.header, columns_start)};
}

std::size_t ParseIntVectorLength(const NpyArray& array) {
  if (array.header.find("'descr': '<i8'") == std::string::npos ||
      array.header.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error("SemBench row indexes must be C-order little-endian int64 arrays");
  }
  const std::size_t shape = array.header.find("'shape': (");
  if (shape == std::string::npos) {
    throw std::runtime_error("NPY array has no shape");
  }
  return ReadDimension(array.header, shape + std::string("'shape': (").size());
}

std::string QuoteSqlString(const std::filesystem::path& path) {
  std::string quoted = path.string();
  std::size_t position = 0;
  while ((position = quoted.find('\'', position)) != std::string::npos) {
    quoted.insert(position, "'");
    position += 2;
  }
  return "'" + quoted + "'";
}

void RequireSuccess(const duckdb::MaterializedQueryResult& result, std::string_view action) {
  if (result.HasError()) {
    throw std::runtime_error(std::string(action) + ": " + result.GetError());
  }
}

}  // namespace

SemBenchDatasetPaths FindSemBenchDataset(
    const std::filesystem::path& sembench_root,
    SemBenchDataset dataset) {
  switch (dataset) {
    case SemBenchDataset::Movie: {
      const auto data = sembench_root / "files/movie/data/sf_6206_balanced";
      return {data / "Reviews.csv",
              data / "embeddings/gte-large-en-v1.5/review_embeddings.npz"};
    }
    case SemBenchDataset::Fever: {
      const auto data = sembench_root / "files/fever/data/sf_6206_balanced";
      return {data / "Claims.csv",
              data / "embeddings/gte-large-en-v1.5/claim_embeddings.npz"};
    }
  }
  throw std::invalid_argument("Unknown SemBench dataset");
}

std::vector<SemBenchExample> LoadSemBenchExamples(
    duckdb::Connection& connection,
    const SemBenchDatasetPaths& dataset,
    std::size_t maximum_rows) {
  if (maximum_rows == 0) {
    throw std::invalid_argument("SemBench test must load at least one row");
  }
  const std::vector<std::uint8_t> archive = ReadFile(dataset.embeddings);
  const auto entries = ReadZipDirectory(archive);
  const auto embeddings_entry = entries.find("embeddings.npy");
  const auto row_index_entry = entries.find("row_index.npy");
  if (embeddings_entry == entries.end() || row_index_entry == entries.end()) {
    throw std::runtime_error("SemBench embedding archive needs embeddings.npy and row_index.npy");
  }
  const std::vector<std::uint8_t> embeddings_bytes = InflateZipEntry(archive, embeddings_entry->second);
  const std::vector<std::uint8_t> row_index_bytes = InflateZipEntry(archive, row_index_entry->second);
  const NpyArray embeddings = ParseNpyArray(embeddings_bytes);
  const NpyArray row_indexes = ParseNpyArray(row_index_bytes);
  const auto [row_count, dimensions] = ParseFloatMatrixShape(embeddings);
  if (ParseIntVectorLength(row_indexes) != row_count ||
      embeddings.data_offset + row_count * dimensions * sizeof(float) != embeddings_bytes.size() ||
      row_indexes.data_offset + row_count * sizeof(std::int64_t) != row_index_bytes.size()) {
    throw std::runtime_error("SemBench NPZ array sizes do not agree");
  }

  auto source_rows = connection.Query(
      "SELECT row_number() OVER () - 1, reviewText, scoreSentiment FROM read_csv_auto(" +
      QuoteSqlString(dataset.csv) + ")");
  RequireSuccess(*source_rows, "Unable to read SemBench CSV");
  std::unordered_map<std::int64_t, std::pair<std::string, int>> metadata;
  metadata.reserve(source_rows->RowCount());
  for (duckdb::idx_t row = 0; row < source_rows->RowCount(); ++row) {
    const std::string sentiment = source_rows->GetValue(2, row).ToString();
    if (sentiment != "POSITIVE" && sentiment != "NEGATIVE") {
      throw std::runtime_error("Unexpected SemBench scoreSentiment value: " + sentiment);
    }
    metadata.emplace(source_rows->GetValue(0, row).GetValue<std::int64_t>(),
                     std::make_pair(source_rows->GetValue(1, row).ToString(),
                                    sentiment == "POSITIVE" ? 1 : 0));
  }

  std::vector<SemBenchExample> examples;
  examples.reserve(std::min(maximum_rows, row_count));
  for (std::size_t row = 0; row < row_count && examples.size() < maximum_rows; ++row) {
    std::int64_t row_index = 0;
    std::memcpy(&row_index, row_index_bytes.data() + row_indexes.data_offset + row * sizeof(row_index),
                sizeof(row_index));
    const auto source = metadata.find(row_index);
    if (source == metadata.end()) {
      throw std::runtime_error("Embedding row_index was not found in SemBench CSV");
    }
    SemBenchExample example;
    example.candidate.id = std::to_string(row_index);
    example.candidate.text = source->second.first;
    example.candidate.embedding.resize(dimensions);
    std::memcpy(example.candidate.embedding.data(),
                embeddings_bytes.data() + embeddings.data_offset + row * dimensions * sizeof(float),
                dimensions * sizeof(float));
    example.label = source->second.second;
    examples.push_back(std::move(example));
  }
  return examples;
}

void CreateExamplesTable(
    duckdb::Connection& connection,
    const std::vector<SemBenchExample>& examples) {
  auto create = connection.Query(
      "CREATE TABLE examples (id VARCHAR PRIMARY KEY, text VARCHAR NOT NULL, embedding FLOAT[] NOT NULL)");
  RequireSuccess(*create, "Unable to create SemBench test table");

  duckdb::Appender appender(connection, "examples");
  for (const SemBenchExample& example : examples) {
    std::vector<duckdb::Value> values;
    values.reserve(example.candidate.embedding.size());
    for (const float value : example.candidate.embedding) {
      values.emplace_back(value);
    }
    appender.BeginRow();
    appender.Append(example.candidate.id.c_str());
    appender.Append(example.candidate.text.c_str());
    appender.Append(duckdb::Value::LIST(duckdb::LogicalType::FLOAT, std::move(values)));
    appender.EndRow();
  }
  appender.Close();
}

std::unordered_map<RowId, int> LabelsById(
    const std::vector<SemBenchExample>& examples) {
  std::unordered_map<RowId, int> labels;
  labels.reserve(examples.size());
  for (const SemBenchExample& example : examples) {
    labels.emplace(example.candidate.id, example.label);
  }
  return labels;
}

}  // namespace kea::test
