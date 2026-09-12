#include "detail/duckdb_sql_helpers.h"

#include <cctype>
#include <stdexcept>

namespace kea::detail {
namespace {

bool IsIdentifierStart(char character) {
  const auto value = static_cast<unsigned char>(character);
  return std::isalpha(value) || character == '_';
}

bool IsIdentifierCharacter(char character) {
  const auto value = static_cast<unsigned char>(character);
  return std::isalnum(value) || character == '_' || character == '$';
}

}  // namespace

std::string QuoteDuckDbIdentifier(std::string_view identifier) {
  if (identifier.empty() || !IsIdentifierStart(identifier.front())) {
    throw std::invalid_argument("DuckDB identifiers must start with a letter or underscore");
  }
  for (const char character : identifier) {
    if (!IsIdentifierCharacter(character)) {
      throw std::invalid_argument("DuckDB identifier contains an unsupported character");
    }
  }
  return "\"" + std::string(identifier) + "\"";
}

std::string QuoteDuckDbQualifiedIdentifier(std::string_view identifier) {
  std::string quoted;
  std::size_t start = 0;
  while (start < identifier.size()) {
    const std::size_t end = identifier.find('.', start);
    const std::string_view part = identifier.substr(start, end - start);
    if (!quoted.empty()) {
      quoted += '.';
    }
    quoted += QuoteDuckDbIdentifier(part);

    if (end == std::string_view::npos) {
      return quoted;
    }
    start = end + 1;
  }
  throw std::invalid_argument("DuckDB table name must not be empty");
}

std::string QuoteDuckDbStringLiteral(std::string_view value) {
  std::string quoted = "'";
  for (const char character : value) {
    quoted += character;
    if (character == '\'') {
      quoted += '\'';
    }
  }
  quoted += '\'';
  return quoted;
}

}  // namespace kea::detail
