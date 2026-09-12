#pragma once

#include <string>
#include <string_view>

namespace kea::detail {

// Quotes a simple DuckDB identifier after validating the supported character
// set. Qualified identifiers such as schema.table are handled separately.
[[nodiscard]] std::string QuoteDuckDbIdentifier(std::string_view identifier);

[[nodiscard]] std::string QuoteDuckDbQualifiedIdentifier(
    std::string_view identifier);

[[nodiscard]] std::string QuoteDuckDbStringLiteral(std::string_view value);

}  // namespace kea::detail
