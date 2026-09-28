#pragma once

#include <filesystem>
#include <string>

namespace pr {

inline std::string PathUtf8(const std::filesystem::path &p) {
    auto s = p.u8string();
    return std::string(s.begin(), s.end());
}
inline std::filesystem::path Utf8Path(const std::string &s) {
    return std::filesystem::path(std::u8string(s.begin(), s.end()));
}

} // namespace pr
