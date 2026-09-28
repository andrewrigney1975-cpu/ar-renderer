#pragma once

#include <format>
#include <string>

namespace pr {

enum class LogLevel { Verbose, Info, Warning, Error };

void SetLogLevel(LogLevel level);
void LogMessage(LogLevel level, const std::string &msg);

template <typename... Args> void LogVerbose(std::format_string<Args...> fmt, Args &&...args) {
    LogMessage(LogLevel::Verbose, std::format(fmt, std::forward<Args>(args)...));
}
template <typename... Args> void LogInfo(std::format_string<Args...> fmt, Args &&...args) {
    LogMessage(LogLevel::Info, std::format(fmt, std::forward<Args>(args)...));
}
template <typename... Args> void LogWarning(std::format_string<Args...> fmt, Args &&...args) {
    LogMessage(LogLevel::Warning, std::format(fmt, std::forward<Args>(args)...));
}
template <typename... Args> void LogError(std::format_string<Args...> fmt, Args &&...args) {
    LogMessage(LogLevel::Error, std::format(fmt, std::forward<Args>(args)...));
}

// Writes one line to stdout atomically (used for the JSON progress protocol).
void WriteStdoutLine(const std::string &line);

} // namespace pr
