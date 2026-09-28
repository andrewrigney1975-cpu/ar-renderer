#include "core/log.h"

#include <cstdio>
#include <mutex>

namespace pr {

static LogLevel gLevel = LogLevel::Info;
static std::mutex gLogMutex;

void SetLogLevel(LogLevel level) { gLevel = level; }

void LogMessage(LogLevel level, const std::string &msg) {
    if (level < gLevel) return;
    static const char *prefix[] = {"[verbose] ", "", "[warning] ", "[error] "};
    std::lock_guard<std::mutex> lock(gLogMutex);
    std::fprintf(stderr, "%s%s\n", prefix[int(level)], msg.c_str());
    std::fflush(stderr);
}

void WriteStdoutLine(const std::string &line) {
    std::lock_guard<std::mutex> lock(gLogMutex);
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

} // namespace pr
