#include "core/parallel.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace pr {

static int gThreadCount = 0;

void SetThreadCount(int n) { gThreadCount = n; }

int ThreadCount() {
    if (gThreadCount > 0) return gThreadCount;
    return std::max(1, int(std::thread::hardware_concurrency()));
}

void ParallelFor(int64_t count, const std::function<void(int64_t, int)> &func, int64_t chunkSize) {
    if (count <= 0) return;
    chunkSize = std::max<int64_t>(1, chunkSize);
    int nThreads = int(std::min<int64_t>(ThreadCount(), (count + chunkSize - 1) / chunkSize));
    std::atomic<int64_t> next{0};
    auto worker = [&](int threadIndex) {
        while (true) {
            int64_t start = next.fetch_add(chunkSize);
            if (start >= count) break;
            int64_t end = std::min(count, start + chunkSize);
            for (int64_t i = start; i < end; ++i) func(i, threadIndex);
        }
    };
    if (nThreads <= 1) {
        worker(0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(nThreads - 1);
    for (int t = 1; t < nThreads; ++t) threads.emplace_back(worker, t);
    worker(0);
    for (auto &t : threads) t.join();
}

} // namespace pr
