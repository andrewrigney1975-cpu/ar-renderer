#pragma once

#include <atomic>
#include <cstdint>
#include <functional>

namespace pr {

void SetThreadCount(int n);  // 0 = hardware concurrency
int ThreadCount();

// Run func(i) for i in [0, count) across the worker threads; chunkSize indices are claimed at once.
// Blocks until all iterations complete. The calling thread participates.
void ParallelFor(int64_t count, const std::function<void(int64_t index, int threadIndex)> &func,
                 int64_t chunkSize = 1);

} // namespace pr
