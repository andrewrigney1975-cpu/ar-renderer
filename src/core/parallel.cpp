#include "core/parallel.h"

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace pr {

namespace {

int gThreadCount = 0;

// Index of the current thread inside a running ParallelFor job (-1 outside jobs). Nested
// ParallelFor calls from inside a job run serially on the calling thread.
thread_local int tJobThread = -1;

// Persistent worker pool: threads are created once and sleep between jobs, so frequent
// ParallelFor calls (one per progressive pass or MLT round) cost a wake-up instead of 24 thread
// creations. Work is handed out dynamically from an atomic counter, which balances P- and E-cores.
class ThreadPool {
  public:
    explicit ThreadPool(int nWorkers) {
        for (int i = 0; i < nWorkers; ++i) workers_.emplace_back([this, i] { WorkerLoop(i + 1); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto &t : workers_) t.join();
    }
    int Size() const { return int(workers_.size()) + 1; }

    void Run(int64_t count, const std::function<void(int64_t, int)> &func, int64_t chunk, int nThreads) {
        std::lock_guard<std::mutex> jobLock(jobMutex_);  // one job at a time
        {
            std::lock_guard<std::mutex> lock(m_);
            func_ = &func;
            count_ = count;
            chunk_ = chunk;
            next_.store(0);
            participants_ = std::min(nThreads, Size());
            running_ = participants_ - 1;  // workers that still have to finish
            ++jobId_;
        }
        wake_.notify_all();
        Work(0);
        std::unique_lock<std::mutex> lock(m_);
        done_.wait(lock, [this] { return running_ == 0; });
        func_ = nullptr;
    }

  private:
    void Work(int threadIndex) {
        tJobThread = threadIndex;
        while (true) {
            int64_t start = next_.fetch_add(chunk_);
            if (start >= count_) break;
            int64_t end = std::min(count_, start + chunk_);
            for (int64_t i = start; i < end; ++i) (*func_)(i, threadIndex);
        }
        tJobThread = -1;
    }

    void WorkerLoop(int threadIndex) {
        uint64_t seen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(m_);
                wake_.wait(lock, [&] { return stop_ || jobId_ != seen; });
                if (stop_) return;
                seen = jobId_;
                if (threadIndex >= participants_) continue;  // not needed for this job
            }
            Work(threadIndex);
            {
                std::lock_guard<std::mutex> lock(m_);
                if (--running_ == 0) done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_, jobMutex_;
    std::condition_variable wake_, done_;
    const std::function<void(int64_t, int)> *func_ = nullptr;
    int64_t count_ = 0, chunk_ = 1;
    std::atomic<int64_t> next_{0};
    int participants_ = 0, running_ = 0;
    uint64_t jobId_ = 0;
    bool stop_ = false;
};

std::mutex gPoolMutex;
std::unique_ptr<ThreadPool> gPool;

ThreadPool &Pool() {
    std::lock_guard<std::mutex> lock(gPoolMutex);
    int want = ThreadCount();
    if (!gPool || gPool->Size() != want) {
        gPool.reset();
        gPool = std::make_unique<ThreadPool>(want - 1);
    }
    return *gPool;
}

} // namespace

void SetThreadCount(int n) { gThreadCount = n; }

int ThreadCount() {
    if (gThreadCount > 0) return gThreadCount;
    return std::max(1, int(std::thread::hardware_concurrency()));
}

void ParallelFor(int64_t count, const std::function<void(int64_t, int)> &func, int64_t chunkSize) {
    if (count <= 0) return;
    chunkSize = std::max<int64_t>(1, chunkSize);
    int nThreads = int(std::min<int64_t>(ThreadCount(), (count + chunkSize - 1) / chunkSize));
    if (nThreads <= 1 || tJobThread >= 0) {
        // Single chunk, or nested inside a running job: run inline on this thread.
        int ti = std::max(0, tJobThread);
        for (int64_t i = 0; i < count; ++i) func(i, ti);
        return;
    }
    Pool().Run(count, func, chunkSize, nThreads);
}

} // namespace pr
