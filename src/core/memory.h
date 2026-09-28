#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace pr {

// Bump allocator for short-lived per-sample objects (BxDFs, etc.). Objects allocated here must be
// trivially destructible; Reset() releases everything at once.
class ScratchBuffer {
  public:
    explicit ScratchBuffer(size_t size = 256 * 1024) : size_(size) {
        ptr_ = static_cast<char *>(::operator new(size_, std::align_val_t(64)));
    }
    ~ScratchBuffer() {
        Reset();
        ::operator delete(ptr_, std::align_val_t(64));
    }
    ScratchBuffer(const ScratchBuffer &) = delete;
    ScratchBuffer &operator=(const ScratchBuffer &) = delete;

    void *Alloc(size_t size, size_t align) {
        if ((offset_ % align) != 0) offset_ += align - (offset_ % align);
        if (offset_ + size > size_) Grow(size, align);
        void *p = ptr_ + offset_;
        offset_ += size;
        return p;
    }
    template <typename T, typename... Args> T *New(Args &&...args) {
        void *mem = Alloc(sizeof(T), alignof(T));
        return new (mem) T(std::forward<Args>(args)...);
    }
    void Reset() {
        for (auto &b : smallBuffers_) ::operator delete(b.first, std::align_val_t(64));
        smallBuffers_.clear();
        offset_ = 0;
    }

  private:
    void Grow(size_t size, size_t align) {
        // Keep the current block alive until Reset(), and continue in a new larger block.
        smallBuffers_.push_back({ptr_, size_});
        size_ = std::max(2 * size + align, size_ * 2);
        ptr_ = static_cast<char *>(::operator new(size_, std::align_val_t(64)));
        offset_ = 0;
    }
    char *ptr_ = nullptr;
    size_t size_ = 0, offset_ = 0;
    std::vector<std::pair<char *, size_t>> smallBuffers_;
};

} // namespace pr
