#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace pr {

// Minimal little-endian binary serialization for checkpoints.
class BinaryWriter {
  public:
    template <typename T> void Put(const T &v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto *p = reinterpret_cast<const uint8_t *>(&v);
        data_.insert(data_.end(), p, p + sizeof(T));
    }
    void PutBytes(const void *p, size_t n) {
        const auto *b = static_cast<const uint8_t *>(p);
        data_.insert(data_.end(), b, b + n);
    }
    const std::vector<uint8_t> &Data() const { return data_; }

  private:
    std::vector<uint8_t> data_;
};

class BinaryReader {
  public:
    explicit BinaryReader(std::vector<uint8_t> data) : data_(std::move(data)) {}
    template <typename T> bool Get(T *v) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (pos_ + sizeof(T) > data_.size()) return ok_ = false;
        std::memcpy(v, data_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return true;
    }
    bool GetBytes(void *p, size_t n) {
        if (pos_ + n > data_.size()) return ok_ = false;
        std::memcpy(p, data_.data() + pos_, n);
        pos_ += n;
        return true;
    }
    bool Ok() const { return ok_; }
    bool AtEnd() const { return pos_ == data_.size(); }

  private:
    std::vector<uint8_t> data_;
    size_t pos_ = 0;
    bool ok_ = true;
};

} // namespace pr
