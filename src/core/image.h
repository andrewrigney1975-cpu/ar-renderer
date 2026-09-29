#pragma once

#include "core/color.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pr {

// Simple 3-channel float image (row-major, top row first).
class Image {
  public:
    Image() = default;
    Image(int w, int h) : width_(w), height_(h), data_(size_t(w) * h * 3, 0.f) {}

    int Width() const { return width_; }
    int Height() const { return height_; }
    bool Empty() const { return data_.empty(); }
    RGB Get(int x, int y) const {
        const float *p = &data_[3 * (size_t(y) * width_ + x)];
        return {p[0], p[1], p[2]};
    }
    void Set(int x, int y, RGB c) {
        float *p = &data_[3 * (size_t(y) * width_ + x)];
        p[0] = c.r; p[1] = c.g; p[2] = c.b;
    }
    const std::vector<float> &Data() const { return data_; }
    std::vector<float> &Data() { return data_; }

  private:
    int width_ = 0, height_ = 0;
    std::vector<float> data_;
};

enum class TextureEncoding { sRGB, Linear };

// Load an image as linear float RGB. 8-bit images are decoded with the given encoding;
// float formats (.hdr, .pfm) are always linear.
bool ReadImage(const std::string &path, TextureEncoding enc, Image *out, std::string *err);

// Decode an encoded image (PNG, JPEG, HDR, ...) held in memory.
bool ReadImageFromMemory(const unsigned char *data, size_t size, TextureEncoding enc, Image *out, std::string *err);

// Supplies one row of linear RGB floats (3 * width values) for row y; may be called from
// several threads at once.
using RowFn = std::function<void(int y, float *rgb)>;

// Streamed writers: rows are requested on demand, so no full-resolution copy is made (needed for
// 8K/16K output). EXR uses ZIP compression (16-line blocks, encoded in parallel) unless zip=false.
bool WriteEXRRows(const std::string &path, int w, int h, const RowFn &rows, bool half, bool zip, std::string *err);
bool WritePFMRows(const std::string &path, int w, int h, const RowFn &rows, std::string *err);
bool WriteLDRBytes(const std::string &path, int w, int h, const uint8_t *rgb, std::string *err);

// Write linear float RGB (EXR: ZIP-compressed float32 scanlines; PFM).
bool WriteEXR(const std::string &path, const Image &img, bool half, std::string *err);
bool WritePFM(const std::string &path, const Image &img, std::string *err);
// Write display-referred [0,1] values (already tone-mapped and encoded) as 8-bit PNG/JPEG.
bool WriteLDR(const std::string &path, const Image &display, std::string *err);

} // namespace pr
