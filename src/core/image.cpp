#include "core/image.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "core/fsutil.h"
#include "core/parallel.h"

#include "miniz.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBIW_WINDOWS_UTF8
#include "stb/stb_image_write.h"

namespace pr {

static std::string Lower(std::string s) {
    for (char &c : s) c = char(std::tolower((unsigned char)c));
    return s;
}

static std::string Extension(const std::string &path) {
    return Lower(std::filesystem::path(path).extension().string());
}

static bool ReadPFM(const std::string &path, Image *out, std::string *err) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        *err = "cannot open " + path;
        return false;
    }
    char hdr[3] = {0};
    int w, h;
    float scale;
    bool ok = std::fscanf(f, "%2s %d %d %f", hdr, &w, &h, &scale) == 4;
    std::fgetc(f);
    int nc = (ok && std::strcmp(hdr, "PF") == 0) ? 3 : (ok && std::strcmp(hdr, "Pf") == 0 ? 1 : 0);
    if (!ok || nc == 0 || w <= 0 || h <= 0) {
        std::fclose(f);
        *err = "bad PFM header";
        return false;
    }
    std::vector<float> data(size_t(w) * h * nc);
    ok = std::fread(data.data(), sizeof(float), data.size(), f) == data.size();
    std::fclose(f);
    if (!ok) {
        *err = "truncated PFM";
        return false;
    }
    bool bigEndian = scale > 0;
    if (bigEndian)
        for (float &v : data) {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            u = (u >> 24) | ((u >> 8) & 0xff00) | ((u << 8) & 0xff0000) | (u << 24);
            std::memcpy(&v, &u, 4);
        }
    *out = Image(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float *p = &data[(size_t(h - 1 - y) * w + x) * nc];
            out->Set(x, y, nc == 3 ? RGB(p[0], p[1], p[2]) : RGB(p[0], p[0], p[0]));
        }
    return true;
}

bool ReadEXR(const std::string &path, Image *out, std::string *err);  // exr_read.cpp

bool ReadImage(const std::string &path, TextureEncoding enc, Image *out, std::string *err) {
    std::string ext = Extension(path);
    if (ext == ".pfm") return ReadPFM(path, out, err);
    if (ext == ".exr") return ReadEXR(path, out, err);
    int w, h, n;
    if (ext == ".hdr") {
        float *d = stbi_loadf(path.c_str(), &w, &h, &n, 3);
        if (!d) {
            *err = std::string("failed to read ") + path + ": " + stbi_failure_reason();
            return false;
        }
        *out = Image(w, h);
        std::memcpy(out->Data().data(), d, sizeof(float) * size_t(w) * h * 3);
        stbi_image_free(d);
        return true;
    }
    unsigned char *d = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!d) {
        *err = std::string("failed to read ") + path + ": " + stbi_failure_reason();
        return false;
    }
    float lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = enc == TextureEncoding::sRGB ? SRGBDecode(i / 255.f) : i / 255.f;
    *out = Image(w, h);
    auto &dst = out->Data();
    for (size_t i = 0; i < size_t(w) * h * 3; ++i) dst[i] = lut[d[i]];
    stbi_image_free(d);
    return true;
}

bool ReadImageFromMemory(const unsigned char *data, size_t size, TextureEncoding enc, Image *out, std::string *err) {
    int w, h, n;
    int len = int(std::min<size_t>(size, size_t(INT32_MAX)));
    if (stbi_is_hdr_from_memory(data, len)) {
        float *d = stbi_loadf_from_memory(data, len, &w, &h, &n, 3);
        if (!d) {
            *err = std::string("image decode failed: ") + stbi_failure_reason();
            return false;
        }
        *out = Image(w, h);
        std::memcpy(out->Data().data(), d, sizeof(float) * size_t(w) * h * 3);
        stbi_image_free(d);
        return true;
    }
    unsigned char *d = stbi_load_from_memory(data, len, &w, &h, &n, 3);
    if (!d) {
        *err = std::string("image decode failed: ") + stbi_failure_reason();
        return false;
    }
    float lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = enc == TextureEncoding::sRGB ? SRGBDecode(i / 255.f) : i / 255.f;
    *out = Image(w, h);
    auto &dst = out->Data();
    for (size_t i = 0; i < size_t(w) * h * 3; ++i) dst[i] = lut[d[i]];
    stbi_image_free(d);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Minimal OpenEXR writer: scanline, ZIP-compressed (or uncompressed), B/G/R channels, streamed by row.
namespace {

void Put(std::vector<uint8_t> &b, const void *p, size_t n) {
    const uint8_t *c = static_cast<const uint8_t *>(p);
    b.insert(b.end(), c, c + n);
}
template <typename T> void PutT(std::vector<uint8_t> &b, T v) { Put(b, &v, sizeof(T)); }
void PutStr(std::vector<uint8_t> &b, const char *s) { Put(b, s, std::strlen(s) + 1); }
void Attr(std::vector<uint8_t> &b, const char *name, const char *type, const std::vector<uint8_t> &data) {
    PutStr(b, name);
    PutStr(b, type);
    PutT<int32_t>(b, int32_t(data.size()));
    Put(b, data.data(), data.size());
}

uint16_t FloatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = int32_t((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31) return uint16_t(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000;
        uint32_t shift = uint32_t(14 - exp);
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1) half += 1;  // round
        return uint16_t(sign | half);
    }
    uint32_t half = sign | (uint32_t(exp) << 10) | (mant >> 13);
    if (mant & 0x1000) half += 1;  // round to nearest
    return uint16_t(half);
}

} // namespace

bool WriteEXRRows(const std::string &path, int w, int h, const RowFn &rows, bool half, bool zip, std::string *err) {
    const int pixelType = half ? 1 : 2;
    const int bytesPer = half ? 2 : 4;
    const int linesPerBlock = zip ? 16 : 1;  // ZIP_COMPRESSION packs 16 scanlines per block
    std::vector<uint8_t> b;
    PutT<uint32_t>(b, 20000630);
    PutT<uint32_t>(b, 2);
    {
        std::vector<uint8_t> ch;
        for (const char *name : {"B", "G", "R"}) {
            PutStr(ch, name);
            PutT<int32_t>(ch, pixelType);
            PutT<uint8_t>(ch, 0);
            PutT<uint8_t>(ch, 0); PutT<uint8_t>(ch, 0); PutT<uint8_t>(ch, 0);
            PutT<int32_t>(ch, 1);
            PutT<int32_t>(ch, 1);
        }
        PutT<uint8_t>(ch, 0);
        Attr(b, "channels", "chlist", ch);
    }
    Attr(b, "compression", "compression", {uint8_t(zip ? 3 : 0)});
    {
        std::vector<uint8_t> box;
        PutT<int32_t>(box, 0); PutT<int32_t>(box, 0); PutT<int32_t>(box, w - 1); PutT<int32_t>(box, h - 1);
        Attr(b, "dataWindow", "box2i", box);
        Attr(b, "displayWindow", "box2i", box);
    }
    Attr(b, "lineOrder", "lineOrder", {0});
    {
        std::vector<uint8_t> v;
        PutT<float>(v, 1.f);
        Attr(b, "pixelAspectRatio", "float", v);
    }
    {
        std::vector<uint8_t> v;
        PutT<float>(v, 0.f); PutT<float>(v, 0.f);
        Attr(b, "screenWindowCenter", "v2f", v);
    }
    {
        std::vector<uint8_t> v;
        PutT<float>(v, 1.f);
        Attr(b, "screenWindowWidth", "float", v);
    }
    PutT<uint8_t>(b, 0);  // end of header

    FILE *f = _wfopen(Utf8Path(path).wstring().c_str(), L"wb");
    if (!f) {
        *err = "cannot open " + path + " for writing";
        return false;
    }
    const int nBlocks = (h + linesPerBlock - 1) / linesPerBlock;
    bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
    // Offset table placeholder, filled in once the (variable-size) blocks are written.
    const int64_t tablePos = int64_t(b.size());
    std::vector<uint64_t> offsets(size_t(nBlocks), 0);
    ok = ok && std::fwrite(offsets.data(), 8, offsets.size(), f) == offsets.size();
    int64_t pos = tablePos + int64_t(nBlocks) * 8;

    // Encode blocks in parallel batches (rows are resolved on demand, so no full-image copy),
    // then write them in order.
    const size_t lineBytes = size_t(w) * 3 * bytesPer;
    const int batch = std::max(1, ThreadCount() * 4);
    std::vector<std::vector<uint8_t>> encoded(static_cast<size_t>(batch));
    for (int first = 0; first < nBlocks && ok; first += batch) {
        int count = std::min(batch, nBlocks - first);
        ParallelFor(count, [&](int64_t k, int) {
            int blk = first + int(k);
            int y0 = blk * linesPerBlock, y1 = std::min(h, y0 + linesPerBlock);
            std::vector<float> row(size_t(w) * 3);
            std::vector<uint8_t> raw(lineBytes * size_t(y1 - y0));
            uint8_t *dst = raw.data();
            for (int y = y0; y < y1; ++y) {
                rows(y, row.data());
                for (int c : {2, 1, 0})
                    for (int x = 0; x < w; ++x) {
                        float v = row[size_t(x) * 3 + c];
                        if (half) {
                            uint16_t hv = FloatToHalf(v);
                            std::memcpy(dst, &hv, 2);
                            dst += 2;
                        } else {
                            std::memcpy(dst, &v, 4);
                            dst += 4;
                        }
                    }
            }
            std::vector<uint8_t> &out = encoded[size_t(k)];
            out.clear();
            PutT<int32_t>(out, y0);
            if (zip) {
                // OpenEXR ZIP: interleave even/odd bytes, delta-predict, then zlib-compress.
                size_t n = raw.size();
                std::vector<uint8_t> tmp(n);
                uint8_t *t1 = tmp.data(), *t2 = tmp.data() + (n + 1) / 2;
                for (size_t i = 0; i < n; ++i) (i & 1 ? *t2++ : *t1++) = raw[i];
                for (size_t i = n - 1; i > 0; --i) tmp[i] = uint8_t(int(tmp[i]) - int(tmp[i - 1]) + 128);
                mz_ulong clen = mz_compressBound(mz_ulong(n));
                std::vector<uint8_t> comp(clen);
                if (mz_compress2(comp.data(), &clen, tmp.data(), mz_ulong(n), 4) == MZ_OK && clen < n) {
                    PutT<int32_t>(out, int32_t(clen));
                    Put(out, comp.data(), clen);
                    return;
                }
            }
            PutT<int32_t>(out, int32_t(raw.size()));  // stored uncompressed
            Put(out, raw.data(), raw.size());
        });
        for (int k = 0; k < count && ok; ++k) {
            offsets[size_t(first + k)] = uint64_t(pos);
            ok = std::fwrite(encoded[size_t(k)].data(), 1, encoded[size_t(k)].size(), f) == encoded[size_t(k)].size();
            pos += int64_t(encoded[size_t(k)].size());
        }
    }
    ok = ok && _fseeki64(f, tablePos, SEEK_SET) == 0 && std::fwrite(offsets.data(), 8, offsets.size(), f) == offsets.size();
    ok = std::fclose(f) == 0 && ok;
    if (!ok) *err = "write failed: " + path;
    return ok;
}

bool WriteEXR(const std::string &path, const Image &img, bool half, std::string *err) {
    return WriteEXRRows(path, img.Width(), img.Height(),
                        [&](int y, float *rgb) {
                            std::memcpy(rgb, img.Data().data() + size_t(y) * img.Width() * 3, sizeof(float) * img.Width() * 3);
                        },
                        half, true, err);
}

bool WritePFMRows(const std::string &path, int w, int h, const RowFn &rows, std::string *err) {
    FILE *f = _wfopen(Utf8Path(path).wstring().c_str(), L"wb");
    if (!f) {
        *err = "cannot open " + path + " for writing";
        return false;
    }
    std::fprintf(f, "PF\n%d %d\n-1.0\n", w, h);
    std::vector<float> row(size_t(w) * 3);
    bool ok = true;
    for (int y = h - 1; y >= 0 && ok; --y) {  // PFM stores the bottom row first
        rows(y, row.data());
        ok = std::fwrite(row.data(), sizeof(float), row.size(), f) == row.size();
    }
    ok = std::fclose(f) == 0 && ok;
    if (!ok) *err = "write failed: " + path;
    return ok;
}

bool WritePFM(const std::string &path, const Image &img, std::string *err) {
    return WritePFMRows(path, img.Width(), img.Height(),
                        [&](int y, float *rgb) {
                            std::memcpy(rgb, img.Data().data() + size_t(y) * img.Width() * 3, sizeof(float) * img.Width() * 3);
                        },
                        err);
}

bool WriteLDRBytes(const std::string &path, int w, int h, const uint8_t *rgb, std::string *err) {
    std::string ext = Extension(path);
    int ok = 0;
    // stb's zlib is single-threaded; trade a little size for speed on very large images.
    stbi_write_png_compression_level = int64_t(w) * h > 32'000'000 ? 2 : 8;
    if (ext == ".png") ok = stbi_write_png(path.c_str(), w, h, 3, rgb, w * 3);
    else if (ext == ".jpg" || ext == ".jpeg") ok = stbi_write_jpg(path.c_str(), w, h, 3, rgb, 95);
    else if (ext == ".bmp") ok = stbi_write_bmp(path.c_str(), w, h, 3, rgb);
    else if (ext == ".tga") ok = stbi_write_tga(path.c_str(), w, h, 3, rgb);
    else {
        *err = "unsupported LDR format: " + ext;
        return false;
    }
    if (!ok) *err = "failed to write " + path;
    return ok != 0;
}

bool WriteLDR(const std::string &path, const Image &display, std::string *err) {
    const int w = display.Width(), h = display.Height();
    std::vector<uint8_t> px(size_t(w) * h * 3);
    for (size_t i = 0; i < px.size(); ++i) px[i] = uint8_t(Clamp(int(display.Data()[i] * 255.f + 0.5f), 0, 255));
    return WriteLDRBytes(path, w, h, px.data(), err);
}

} // namespace pr
