#include "core/image.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "core/fsutil.h"

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

bool ReadImage(const std::string &path, TextureEncoding enc, Image *out, std::string *err) {
    std::string ext = Extension(path);
    if (ext == ".pfm") return ReadPFM(path, out, err);
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

// ---------------------------------------------------------------------------------------------
// Minimal OpenEXR writer: scanline, no compression, B/G/R channels.
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

bool WriteEXR(const std::string &path, const Image &img, bool half, std::string *err) {
    const int w = img.Width(), h = img.Height();
    const int pixelType = half ? 1 : 2;
    const int bytesPer = half ? 2 : 4;
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
    Attr(b, "compression", "compression", {0});
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

    const size_t lineBytes = size_t(w) * 3 * bytesPer;
    uint64_t offset = b.size() + size_t(h) * 8;
    for (int y = 0; y < h; ++y) {
        PutT<uint64_t>(b, offset);
        offset += 8 + lineBytes;
    }
    for (int y = 0; y < h; ++y) {
        PutT<int32_t>(b, y);
        PutT<int32_t>(b, int32_t(lineBytes));
        for (int c : {2, 1, 0}) {
            for (int x = 0; x < w; ++x) {
                float v = img.Get(x, y)[c];
                if (half) PutT<uint16_t>(b, FloatToHalf(v));
                else PutT<float>(b, v);
            }
        }
    }
    FILE *f = _wfopen(Utf8Path(path).wstring().c_str(), L"wb");
    if (!f) {
        *err = "cannot open " + path + " for writing";
        return false;
    }
    bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
    std::fclose(f);
    if (!ok) *err = "write failed: " + path;
    return ok;
}

bool WritePFM(const std::string &path, const Image &img, std::string *err) {
    FILE *f = _wfopen(Utf8Path(path).wstring().c_str(), L"wb");
    if (!f) {
        *err = "cannot open " + path + " for writing";
        return false;
    }
    std::fprintf(f, "PF\n%d %d\n-1.0\n", img.Width(), img.Height());
    for (int y = img.Height() - 1; y >= 0; --y)
        for (int x = 0; x < img.Width(); ++x) {
            RGB c = img.Get(x, y);
            std::fwrite(&c.r, 4, 1, f);
            std::fwrite(&c.g, 4, 1, f);
            std::fwrite(&c.b, 4, 1, f);
        }
    std::fclose(f);
    return true;
}

bool WriteLDR(const std::string &path, const Image &display, std::string *err) {
    const int w = display.Width(), h = display.Height();
    std::vector<uint8_t> px(size_t(w) * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            RGB c = display.Get(x, y);
            for (int k = 0; k < 3; ++k)
                px[3 * (size_t(y) * w + x) + k] = uint8_t(Clamp(int(c[k] * 255.f + 0.5f), 0, 255));
        }
    std::string ext = Extension(path);
    int ok = 0;
    if (ext == ".png") ok = stbi_write_png(path.c_str(), w, h, 3, px.data(), w * 3);
    else if (ext == ".jpg" || ext == ".jpeg") ok = stbi_write_jpg(path.c_str(), w, h, 3, px.data(), 95);
    else if (ext == ".bmp") ok = stbi_write_bmp(path.c_str(), w, h, 3, px.data());
    else if (ext == ".tga") ok = stbi_write_tga(path.c_str(), w, h, 3, px.data());
    else {
        *err = "unsupported LDR format: " + ext;
        return false;
    }
    if (!ok) *err = "failed to write " + path;
    return ok != 0;
}

} // namespace pr
