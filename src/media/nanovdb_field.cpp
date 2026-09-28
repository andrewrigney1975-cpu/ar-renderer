// NanoVDB density fields (uncompressed .nvdb files, float grids).

#include "media/medium.h"

#include "core/fsutil.h"

#include <nanovdb/NanoVDB.h>

#include <cstdio>
#include <cstring>
#include <fstream>

namespace pr {

namespace {

class NanoVDBField : public DensityField {
  public:
    NanoVDBField(void *buffer, const nanovdb::NanoGrid<float> *grid) : buffer_(buffer), grid_(grid) {
        auto wb = grid_->worldBBox();
        bounds = Bounds3f(Vec3f(float(wb.min()[0]), float(wb.min()[1]), float(wb.min()[2])),
                          Vec3f(float(wb.max()[0]), float(wb.max()[1]), float(wb.max()[2])));
        maxDensity_ = std::max(0.f, grid_->tree().root().maximum());
    }
    ~NanoVDBField() override { ::operator delete(buffer_, std::align_val_t(NANOVDB_DATA_ALIGNMENT)); }

    float Density(const Vec3f &p) const override {
        if (!bounds.Inside(p)) return 0;
        nanovdb::Vec3f ip = grid_->worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
        // Trilinear interpolation between voxel centres (NanoVDB voxel centres are at integers).
        float fx = std::floor(ip[0]), fy = std::floor(ip[1]), fz = std::floor(ip[2]);
        nanovdb::Coord c{int(fx), int(fy), int(fz)};
        float dx = ip[0] - fx, dy = ip[1] - fy, dz = ip[2] - fz;
        auto acc = grid_->getAccessor();
        auto v = [&](int i, int j, int k) { return float(acc.getValue(c.offsetBy(i, j, k))); };
        float c00 = Lerp(dx, v(0, 0, 0), v(1, 0, 0)), c10 = Lerp(dx, v(0, 1, 0), v(1, 1, 0));
        float c01 = Lerp(dx, v(0, 0, 1), v(1, 0, 1)), c11 = Lerp(dx, v(0, 1, 1), v(1, 1, 1));
        return std::max(0.f, Lerp(dz, Lerp(dy, c00, c10), Lerp(dy, c01, c11)));
    }
    float MaxDensity() const override { return maxDensity_; }

  private:
    void *buffer_;
    const nanovdb::NanoGrid<float> *grid_;
    float maxDensity_ = 0;
};

} // namespace

std::shared_ptr<DensityField> LoadNanoVDB(const std::string &path, const std::string &gridName, std::string *err) {
    std::ifstream in(Utf8Path(path), std::ios::binary);
    if (!in) {
        *err = "cannot open " + path;
        return nullptr;
    }
    nanovdb::io::FileHeader head;
    while (in.read(reinterpret_cast<char *>(&head), sizeof(head))) {
        if (!head.isValid()) {
            *err = path + ": not a NanoVDB file";
            return nullptr;
        }
        if (head.codec != nanovdb::io::Codec::NONE) {
            *err = path + ": compressed NanoVDB files are not supported (re-save without ZIP/Blosc)";
            return nullptr;
        }
        for (uint16_t i = 0; i < head.gridCount; ++i) {
            nanovdb::io::FileMetaData meta;
            if (!in.read(reinterpret_cast<char *>(&meta), sizeof(meta))) break;
            std::string name(meta.nameSize, '\0');
            in.read(name.data(), meta.nameSize);
            if (!name.empty() && name.back() == '\0') name.pop_back();
            bool wanted = meta.gridType == nanovdb::GridType::Float && (gridName.empty() || name == gridName);
            if (!wanted) {
                in.seekg(std::streamoff(meta.gridSize), std::ios::cur);
                continue;
            }
            void *buf = ::operator new(meta.gridSize, std::align_val_t(NANOVDB_DATA_ALIGNMENT));
            if (!in.read(static_cast<char *>(buf), std::streamsize(meta.gridSize))) {
                ::operator delete(buf, std::align_val_t(NANOVDB_DATA_ALIGNMENT));
                *err = path + ": truncated grid '" + name + "'";
                return nullptr;
            }
            auto *grid = reinterpret_cast<const nanovdb::NanoGrid<float> *>(buf);
            if (!grid->isValid() || grid->gridType() != nanovdb::GridType::Float) {
                ::operator delete(buf, std::align_val_t(NANOVDB_DATA_ALIGNMENT));
                *err = path + ": grid '" + name + "' is not a valid float grid";
                return nullptr;
            }
            return std::make_shared<NanoVDBField>(buf, grid);
        }
    }
    *err = path + ": no float grid" + (gridName.empty() ? std::string() : " named '" + gridName + "'") + " found";
    return nullptr;
}

} // namespace pr
