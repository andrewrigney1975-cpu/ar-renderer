#include "core/math.h"

namespace pr {

bool Mat4::Inverse(Mat4 *out) const {
    // Gauss-Jordan elimination with full pivoting, in double precision.
    int indxc[4], indxr[4];
    int ipiv[4] = {0, 0, 0, 0};
    double minv[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) minv[i][j] = m[i][j];
    for (int i = 0; i < 4; i++) {
        int irow = 0, icol = 0;
        double big = 0.;
        for (int j = 0; j < 4; j++) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < 4; k++) {
                    if (ipiv[k] == 0) {
                        if (std::abs(minv[j][k]) >= big) {
                            big = std::abs(minv[j][k]);
                            irow = j;
                            icol = k;
                        }
                    } else if (ipiv[k] > 1) {
                        return false;
                    }
                }
            }
        }
        ++ipiv[icol];
        if (irow != icol)
            for (int k = 0; k < 4; ++k) std::swap(minv[irow][k], minv[icol][k]);
        indxr[i] = irow;
        indxc[i] = icol;
        if (minv[icol][icol] == 0.) return false;
        double pivinv = 1. / minv[icol][icol];
        minv[icol][icol] = 1.;
        for (int j = 0; j < 4; j++) minv[icol][j] *= pivinv;
        for (int j = 0; j < 4; j++) {
            if (j != icol) {
                double save = minv[j][icol];
                minv[j][icol] = 0;
                for (int k = 0; k < 4; k++) minv[j][k] -= minv[icol][k] * save;
            }
        }
    }
    for (int j = 3; j >= 0; j--) {
        if (indxr[j] != indxc[j]) {
            for (int k = 0; k < 4; k++) std::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out->m[i][j] = float(minv[i][j]);
    return true;
}

Transform Transform::Translate(const Vec3f &d) {
    Mat4 m, mi;
    m.m[0][3] = d.x; m.m[1][3] = d.y; m.m[2][3] = d.z;
    mi.m[0][3] = -d.x; mi.m[1][3] = -d.y; mi.m[2][3] = -d.z;
    return Transform(m, mi);
}

Transform Transform::Scale(const Vec3f &s) {
    Mat4 m, mi;
    m.m[0][0] = s.x; m.m[1][1] = s.y; m.m[2][2] = s.z;
    mi.m[0][0] = 1 / s.x; mi.m[1][1] = 1 / s.y; mi.m[2][2] = 1 / s.z;
    return Transform(m, mi);
}

Transform Transform::Rotate(float degrees, const Vec3f &axis) {
    Vec3f a = Normalize(axis);
    float sinTheta = std::sin(Radians(degrees)), cosTheta = std::cos(Radians(degrees));
    Mat4 m;
    m.m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    m.m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    m.m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    m.m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    m.m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    m.m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    m.m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    m.m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    m.m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    return Transform(m, m.Transposed());
}

Transform Transform::LookAt(const Vec3f &pos, const Vec3f &target, const Vec3f &up) {
    // Right-handed world; camera space: +x right, +y up, +z forward.
    Vec3f dir = Normalize(target - pos);
    Vec3f right = Cross(dir, Normalize(up));
    if (LengthSquared(right) < 1e-12f) {
        Vec3f t1, t2;
        CoordinateSystem(dir, &t1, &t2);
        right = t1;
    }
    right = Normalize(right);
    Vec3f newUp = Cross(right, dir);
    Mat4 m;
    m.m[0][0] = right.x; m.m[1][0] = right.y; m.m[2][0] = right.z;
    m.m[0][1] = newUp.x; m.m[1][1] = newUp.y; m.m[2][1] = newUp.z;
    m.m[0][2] = dir.x;   m.m[1][2] = dir.y;   m.m[2][2] = dir.z;
    m.m[0][3] = pos.x;   m.m[1][3] = pos.y;   m.m[2][3] = pos.z;
    return Transform(m);
}

} // namespace pr
