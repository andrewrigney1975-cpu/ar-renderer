using System.Numerics;

namespace PRenderUI.Scene;

/// <summary>
/// Tiny CPU ray caster used for the interactive camera viewport. It is only a layout aid:
/// Lambert + headlight + hard key-light shadows + a hint of Fresnel for glass and metal.
/// </summary>
public sealed class PreviewTracer
{
    private readonly record struct Sph(Vector3 C, float R, PreviewMaterial M);
    private readonly record struct Tri(Vector3 A, Vector3 E1, Vector3 E2, Vector3 N, Vector2 UA, Vector2 UB, Vector2 UC, PreviewMaterial M);
    private readonly record struct PointGlow(Vector3 C, float R, Vector3 E);

    private readonly List<Sph> _spheres = new();
    private readonly List<Tri> _tris = new();
    private readonly List<PointGlow> _glows = new();
    private readonly Vector3 _keyPos;
    private readonly bool _hasKey;
    private readonly Vector3 _sky;

    public PreviewTracer(SceneDocument doc)
    {
        foreach (var o in doc.Objects)
        {
            if (o.IsInterface) continue;
            if (o.Kind == ShapeKind.Sphere)
            {
                _spheres.Add(new Sph(o.Center, o.WorldRadius, o.Material));
                if (o.Material.Emission != Vector3.Zero && !o.IsLight)
                    _glows.Add(new PointGlow(o.Center, o.WorldRadius, o.Material.Emission));
            }
            else
            {
                foreach (var (a, b, c, ua, ub, uc) in o.Triangles())
                {
                    Vector3 e1 = b - a, e2 = c - a;
                    Vector3 n = Vector3.Cross(e1, e2);
                    if (n.LengthSquared() == 0) continue;
                    _tris.Add(new Tri(a, e1, e2, Vector3.Normalize(n), ua, ub, uc, o.Material));
                }
            }
        }
        var key = doc.Lights.FirstOrDefault(l => l.Type == "sun") ?? doc.Lights.FirstOrDefault(l => l.Type != "environment");
        if (key is not null)
        {
            _keyPos = key.Position;
            _hasKey = true;
        }
        _sky = doc.SkyColor;
    }

    /// <summary>Renders BGRA8 pixels for the given camera.</summary>
    public byte[] Render(CameraInfo cam, int width, int height)
    {
        var pixels = new byte[width * height * 4];
        Vector3 forward = Vector3.Normalize(cam.LookAt - cam.Position);
        Vector3 right = Vector3.Cross(forward, cam.Up);
        right = right.LengthSquared() < 1e-12f ? Vector3.UnitX : Vector3.Normalize(right);
        Vector3 up = Vector3.Cross(right, forward);
        float aspect = (float)width / height;
        float tanHalf = MathF.Tan(cam.FovY * MathF.PI / 360f);
        float halfH = cam.OrthoHeight / 2, halfW = halfH * aspect;

        Parallel.For(0, height, y =>
        {
            for (int x = 0; x < width; ++x)
            {
                float sx = 2 * (x + 0.5f) / width - 1, sy = 1 - 2 * (y + 0.5f) / height;
                Vector3 o, d;
                if (cam.IsOrtho)
                {
                    o = cam.Position + right * (sx * halfW) + up * (sy * halfH);
                    d = forward;
                }
                else
                {
                    o = cam.Position;
                    d = Vector3.Normalize(forward + right * (sx * tanHalf * aspect) + up * (sy * tanHalf));
                }
                Vector3 c = Shade(o, d);
                int i = 4 * (y * width + x);
                pixels[i + 0] = ToByte(c.Z);
                pixels[i + 1] = ToByte(c.Y);
                pixels[i + 2] = ToByte(c.X);
                pixels[i + 3] = 255;
            }
        });
        return pixels;
    }

    private static byte ToByte(float v)
    {
        v = v <= 0 ? 0 : v / (1 + v * 0.35f);  // soft shoulder
        v = MathF.Pow(Math.Clamp(v, 0, 1), 1 / 2.2f);
        return (byte)(v * 255 + 0.5f);
    }

    private bool Intersect(Vector3 o, Vector3 d, float tMax, out float tHit, out Vector3 n, out PreviewMaterial? mat, out Vector2 uv)
    {
        tHit = tMax;
        n = default;
        mat = null;
        uv = default;
        foreach (var s in _spheres)
        {
            Vector3 f = o - s.C;
            float b = Vector3.Dot(f, d);
            float c = Vector3.Dot(f, f) - s.R * s.R;
            float disc = b * b - c;
            if (disc < 0) continue;
            float sq = MathF.Sqrt(disc);
            float t = -b - sq;
            if (t <= 1e-4f) t = -b + sq;
            if (t <= 1e-4f || t >= tHit) continue;
            tHit = t;
            n = Vector3.Normalize(o + d * t - s.C);
            mat = s.M;
        }
        foreach (var tr in _tris)
        {
            Vector3 p = Vector3.Cross(d, tr.E2);
            float det = Vector3.Dot(tr.E1, p);
            if (MathF.Abs(det) < 1e-12f) continue;
            float inv = 1 / det;
            Vector3 tv = o - tr.A;
            float u = Vector3.Dot(tv, p) * inv;
            if (u < 0 || u > 1) continue;
            Vector3 q = Vector3.Cross(tv, tr.E1);
            float v = Vector3.Dot(d, q) * inv;
            if (v < 0 || u + v > 1) continue;
            float t = Vector3.Dot(tr.E2, q) * inv;
            if (t <= 1e-4f || t >= tHit) continue;
            tHit = t;
            n = tr.N;
            mat = tr.M;
            uv = tr.UA * (1 - u - v) + tr.UB * u + tr.UC * v;
        }
        return mat is not null;
    }

    private Vector3 Background(Vector3 d)
    {
        float t = Math.Clamp(0.5f + 0.5f * d.Y, 0, 1);
        Vector3 horizon = new(0.10f, 0.11f, 0.13f);
        Vector3 zenith = Vector3.Clamp(_sky * 2.5f + new Vector3(0.05f), Vector3.Zero, Vector3.One);
        return Vector3.Lerp(horizon, zenith, t);
    }

    private Vector3 Shade(Vector3 o, Vector3 d, int depth = 0)
    {
        if (!Intersect(o, d, float.MaxValue, out float t, out Vector3 n, out var m, out Vector2 uv) || m is null)
            return Background(d);
        bool inside = Vector3.Dot(n, d) > 0;
        if (inside) n = -n;
        Vector3 p = o + d * t;

        // Glass and metal get a few levels of real reflection/refraction so the layout preview
        // reads correctly; everything else is simple local shading.
        if (depth < 4 && (m.IsGlass || m.IsMetal))
        {
            float cosI = MathF.Max(0, Vector3.Dot(n, -d));
            Vector3 reflDir = Vector3.Reflect(d, n);
            if (m.IsMetal)
            {
                Vector3 r = Shade(p + n * 1e-3f, reflDir, depth + 1);
                float schlick = 0.6f + 0.4f * MathF.Pow(1 - cosI, 5);
                return m.Color * r * schlick + LocalShade(p, n, d, m, uv) * 0.25f;
            }
            float eta = inside ? 1.5f : 1 / 1.5f;
            float k = 1 - eta * eta * (1 - cosI * cosI);
            float f0 = 0.04f;
            float F = k < 0 ? 1 : f0 + (1 - f0) * MathF.Pow(1 - cosI, 5);
            Vector3 refl = Shade(p + n * 1e-3f, reflDir, depth + 1);
            if (k < 0) return refl;
            Vector3 refrDir = Vector3.Normalize(d * eta + n * (eta * cosI - MathF.Sqrt(k)));
            Vector3 refr = Shade(p - n * 1e-3f, refrDir, depth + 1);
            Vector3 tint = inside ? m.Color : Vector3.One;
            return Vector3.Lerp(refr * tint, refl, F);
        }
        return LocalShade(p, n, d, m, uv);
    }

    private Vector3 LocalShade(Vector3 p, Vector3 n, Vector3 d, PreviewMaterial m, Vector2 uv)
    {
        if (m.Emission != Vector3.Zero) return m.Emission;

        Vector3 baseColor = m.Color;
        if (m.Checker)
        {
            bool odd = (((int)MathF.Floor(uv.X * m.CheckerScale) + (int)MathF.Floor(uv.Y * m.CheckerScale)) & 1) != 0;
            baseColor = odd ? m.CheckerB : m.Color;
        }

        Vector3 light = new(0.12f);                                     // ambient
        light += new Vector3(0.35f) * MathF.Max(0, Vector3.Dot(n, -d));  // headlight
        if (_hasKey)
        {
            Vector3 toL = _keyPos - p;
            float dist = toL.Length();
            toL /= dist;
            float ndl = MathF.Max(0, Vector3.Dot(n, toL));
            if (ndl > 0 && Intersect(p + n * 1e-3f, toL, dist, out _, out _, out var occ, out _) && occ is not null && !occ.IsGlass)
                ndl *= 0.25f;
            light += new Vector3(0.9f, 0.88f, 0.84f) * ndl;
        }
        foreach (var g in _glows)
        {
            Vector3 toG = g.C - p;
            float d2 = toG.LengthSquared();
            if (d2 < 1e-6f) continue;
            float ndl = MathF.Max(0, Vector3.Dot(n, toG / MathF.Sqrt(d2)));
            light += g.E * (0.6f * g.R * g.R / d2 * ndl);
        }

        float cosV = MathF.Max(0, Vector3.Dot(n, -d));
        float fresnel = 0.04f + 0.96f * MathF.Pow(1 - cosV, 5);
        Vector3 refl = Vector3.Reflect(d, n);
        if (m.IsGlass)
        {
            Vector3 env = Background(refl);
            return baseColor * (0.18f + 0.25f * light.Y) + env * (0.3f + fresnel) + new Vector3(0.15f * fresnel);
        }
        if (m.IsMetal)
        {
            float spec = 0;
            if (_hasKey) spec = MathF.Pow(MathF.Max(0, Vector3.Dot(refl, Vector3.Normalize(_keyPos - p))), 24);
            return baseColor * (0.25f * light.X + 0.55f * spec + 0.35f * Background(refl).Y + 0.2f);
        }
        return baseColor * light;
    }
}
