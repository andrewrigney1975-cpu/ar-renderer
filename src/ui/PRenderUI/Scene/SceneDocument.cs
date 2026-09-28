using System.Numerics;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace PRenderUI.Scene;

public enum ShapeKind { Sphere, Rect, Box, Disk, Mesh }

/// <summary>A scene object reduced to what the viewports need for display.</summary>
public sealed class SceneObject
{
    public required string Name { get; init; }
    public required ShapeKind Kind { get; init; }
    public Matrix4x4 Transform { get; init; } = Matrix4x4.Identity;  // row-vector convention
    public float Radius { get; init; } = 1;                          // sphere / disk
    public Vector3 Size { get; init; } = Vector3.One;                // rect (x, z) / box
    public required PreviewMaterial Material { get; init; }
    public bool IsLight { get; init; }
    public bool IsInterface { get; init; }                            // invisible medium boundary

    public Vector3 Center => Vector3.Transform(Vector3.Zero, Transform);
    public float WorldRadius
    {
        get
        {
            float s = (Vector3.TransformNormal(Vector3.UnitX, Transform).Length() +
                       Vector3.TransformNormal(Vector3.UnitY, Transform).Length() +
                       Vector3.TransformNormal(Vector3.UnitZ, Transform).Length()) / 3f;
            return Radius * s;
        }
    }

    /// <summary>World-space triangles (with per-vertex uv) for planar/box shapes.</summary>
    public List<(Vector3 A, Vector3 B, Vector3 C, Vector2 UA, Vector2 UB, Vector2 UC)> Triangles()
    {
        var tris = new List<(Vector3, Vector3, Vector3, Vector2, Vector2, Vector2)>();
        Vector3 T(Vector3 p) => Vector3.Transform(p, Transform);
        switch (Kind)
        {
            case ShapeKind.Rect:
            {
                float hw = Size.X / 2, hd = Size.Z / 2;
                Vector3 p0 = T(new(-hw, 0, -hd)), p1 = T(new(hw, 0, -hd)), p2 = T(new(hw, 0, hd)), p3 = T(new(-hw, 0, hd));
                tris.Add((p0, p2, p1, new(0, 0), new(1, 1), new(1, 0)));
                tris.Add((p0, p3, p2, new(0, 0), new(0, 1), new(1, 1)));
                break;
            }
            case ShapeKind.Disk:
            {
                const int n = 48;
                Vector3 c = T(Vector3.Zero);
                for (int i = 0; i < n; ++i)
                {
                    float a0 = MathF.Tau * i / n, a1 = MathF.Tau * (i + 1) / n;
                    Vector3 pa = T(new(Radius * MathF.Cos(a0), 0, Radius * MathF.Sin(a0)));
                    Vector3 pb = T(new(Radius * MathF.Cos(a1), 0, Radius * MathF.Sin(a1)));
                    tris.Add((c, pb, pa, new(0.5f, 0.5f), new(0.5f, 0.5f), new(0.5f, 0.5f)));
                }
                break;
            }
            case ShapeKind.Box:
            {
                Vector3 h = Size / 2;
                (Vector3 n, Vector3 u, Vector3 v)[] faces =
                {
                    (new(1, 0, 0), new(0, 0, -1), new(0, 1, 0)), (new(-1, 0, 0), new(0, 0, 1), new(0, 1, 0)),
                    (new(0, 1, 0), new(1, 0, 0), new(0, 0, -1)), (new(0, -1, 0), new(1, 0, 0), new(0, 0, 1)),
                    (new(0, 0, 1), new(1, 0, 0), new(0, 1, 0)), (new(0, 0, -1), new(-1, 0, 0), new(0, 1, 0)),
                };
                foreach (var (n, u, v) in faces)
                {
                    Vector3 c = n * h, U = u * h, V = v * h;
                    Vector3 a = T(c - U - V), b = T(c + U - V), cc = T(c + U + V), d = T(c - U + V);
                    tris.Add((a, b, cc, new(0, 0), new(1, 0), new(1, 1)));
                    tris.Add((a, cc, d, new(0, 0), new(1, 1), new(0, 1)));
                }
                break;
            }
        }
        return tris;
    }
}

public sealed class PreviewMaterial
{
    public Vector3 Color { get; init; } = new(0.6f, 0.6f, 0.6f);
    public Vector3 Emission { get; init; }
    public bool IsGlass { get; init; }
    public bool IsMetal { get; init; }
    // Checker pattern (floor textures)
    public bool Checker { get; init; }
    public Vector3 CheckerB { get; init; }
    public float CheckerScale { get; init; } = 1;
    public string Id { get; init; } = "";
}

public sealed class CameraInfo
{
    public required string Name { get; init; }
    public string Type { get; set; } = "perspective";
    public Vector3 Position { get; set; }
    public Vector3 LookAt { get; set; }
    public Vector3 Up { get; set; } = Vector3.UnitY;
    public float FovY { get; set; } = 40;
    public float OrthoHeight { get; set; } = 2;
    public JsonObject Source { get; init; } = new();
    public bool IsOrtho => Type is "orthographic" or "ortho";

    public CameraInfo Clone() => new()
    {
        Name = Name, Type = Type, Position = Position, LookAt = LookAt, Up = Up, FovY = FovY,
        OrthoHeight = OrthoHeight, Source = (JsonObject)Source.DeepClone(),
    };
}

public sealed class LightInfo
{
    public required string Name { get; init; }
    public required string Type { get; init; }
    public Vector3 Position { get; init; }
    public Vector3 Direction { get; init; } = -Vector3.UnitY;
    public Vector2 Size { get; init; } = Vector2.One;
    public float Radius { get; init; }
}

/// <summary>
/// Loads a .prscene.json file with the same composition rules as the renderer (includes merged
/// first, dictionary sections merged by id, objects appended, active light rigs merged last).
/// </summary>
public sealed class SceneDocument
{
    private static readonly string[] DictSections = { "textures", "materials", "media", "geometry", "lights", "cameras", "rigs" };
    private static readonly JsonDocumentOptions DocOptions = new() { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };

    public string Path { get; private set; } = "";
    public List<SceneObject> Objects { get; } = new();
    public List<LightInfo> Lights { get; } = new();
    public Dictionary<string, CameraInfo> Cameras { get; } = new();
    public List<string> RigNames { get; } = new();
    public List<string> DefaultActiveRigs { get; } = new();
    public string DefaultCamera { get; private set; } = "";
    public int FilmWidth { get; private set; } = 640;
    public int FilmHeight { get; private set; } = 360;
    public string Integrator { get; private set; } = "mmlt";
    public double MutationsPerPixel { get; private set; } = 256;
    public int Spp { get; private set; } = 64;
    public Vector3 SkyColor { get; private set; } = new(0.1f, 0.12f, 0.16f);
    public List<string> Warnings { get; } = new();

    public static SceneDocument Load(string path, IReadOnlyCollection<string>? activeRigs = null)
    {
        var doc = new SceneDocument { Path = System.IO.Path.GetFullPath(path) };
        JsonObject root = LoadComposed(doc.Path, 0);
        doc.Build(root, activeRigs);
        return doc;
    }

    private static JsonObject ReadJson(string path)
    {
        string text = File.ReadAllText(path);
        var node = JsonNode.Parse(text, null, DocOptions) as JsonObject
                   ?? throw new InvalidDataException($"{path}: root must be an object");
        ResolvePaths(node, System.IO.Path.GetDirectoryName(path)!);
        return node;
    }

    private static void ResolvePaths(JsonNode? node, string dir)
    {
        if (node is JsonObject obj)
        {
            foreach (var key in obj.Select(kv => kv.Key).ToList())
            {
                var v = obj[key];
                if (key == "file" && v is JsonValue fv && fv.TryGetValue(out string? s) && !System.IO.Path.IsPathRooted(s))
                    obj[key] = System.IO.Path.GetFullPath(System.IO.Path.Combine(dir, s));
                else if (key == "rigs" && v is JsonObject rigs)
                {
                    foreach (var rk in rigs.Select(kv => kv.Key).ToList())
                    {
                        if (rigs[rk] is JsonValue rv && rv.TryGetValue(out string? rs) && !System.IO.Path.IsPathRooted(rs))
                            rigs[rk] = System.IO.Path.GetFullPath(System.IO.Path.Combine(dir, rs));
                        else ResolvePaths(rigs[rk], dir);
                    }
                }
                else ResolvePaths(v, dir);
            }
        }
        else if (node is JsonArray arr)
        {
            foreach (var e in arr) ResolvePaths(e, dir);
        }
    }

    private static void Merge(JsonObject dst, JsonObject src)
    {
        foreach (var (key, value) in src.ToList())
        {
            if (key == "include") continue;
            if (DictSections.Contains(key) && value is JsonObject sec)
            {
                if (dst[key] is not JsonObject d) dst[key] = d = new JsonObject();
                foreach (var (id, item) in sec) d[id] = item?.DeepClone();
            }
            else if (key == "objects" && value is JsonArray objs)
            {
                if (dst[key] is not JsonArray d) dst[key] = d = new JsonArray();
                foreach (var o in objs) d.Add(o?.DeepClone());
            }
            else if (key == "render" && dst[key] is JsonObject dr && value is JsonObject sr)
            {
                DeepMerge(dr, sr);
            }
            else dst[key] = value?.DeepClone();
        }
    }

    private static void DeepMerge(JsonObject dst, JsonObject src)
    {
        foreach (var (key, value) in src.ToList())
        {
            if (dst[key] is JsonObject d && value is JsonObject s) DeepMerge(d, s);
            else dst[key] = value?.DeepClone();
        }
    }

    private static JsonObject LoadComposed(string path, int depth)
    {
        if (depth > 16) throw new InvalidDataException("include depth exceeded");
        JsonObject doc = ReadJson(path);
        var result = new JsonObject();
        if (doc["include"] is JsonArray inc)
        {
            foreach (var i in inc)
            {
                string p = i!.GetValue<string>();
                if (!System.IO.Path.IsPathRooted(p)) p = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(path)!, p);
                Merge(result, LoadComposed(System.IO.Path.GetFullPath(p), depth + 1));
            }
        }
        Merge(result, doc);
        return result;
    }

    private void Build(JsonObject root, IReadOnlyCollection<string>? activeRigs)
    {
        foreach (var s in DictSections) root[s] ??= new JsonObject();
        root["objects"] ??= new JsonArray();

        foreach (var (name, _) in root["rigs"]!.AsObject()) RigNames.Add(name);
        if (root["active_rigs"] is JsonArray ar)
            foreach (var r in ar) DefaultActiveRigs.Add(r!.GetValue<string>());
        foreach (var rig in activeRigs ?? DefaultActiveRigs)
        {
            if (root["rigs"]![rig] is not JsonNode rn) continue;
            JsonObject rigDoc = rn switch
            {
                JsonValue v => LoadComposed(v.GetValue<string>(), 0),
                JsonObject o when o["file"] is JsonValue f => LoadComposed(f.GetValue<string>(), 0),
                JsonObject o => (JsonObject)o.DeepClone(),
                _ => new JsonObject(),
            };
            Merge(root, rigDoc);
        }

        var render = root["render"] as JsonObject ?? new JsonObject();
        DefaultCamera = render["camera"]?.GetValue<string>() ?? "";
        if (render["film"] is JsonObject film)
        {
            FilmWidth = (int)Num(film["width"], 640);
            FilmHeight = (int)Num(film["height"], 360);
        }
        if (render["integrator"] is JsonObject integ)
        {
            Integrator = integ["type"]?.GetValue<string>() ?? "mmlt";
            MutationsPerPixel = Num(integ["mutations_per_pixel"], 256);
            Spp = (int)Num(integ["spp"], 64);
        }

        var materials = root["materials"]!.AsObject();
        var textures = root["textures"]!.AsObject();
        var geometry = root["geometry"]!.AsObject();
        var matCache = new Dictionary<string, PreviewMaterial>();
        PreviewMaterial Mat(string id)
        {
            if (matCache.TryGetValue(id, out var m)) return m;
            m = MakeMaterial(id, materials[id] as JsonObject, textures);
            matCache[id] = m;
            return m;
        }

        int index = 0;
        foreach (var on in root["objects"]!.AsArray())
        {
            if (on is not JsonObject o) continue;
            index++;
            if (o["visible"] is JsonValue vis && !vis.GetValue<bool>()) continue;
            string gid = o["geometry"]?.GetValue<string>() ?? "";
            if (geometry[gid] is not JsonObject g)
            {
                Warnings.Add($"object {index}: unknown geometry '{gid}'");
                continue;
            }
            string matId = o["material"]?.GetValue<string>() ?? "";
            var mj = materials[matId] as JsonObject;
            bool isInterface = mj?["type"]?.GetValue<string>() == "interface";
            var mat = Mat(matId);
            if (o["emission"] is JsonObject em) mat = WithEmission(mat, em);
            Objects.Add(MakeObject(o["name"]?.GetValue<string>() ?? $"object{index}", g, ParseTransform(o["transform"]), mat,
                                   isLight: false, isInterface));
        }

        foreach (var (id, ln) in root["lights"]!.AsObject())
        {
            if (ln is not JsonObject l) continue;
            if (l["enabled"] is JsonValue en && !en.GetValue<bool>()) continue;
            string type = l["type"]?.GetValue<string>() ?? "";
            if (type is "environment" or "env" or "sky")
            {
                if (l["color"] is JsonArray c) SkyColor = Vec(c) * (float)Num(l["intensity"], 1);
                Lights.Add(new LightInfo { Name = id, Type = "environment" });
                continue;
            }
            Vector3 pos = l["position"] is JsonNode pn ? Vec(pn) : Vector3.Zero;
            Vector3 dir = l["look_at"] is JsonNode la ? Vector3.Normalize(Vec(la) - pos)
                        : l["direction"] is JsonNode dn ? Vector3.Normalize(Vec(dn)) : -Vector3.UnitY;
            Vector2 size = l["size"] is JsonArray sz ? new((float)Num(sz[0], 1), (float)Num(sz[1], 1)) : Vector2.One;
            float radius = (float)Num(l["radius"], type == "sphere" ? 0.1 : 0.5);
            Lights.Add(new LightInfo { Name = id, Type = type, Position = pos, Direction = dir, Size = size, Radius = radius });
            if (type is "rect" or "sphere" or "disk")
            {
                var lightMat = new PreviewMaterial { Color = new(1, 0.95f, 0.8f), Emission = new(1.6f, 1.5f, 1.3f), Id = id };
                // Rect/disk: local +Y maps to the light direction.
                Matrix4x4 xf = Matrix4x4.CreateTranslation(pos);
                if (type != "sphere") xf = BasisFromY(dir) * xf;
                var g = new JsonObject { ["type"] = type == "rect" ? "rect" : type };
                if (type == "rect") g["size"] = new JsonArray((double)size.X, (double)size.Y);
                else g["radius"] = (double)radius;
                Objects.Add(MakeObject("light:" + id, g, xf, lightMat, isLight: true, isInterface: false));
            }
        }

        foreach (var (name, cn) in root["cameras"]!.AsObject())
        {
            if (cn is not JsonObject c) continue;
            var cam = new CameraInfo
            {
                Name = name,
                Type = c["type"]?.GetValue<string>() ?? "perspective",
                Position = c["position"] is JsonNode p ? Vec(p) : new Vector3(0, 0, 5),
                Up = c["up"] is JsonNode u ? Vec(u) : Vector3.UnitY,
                OrthoHeight = (float)Num(c["ortho_height"], 2),
                Source = (JsonObject)c.DeepClone(),
            };
            cam.LookAt = c["look_at"] is JsonNode t ? Vec(t) : cam.Position + (c["direction"] is JsonNode d ? Vec(d) : -Vector3.UnitZ);
            if (c["fov_y"] is JsonNode f) cam.FovY = (float)Num(f, 40);
            else if (c["focal_length_mm"] is JsonNode fl)
                cam.FovY = (float)(2 * Math.Atan(Num(c["sensor_height_mm"], 24) / (2 * Num(fl, 50))) * 180 / Math.PI);
            Cameras[name] = cam;
        }
        if (string.IsNullOrEmpty(DefaultCamera) && Cameras.Count > 0) DefaultCamera = Cameras.Keys.First();
    }

    // ---- helpers -------------------------------------------------------------------------

    private static double Num(JsonNode? n, double def) =>
        n is JsonValue v && v.TryGetValue(out double d) ? d : def;

    private static Vector3 Vec(JsonNode n)
    {
        if (n is JsonArray a && a.Count == 3) return new((float)Num(a[0], 0), (float)Num(a[1], 0), (float)Num(a[2], 0));
        float s = (float)Num(n, 0);
        return new(s, s, s);
    }

    private static Matrix4x4 BasisFromY(Vector3 y)
    {
        y = Vector3.Normalize(y);
        Vector3 helper = MathF.Abs(y.Y) > 0.9f ? Vector3.UnitX : Vector3.UnitY;
        Vector3 x = Vector3.Normalize(Vector3.Cross(helper, y));
        Vector3 z = Vector3.Cross(x, y);
        return new Matrix4x4(x.X, x.Y, x.Z, 0, y.X, y.Y, y.Z, 0, z.X, z.Y, z.Z, 0, 0, 0, 0, 1);
    }

    /// <summary>Same semantics as the renderer: T * R * S (column vectors).</summary>
    public static Matrix4x4 ParseTransform(JsonNode? n)
    {
        if (n is not JsonObject t) return Matrix4x4.Identity;
        if (t["matrix"] is JsonArray m && m.Count == 16)
        {
            float F(int i) => (float)Num(m[i], 0);
            // Row-major column-vector matrix -> System.Numerics row-vector matrix (transpose).
            return new Matrix4x4(F(0), F(4), F(8), F(12), F(1), F(5), F(9), F(13), F(2), F(6), F(10), F(14), F(3), F(7), F(11), F(15));
        }
        Matrix4x4 S = Matrix4x4.Identity, R = Matrix4x4.Identity, T = Matrix4x4.Identity;
        if (t["translate"] is JsonNode tr) T = Matrix4x4.CreateTranslation(Vec(tr));
        static Matrix4x4 One(JsonNode e)
        {
            if (e is JsonArray a && a.Count == 4)
                return Matrix4x4.CreateFromAxisAngle(Vector3.Normalize(new((float)Num(a[1], 0), (float)Num(a[2], 0), (float)Num(a[3], 0))),
                                                     (float)(Num(a[0], 0) * Math.PI / 180));
            if (e is JsonObject o)
                return Matrix4x4.CreateFromAxisAngle(Vector3.Normalize(Vec(o["axis"]!)), (float)(Num(o["angle"], 0) * Math.PI / 180));
            return Matrix4x4.Identity;
        }
        if (t["rotate"] is JsonArray rot)
        {
            if (rot.Count > 0 && rot[0] is JsonArray)
                foreach (var e in rot) R = One(e!) * R;  // column R1*R2 == row R2*R1
            else R = One(rot);
        }
        else if (t["rotate"] is JsonObject ro) R = One(ro);
        if (t["euler"] is JsonNode eu)
        {
            Vector3 e = Vec(eu) * (MathF.PI / 180);
            // Column Rz*Ry*Rx == row Rx*Ry*Rz; appended after any explicit rotation.
            R = Matrix4x4.CreateRotationX(e.X) * Matrix4x4.CreateRotationY(e.Y) * Matrix4x4.CreateRotationZ(e.Z) * R;
        }
        if (t["scale"] is JsonNode sc) S = Matrix4x4.CreateScale(Vec(sc));
        return S * R * T;
    }

    private static SceneObject MakeObject(string name, JsonObject g, Matrix4x4 xf, PreviewMaterial mat, bool isLight, bool isInterface)
    {
        string type = g["type"]?.GetValue<string>() ?? "mesh";
        ShapeKind kind = type switch
        {
            "sphere" => ShapeKind.Sphere,
            "rect" or "quad" => ShapeKind.Rect,
            "box" or "cube" => ShapeKind.Box,
            "disk" => ShapeKind.Disk,
            _ => ShapeKind.Mesh,
        };
        Vector3 size = Vector3.One;
        if (kind == ShapeKind.Rect && g["size"] is JsonArray rs) size = new((float)Num(rs[0], 1), 0, (float)Num(rs[1], 1));
        if (kind == ShapeKind.Box && g["size"] is JsonNode bs) size = Vec(bs);
        float radius = (float)Num(g["radius"], 1);
        if (kind == ShapeKind.Sphere && g["center"] is JsonNode cn) xf = Matrix4x4.CreateTranslation(Vec(cn)) * xf;
        return new SceneObject
        {
            Name = name, Kind = kind, Transform = xf, Radius = radius, Size = size, Material = mat,
            IsLight = isLight, IsInterface = isInterface,
        };
    }

    private static PreviewMaterial WithEmission(PreviewMaterial m, JsonObject em) => new()
    {
        Color = m.Color, IsGlass = m.IsGlass, IsMetal = m.IsMetal, Checker = m.Checker, CheckerB = m.CheckerB,
        CheckerScale = m.CheckerScale, Id = m.Id, Emission = EmissionColor(em),
    };

    private static Vector3 EmissionColor(JsonObject em)
    {
        Vector3 c = new(1, 0.95f, 0.85f);
        if (em["temperature"] is JsonNode t) c = Blackbody((float)Num(t, 6500));
        else if (em["color"] is JsonArray col) c = Vec(col);
        return c * 1.4f;
    }

    /// <summary>Rough sRGB tint of a blackbody (Tanner Helland's fit), for previews only.</summary>
    public static Vector3 Blackbody(float kelvin)
    {
        float t = kelvin / 100f, r, g, b;
        r = t <= 66 ? 255 : 329.698727446f * MathF.Pow(t - 60, -0.1332047592f);
        g = t <= 66 ? 99.4708025861f * MathF.Log(t) - 161.1195681661f : 288.1221695283f * MathF.Pow(t - 60, -0.0755148492f);
        b = t >= 66 ? 255 : (t <= 19 ? 0 : 138.5177312231f * MathF.Log(t - 10) - 305.0447927307f);
        return new Vector3(Math.Clamp(r, 0, 255), Math.Clamp(g, 0, 255), Math.Clamp(b, 0, 255)) / 255f;
    }

    private static PreviewMaterial MakeMaterial(string id, JsonObject? m, JsonObject textures)
    {
        if (m is null) return new PreviewMaterial { Id = id };
        string type = m["type"]?.GetValue<string>() ?? "diffuse";
        Vector3 color = new(0.6f, 0.6f, 0.6f);
        bool glass = false, metal = false, checker = false;
        Vector3 checkerB = default;
        float checkerScale = 1;

        (Vector3 c, bool chk, Vector3 b, float s) ColorOf(JsonNode? n, Vector3 def)
        {
            switch (n)
            {
                case JsonArray a: return (Vec(a), false, default, 1);
                case JsonValue v when v.TryGetValue(out double d): return (new Vector3((float)d), false, default, 1);
                case JsonObject o when o["texture"] is JsonValue tv:
                    if (textures[tv.GetValue<string>()] is JsonObject tex && tex["type"]?.GetValue<string>() == "checker")
                    {
                        var a = ColorOf(tex["a"], new(0.8f)).c;
                        var b = ColorOf(tex["b"], new(0.2f)).c;
                        float s = tex["uv_scale"] is JsonArray us ? (float)Num(us[0], 1) : (float)Num(tex["uv_scale"], 1);
                        return (a, true, b, s);
                    }
                    return (new Vector3(0.6f), false, default, 1);
                case JsonObject o when o["temperature"] is JsonNode t: return (Blackbody((float)Num(t, 6500)), false, default, 1);
            }
            return (def, false, default, 1);
        }

        switch (type)
        {
            case "diffuse":
            case "coated_diffuse":
            {
                var r = ColorOf(m["reflectance"] ?? m["color"], new(0.5f));
                (color, checker, checkerB, checkerScale) = r;
                break;
            }
            case "conductor":
            case "metal":
            case "coated_conductor":
                metal = true;
                color = (m["metal"]?.GetValue<string>()) switch
                {
                    "Au" => new(1.0f, 0.78f, 0.34f),
                    "Cu" => new(0.95f, 0.64f, 0.54f),
                    "Ag" => new(0.95f, 0.93f, 0.88f),
                    "Al" => new(0.91f, 0.92f, 0.92f),
                    _ => ColorOf(m["reflectance"], new(0.9f)).c,
                };
                break;
            case "dielectric":
            case "glass":
                glass = true;
                color = m["medium"] is not null ? new(0.75f, 0.9f, 0.95f) : new(0.9f, 0.95f, 0.97f);
                break;
            case "subsurface":
                color = ColorOf(m["albedo"], new(0.8f)).c;
                break;
            case "interface":
                color = new(0.5f, 0.6f, 0.8f);
                break;
        }
        if (m["sheen"] is JsonObject sh)
        {
            var sc = ColorOf(sh["color"], Vector3.One).c;
            float w = (float)Num(sh["weight"], 1) * 0.25f;
            color = Vector3.Lerp(color, sc, w);
        }
        Vector3 emission = m["emission"] is JsonObject em ? EmissionColor(em) : Vector3.Zero;
        return new PreviewMaterial
        {
            Id = id, Color = color, IsGlass = glass, IsMetal = metal, Checker = checker, CheckerB = checkerB,
            CheckerScale = checkerScale, Emission = emission,
        };
    }
}
