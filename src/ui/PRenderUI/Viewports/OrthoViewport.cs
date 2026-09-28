using System.Numerics;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.Graphics.Canvas.Text;
using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml.Input;
using PRenderUI.Scene;
using Windows.Foundation;
using Windows.UI;

namespace PRenderUI.Viewports;

public enum OrthoAxis { Top, Front, Right }

/// <summary>
/// Orthographic schematic view (Win2D). Drag to pan, wheel to zoom, double-click to frame the scene.
/// </summary>
public sealed class OrthoViewport
{
    private readonly CanvasControl _canvas;
    private readonly OrthoAxis _axis;
    private SceneDocument? _doc;
    private CameraInfo? _camera;
    private float _scale = 80;      // pixels per scene unit
    private Vector2 _center;        // view centre in projected coordinates
    private bool _panning;
    private Point _last;
    private string? _hover;
    private bool _needsFrame = true;

    public event Action<string?>? HoverChanged;

    public OrthoViewport(CanvasControl canvas, OrthoAxis axis)
    {
        _canvas = canvas;
        _axis = axis;
        _canvas.Draw += OnDraw;
        _canvas.PointerPressed += OnPointerPressed;
        _canvas.PointerMoved += OnPointerMoved;
        _canvas.PointerReleased += (_, e) => { _panning = false; _canvas.ReleasePointerCapture(e.Pointer); };
        _canvas.PointerWheelChanged += OnWheel;
        _canvas.DoubleTapped += (_, _) => { Frame(); _canvas.Invalidate(); };
        _canvas.SizeChanged += (_, e) =>
        {
            // Framing needs a real size; defer it until the first layout pass.
            if (_needsFrame && e.NewSize.Width > 0 && e.NewSize.Height > 0) Frame();
            _canvas.Invalidate();
        };
    }

    public string Title => _axis switch
    {
        OrthoAxis.Top => "Top  ·  X / Z",
        OrthoAxis.Front => "Front  ·  X / Y",
        _ => "Right  ·  −Z / Y",
    };

    public void SetScene(SceneDocument? doc, bool frame)
    {
        _doc = doc;
        if (frame) Frame();
        _canvas.Invalidate();
    }

    public void SetCamera(CameraInfo? cam)
    {
        _camera = cam;
        _canvas.Invalidate();
    }

    private Vector2 Project(Vector3 p) => _axis switch
    {
        OrthoAxis.Top => new(p.X, p.Z),
        OrthoAxis.Front => new(p.X, -p.Y),
        _ => new(-p.Z, -p.Y),
    };

    private Vector2 ToScreen(Vector2 w) =>
        (w - _center) * _scale + new Vector2((float)_canvas.ActualWidth / 2, (float)_canvas.ActualHeight / 2);

    private Vector2 FromScreen(Point s) =>
        (new Vector2((float)s.X, (float)s.Y) - new Vector2((float)_canvas.ActualWidth / 2, (float)_canvas.ActualHeight / 2)) / _scale + _center;

    /// <summary>Frame the "interesting" objects (ignores very large planes such as floors).</summary>
    public void Frame()
    {
        if (_doc is null) return;
        if (_canvas.ActualWidth < 1 || _canvas.ActualHeight < 1)
        {
            _needsFrame = true;
            return;
        }
        _needsFrame = false;
        Vector2 min = new(float.MaxValue), max = new(float.MinValue);
        void Add(Vector3 p, float r)
        {
            Vector2 q = Project(p);
            min = Vector2.Min(min, q - new Vector2(r));
            max = Vector2.Max(max, q + new Vector2(r));
        }
        foreach (var o in _doc.Objects)
        {
            if (o.IsInterface) continue;
            if (o.Kind == ShapeKind.Sphere) Add(o.Center, o.WorldRadius);
            else if (o.Kind == ShapeKind.Rect && MathF.Max(o.Size.X, o.Size.Z) > 6) continue;
            else Add(o.Center, 0.25f);
        }
        if (min.X > max.X) { min = new(-2); max = new(2); }
        _center = (min + max) / 2;
        Vector2 ext = Vector2.Max(max - min, new Vector2(0.5f)) * 1.35f;
        float w = MathF.Max(50, (float)_canvas.ActualWidth), h = MathF.Max(50, (float)_canvas.ActualHeight);
        _scale = MathF.Min(w / ext.X, h / ext.Y);
    }

    private void OnPointerPressed(object sender, PointerRoutedEventArgs e)
    {
        _panning = true;
        _last = e.GetCurrentPoint(_canvas).Position;
        _canvas.CapturePointer(e.Pointer);
    }

    private void OnPointerMoved(object sender, PointerRoutedEventArgs e)
    {
        Point p = e.GetCurrentPoint(_canvas).Position;
        if (_panning)
        {
            _center -= new Vector2((float)(p.X - _last.X), (float)(p.Y - _last.Y)) / _scale;
            _last = p;
            _canvas.Invalidate();
            return;
        }
        string? hit = null;
        if (_doc is not null)
        {
            Vector2 w = FromScreen(p);
            float best = float.MaxValue;
            foreach (var o in _doc.Objects.Where(o => o.Kind == ShapeKind.Sphere))
            {
                float d = Vector2.Distance(Project(o.Center), w);
                if (d < o.WorldRadius && d < best)
                {
                    best = d;
                    hit = o.Name;
                }
            }
        }
        if (hit != _hover)
        {
            _hover = hit;
            HoverChanged?.Invoke(hit);
            _canvas.Invalidate();
        }
    }

    private void OnWheel(object sender, PointerRoutedEventArgs e)
    {
        var pt = e.GetCurrentPoint(_canvas);
        Vector2 before = FromScreen(pt.Position);
        _scale *= MathF.Pow(1.0015f, pt.Properties.MouseWheelDelta);
        _scale = Math.Clamp(_scale, 2, 5000);
        Vector2 after = FromScreen(pt.Position);
        _center += before - after;
        _canvas.Invalidate();
        e.Handled = true;
    }

    private static Color C(Vector3 c, byte a = 255) =>
        Color.FromArgb(a, (byte)(Math.Clamp(c.X, 0, 1) * 255), (byte)(Math.Clamp(c.Y, 0, 1) * 255), (byte)(Math.Clamp(c.Z, 0, 1) * 255));

    private void OnDraw(CanvasControl sender, CanvasDrawEventArgs args)
    {
        var ds = args.DrawingSession;
        ds.Antialiasing = CanvasAntialiasing.Antialiased;
        DrawGrid(ds);
        if (_doc is null) return;

        // Draw large planes first, then everything else back-to-front along the view axis.
        var ordered = _doc.Objects.OrderBy(o => o.Kind == ShapeKind.Rect && MathF.Max(o.Size.X, o.Size.Z) > 6 ? 0 : 1)
                                  .ThenBy(o => Depth(o.Center)).ToList();
        foreach (var o in ordered) DrawObject(sender, ds, o);
        foreach (var l in _doc.Lights.Where(l => l.Type is "point" or "spot")) DrawPointLight(ds, l);
        if (_camera is not null) DrawCamera(ds, _camera);
    }

    private float Depth(Vector3 p) => _axis switch
    {
        OrthoAxis.Top => p.Y,
        OrthoAxis.Front => p.Z,
        _ => p.X,
    };

    private void DrawGrid(CanvasDrawingSession ds)
    {
        float w = (float)_canvas.ActualWidth, h = (float)_canvas.ActualHeight;
        Vector2 a = FromScreen(new Point(0, 0)), b = FromScreen(new Point(w, h));
        float step = 1;
        while (step * _scale < 24) step *= 2;
        while (step * _scale > 120) step /= 2;
        var minor = Color.FromArgb(255, 38, 41, 47);
        var major = Color.FromArgb(255, 52, 56, 64);
        for (float x = MathF.Floor(a.X / step) * step; x <= b.X; x += step)
        {
            float sx = ToScreen(new Vector2(x, 0)).X;
            ds.DrawLine(sx, 0, sx, h, MathF.Abs(x % (step * 5)) < 1e-3f ? major : minor, 1);
        }
        for (float y = MathF.Floor(a.Y / step) * step; y <= b.Y; y += step)
        {
            float sy = ToScreen(new Vector2(0, y)).Y;
            ds.DrawLine(0, sy, w, sy, MathF.Abs(y % (step * 5)) < 1e-3f ? major : minor, 1);
        }
        Vector2 o = ToScreen(Vector2.Zero);
        var (hc, vc) = _axis switch
        {
            OrthoAxis.Top => (Color.FromArgb(160, 220, 80, 80), Color.FromArgb(160, 80, 120, 230)),
            OrthoAxis.Front => (Color.FromArgb(160, 220, 80, 80), Color.FromArgb(160, 90, 200, 90)),
            _ => (Color.FromArgb(160, 80, 120, 230), Color.FromArgb(160, 90, 200, 90)),
        };
        ds.DrawLine(0, o.Y, w, o.Y, hc, 1.2f);
        ds.DrawLine(o.X, 0, o.X, h, vc, 1.2f);
    }

    private void DrawObject(ICanvasResourceCreator rc, CanvasDrawingSession ds, SceneObject o)
    {
        var m = o.Material;
        Vector3 col = m.Emission != Vector3.Zero ? Vector3.Min(m.Emission, Vector3.One) : m.Color;
        bool highlight = o.Name == _hover;
        if (o.IsInterface)
        {
            DrawOutline(rc, ds, o, Color.FromArgb(140, 110, 150, 230), dashed: true);
            return;
        }
        if (o.IsProxy)
        {
            FillShape(rc, ds, o, C(col, 90));
            DrawOutline(rc, ds, o, highlight ? Microsoft.UI.Colors.White : C(col * 0.5f + new Vector3(0.35f), 220), dashed: false);
            return;
        }
        switch (o.Kind)
        {
            case ShapeKind.Sphere:
            {
                Vector2 c = ToScreen(Project(o.Center));
                float r = o.WorldRadius * _scale;
                if (m.Emission != Vector3.Zero)
                    ds.FillCircle(c, r * 1.35f, Color.FromArgb(50, 255, 210, 120));
                byte alpha = m.IsGlass ? (byte)90 : (byte)215;
                ds.FillCircle(c, r, C(col, alpha));
                if (m.IsGlass || m.IsMetal)
                    ds.FillCircle(c - new Vector2(r * 0.35f, r * 0.35f), r * 0.22f, Color.FromArgb(90, 255, 255, 255));
                ds.DrawCircle(c, r, highlight ? Microsoft.UI.Colors.White : C(col * 0.6f + new Vector3(0.25f)), highlight ? 2.5f : 1.2f);
                break;
            }
            default:
            {
                bool bigPlane = o.Kind == ShapeKind.Rect && MathF.Max(o.Size.X, o.Size.Z) > 6;
                Color fill = o.IsLight ? Color.FromArgb(200, 255, 225, 140) : C(col, bigPlane ? (byte)40 : (byte)150);
                FillShape(rc, ds, o, fill);
                DrawOutline(rc, ds, o, o.IsLight ? Color.FromArgb(255, 255, 210, 90) : C(col * 0.5f + new Vector3(0.3f), 200), false);
                if (o.IsLight)
                {
                    // Emission direction: local +Y of the light.
                    Vector3 n = Vector3.Normalize(Vector3.TransformNormal(Vector3.UnitY, o.Transform));
                    Vector2 a = ToScreen(Project(o.Center)), b = ToScreen(Project(o.Center + n * 0.8f));
                    ds.DrawLine(a, b, Color.FromArgb(255, 255, 210, 90), 1.5f);
                    ds.FillCircle(b, 3, Color.FromArgb(255, 255, 210, 90));
                }
                break;
            }
        }
    }

    private List<Vector2> Outline(SceneObject o)
    {
        Vector3 T(Vector3 p) => Vector3.Transform(p, o.Transform);
        var pts = new List<Vector2>();
        if (o.Kind == ShapeKind.Rect)
        {
            float hw = o.Size.X / 2, hd = o.Size.Z / 2;
            foreach (var p in new Vector3[] { new(-hw, 0, -hd), new(hw, 0, -hd), new(hw, 0, hd), new(-hw, 0, hd) })
                pts.Add(ToScreen(Project(T(p))));
        }
        else if (o.Kind == ShapeKind.Disk)
        {
            for (int i = 0; i < 48; ++i)
            {
                float a = MathF.Tau * i / 48;
                pts.Add(ToScreen(Project(T(new(o.Radius * MathF.Cos(a), 0, o.Radius * MathF.Sin(a))))));
            }
        }
        return pts;
    }

    private void FillShape(ICanvasResourceCreator rc, CanvasDrawingSession ds, SceneObject o, Color fill)
    {
        if (o.Kind == ShapeKind.Box)
        {
            foreach (var (a, b, c, _, _, _) in o.Triangles())
            {
                using var g = CanvasGeometry.CreatePolygon(rc, new[] { ToScreen(Project(a)), ToScreen(Project(b)), ToScreen(Project(c)) });
                ds.FillGeometry(g, Color.FromArgb((byte)(fill.A / 4), fill.R, fill.G, fill.B));
            }
            return;
        }
        var pts = Outline(o);
        if (pts.Count < 3) return;
        using var geom = CanvasGeometry.CreatePolygon(rc, pts.ToArray());
        ds.FillGeometry(geom, fill);
    }

    private void DrawOutline(ICanvasResourceCreator rc, CanvasDrawingSession ds, SceneObject o, Color color, bool dashed)
    {
        using var style = new CanvasStrokeStyle { DashStyle = dashed ? CanvasDashStyle.Dash : CanvasDashStyle.Solid };
        if (o.Kind == ShapeKind.Box)
        {
            Vector3 h = o.Size / 2;
            var corners = new Vector3[8];
            for (int i = 0; i < 8; ++i)
                corners[i] = Vector3.Transform(new Vector3((i & 1) != 0 ? h.X : -h.X, (i & 2) != 0 ? h.Y : -h.Y, (i & 4) != 0 ? h.Z : -h.Z), o.Transform);
            int[,] edges = { { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 }, { 0, 2 }, { 1, 3 }, { 4, 6 }, { 5, 7 }, { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } };
            for (int e = 0; e < 12; ++e)
                ds.DrawLine(ToScreen(Project(corners[edges[e, 0]])), ToScreen(Project(corners[edges[e, 1]])), color, 1.2f, style);
            return;
        }
        if (o.Kind == ShapeKind.Sphere)
        {
            ds.DrawCircle(ToScreen(Project(o.Center)), o.WorldRadius * _scale, color, 1.2f, style);
            return;
        }
        var pts = Outline(o);
        for (int i = 0; i < pts.Count; ++i) ds.DrawLine(pts[i], pts[(i + 1) % pts.Count], color, 1.2f, style);
    }

    private void DrawPointLight(CanvasDrawingSession ds, LightInfo l)
    {
        Vector2 c = ToScreen(Project(l.Position));
        var col = Color.FromArgb(255, 255, 210, 90);
        ds.FillCircle(c, 5, col);
        for (int i = 0; i < 8; ++i)
        {
            float a = MathF.Tau * i / 8;
            Vector2 d = new(MathF.Cos(a), MathF.Sin(a));
            ds.DrawLine(c + d * 8, c + d * 13, col, 1.2f);
        }
    }

    private void DrawCamera(CanvasDrawingSession ds, CameraInfo cam)
    {
        var col = Color.FromArgb(255, 120, 200, 255);
        Vector3 f = Vector3.Normalize(cam.LookAt - cam.Position);
        Vector3 r = Vector3.Cross(f, cam.Up);
        r = r.LengthSquared() < 1e-12f ? Vector3.UnitX : Vector3.Normalize(r);
        Vector3 u = Vector3.Cross(r, f);
        float dist = Vector3.Distance(cam.Position, cam.LookAt);
        float aspect = _doc is null ? 16f / 9 : (float)_doc.FilmWidth / _doc.FilmHeight;
        Vector2 pos = ToScreen(Project(cam.Position));
        Vector3[] corners;
        if (cam.IsOrtho)
        {
            float hh = cam.OrthoHeight / 2, hw = hh * aspect;
            corners = new[] { cam.LookAt - r * hw - u * hh, cam.LookAt + r * hw - u * hh, cam.LookAt + r * hw + u * hh, cam.LookAt - r * hw + u * hh };
        }
        else
        {
            float t = MathF.Tan(cam.FovY * MathF.PI / 360) * dist;
            Vector3 c = cam.Position + f * dist;
            corners = new[] { c - r * t * aspect - u * t, c + r * t * aspect - u * t, c + r * t * aspect + u * t, c - r * t * aspect + u * t };
            foreach (var p in corners) ds.DrawLine(pos, ToScreen(Project(p)), Color.FromArgb(150, 120, 200, 255), 1);
        }
        for (int i = 0; i < 4; ++i)
            ds.DrawLine(ToScreen(Project(corners[i])), ToScreen(Project(corners[(i + 1) % 4])), col, 1.3f);
        ds.FillCircle(pos, 5, col);
        ds.DrawLine(pos, ToScreen(Project(cam.LookAt)), Color.FromArgb(90, 120, 200, 255), 1);
        ds.DrawText(cam.Name, pos + new Vector2(8, -18), col, new CanvasTextFormat { FontSize = 11 });
    }
}
