using System.Numerics;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Text;
using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml.Input;
using PRenderUI.Scene;
using Windows.Foundation;
using Windows.Graphics.DirectX;
using Windows.UI;

namespace PRenderUI.Viewports;

/// <summary>
/// Camera view with a quick ray-cast preview, framed to the film aspect ratio.
/// Left-drag orbits the look-at point, right/middle-drag pans, wheel dollies, double-click resets.
/// </summary>
public sealed class CameraViewport
{
    private readonly CanvasControl _canvas;
    private readonly DispatcherQueue _dispatcher;
    private PreviewTracer? _tracer;
    private CameraInfo? _camera;
    private CanvasBitmap? _bitmap;
    private int _filmW = 16, _filmH = 9;
    private bool _busy, _pending, _interactive;
    private int _generation;
    private enum DragMode { None, Orbit, Pan }
    private DragMode _drag;
    private Point _last;

    /// <summary>Raised when the user moves the camera.</summary>
    public event Action? CameraEdited;
    /// <summary>Raised on double-click (request to restore the scene's camera).</summary>
    public event Action? ResetRequested;

    public CameraViewport(CanvasControl canvas)
    {
        _canvas = canvas;
        _dispatcher = DispatcherQueue.GetForCurrentThread();
        _canvas.Draw += OnDraw;
        _canvas.SizeChanged += (_, _) => RequestRender(false);
        _canvas.PointerPressed += OnPointerPressed;
        _canvas.PointerMoved += OnPointerMoved;
        _canvas.PointerReleased += OnPointerReleased;
        _canvas.PointerWheelChanged += OnWheel;
        _canvas.DoubleTapped += (_, _) => ResetRequested?.Invoke();
    }

    public CameraInfo? Camera => _camera;

    public void SetScene(SceneDocument? doc)
    {
        _tracer = doc is null ? null : new PreviewTracer(doc);
        RequestRender(false);
    }

    public void SetFilmAspect(int w, int h)
    {
        _filmW = Math.Max(1, w);
        _filmH = Math.Max(1, h);
        RequestRender(false);
    }

    public void SetCamera(CameraInfo? cam)
    {
        _camera = cam;
        RequestRender(false);
    }

    private Rect FilmRect()
    {
        double w = _canvas.ActualWidth, h = _canvas.ActualHeight;
        double aspect = (double)_filmW / _filmH;
        double fw = w, fh = w / aspect;
        if (fh > h) { fh = h; fw = h * aspect; }
        return new Rect((w - fw) / 2, (h - fh) / 2, fw, fh);
    }

    private void RequestRender(bool interactive)
    {
        _interactive = interactive;
        if (_busy)
        {
            _pending = true;
            return;
        }
        if (_tracer is null || _camera is null || _canvas.ActualWidth < 8 || _canvas.ActualHeight < 8)
        {
            _bitmap = null;
            _canvas.Invalidate();
            return;
        }
        Rect fr = FilmRect();
        double quality = interactive ? 0.4 : 1.0;
        double dpi = _canvas.XamlRoot?.RasterizationScale ?? 1.0;
        int w = Math.Max(16, (int)(fr.Width * quality * Math.Min(dpi, 1.5)));
        int h = Math.Max(9, (int)(fr.Height * quality * Math.Min(dpi, 1.5)));
        var tracer = _tracer;
        var cam = _camera.Clone();
        int gen = ++_generation;
        _busy = true;
        _pending = false;
        Task.Run(() => tracer.Render(cam, w, h)).ContinueWith(t =>
        {
            _dispatcher.TryEnqueue(() =>
            {
                _busy = false;
                if (t.IsCompletedSuccessfully && gen == _generation && _canvas.Device is not null)
                {
                    _bitmap?.Dispose();
                    _bitmap = CanvasBitmap.CreateFromBytes(_canvas, t.Result, w, h, DirectXPixelFormat.B8G8R8A8UIntNormalized);
                    _canvas.Invalidate();
                }
                if (_pending) RequestRender(_interactive);
                else if (_interactive && _drag == DragMode.None) RequestRender(false);
            });
        });
    }

    private void OnDraw(CanvasControl sender, CanvasDrawEventArgs args)
    {
        var ds = args.DrawingSession;
        Rect fr = FilmRect();
        if (_bitmap is not null)
            ds.DrawImage(_bitmap, fr, _bitmap.Bounds, 1, CanvasImageInterpolation.Linear);
        ds.DrawRectangle(fr, Color.FromArgb(255, 70, 76, 86), 1);
        if (_camera is not null)
        {
            string info = _camera.IsOrtho ? $"{_camera.Name} · ortho {_camera.OrthoHeight:0.##}" : $"{_camera.Name} · fov {_camera.FovY:0.#}°";
            ds.DrawText(info, (float)fr.X + 8, (float)(fr.Bottom - 20), Color.FromArgb(200, 220, 225, 235),
                        new CanvasTextFormat { FontSize = 11 });
        }
        else
        {
            ds.DrawText("No camera", (float)fr.X + 8, (float)fr.Y + 8, Microsoft.UI.Colors.Gray);
        }
    }

    private void OnPointerPressed(object sender, PointerRoutedEventArgs e)
    {
        var pt = e.GetCurrentPoint(_canvas);
        _last = pt.Position;
        _drag = pt.Properties.IsLeftButtonPressed ? DragMode.Orbit : DragMode.Pan;
        _canvas.CapturePointer(e.Pointer);
    }

    private void OnPointerReleased(object sender, PointerRoutedEventArgs e)
    {
        _drag = DragMode.None;
        _canvas.ReleasePointerCapture(e.Pointer);
        RequestRender(false);
    }

    private void OnPointerMoved(object sender, PointerRoutedEventArgs e)
    {
        if (_drag == DragMode.None || _camera is null) return;
        Point p = e.GetCurrentPoint(_canvas).Position;
        float dx = (float)(p.X - _last.X), dy = (float)(p.Y - _last.Y);
        _last = p;
        if (_drag == DragMode.Orbit && !_camera.IsOrtho) Orbit(dx, dy);
        else Pan(dx, dy);
        CameraEdited?.Invoke();
        RequestRender(true);
    }

    private void OnWheel(object sender, PointerRoutedEventArgs e)
    {
        if (_camera is null) return;
        int delta = e.GetCurrentPoint(_canvas).Properties.MouseWheelDelta;
        float k = MathF.Pow(0.9f, delta / 120f);
        if (_camera.IsOrtho) _camera.OrthoHeight = Math.Clamp(_camera.OrthoHeight * k, 0.05f, 1000f);
        else
        {
            Vector3 off = _camera.Position - _camera.LookAt;
            float len = Math.Clamp(off.Length() * k, 0.05f, 1000f);
            _camera.Position = _camera.LookAt + Vector3.Normalize(off) * len;
        }
        CameraEdited?.Invoke();
        RequestRender(true);
        e.Handled = true;
    }

    private void Orbit(float dx, float dy)
    {
        Vector3 off = _camera!.Position - _camera.LookAt;
        float r = off.Length();
        float yaw = MathF.Atan2(off.X, off.Z) - dx * 0.008f;
        float pitch = Math.Clamp(MathF.Asin(Math.Clamp(off.Y / r, -1, 1)) + dy * 0.008f, -1.5f, 1.5f);
        off = new Vector3(MathF.Cos(pitch) * MathF.Sin(yaw), MathF.Sin(pitch), MathF.Cos(pitch) * MathF.Cos(yaw)) * r;
        _camera.Position = _camera.LookAt + off;
    }

    private void Pan(float dx, float dy)
    {
        var cam = _camera!;
        Vector3 f = Vector3.Normalize(cam.LookAt - cam.Position);
        Vector3 right = Vector3.Normalize(Vector3.Cross(f, cam.Up));
        Vector3 up = Vector3.Cross(right, f);
        Rect fr = FilmRect();
        float unitsPerPixel = cam.IsOrtho
            ? cam.OrthoHeight / (float)Math.Max(1, fr.Height)
            : 2 * MathF.Tan(cam.FovY * MathF.PI / 360) * Vector3.Distance(cam.Position, cam.LookAt) / (float)Math.Max(1, fr.Height);
        Vector3 delta = (-right * dx + up * dy) * unitsPerPixel;
        cam.Position += delta;
        cam.LookAt += delta;
    }
}
