using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using PRenderUI.Render;
using PRenderUI.Scene;
using PRenderUI.Viewports;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;

namespace PRenderUI;

public sealed partial class MainWindow : Window
{
    private readonly OrthoViewport _top, _front, _right;
    private readonly CameraViewport _cameraView;
    private SceneDocument? _doc;
    private string? _scenePath;
    private readonly Dictionary<string, CameraInfo> _cameras = new();   // working copies (may be edited)
    private readonly HashSet<string> _editedCameras = new();
    private RenderJob? _job;
    private string? _rendererExe;
    private string? _outputDir;
    private string? _finalImage;
    private bool _suppressEvents;
    private List<string>? _lastArgs;      // arguments of the last render (for Continue)
    private double _lastSamples;
    private string? _lastCheckpoint;
    private int _previewVersion;

    private static readonly (string Label, int W, int H)[] Resolutions =
    {
        ("480 × 270", 480, 270), ("960 × 540", 960, 540), ("1280 × 720", 1280, 720), ("1920 × 1080", 1920, 1080),
        ("800 × 800", 800, 800), ("4K UHD · 3840 × 2160", 3840, 2160), ("8K UHD · 7680 × 4320", 7680, 4320),
        ("16K · 15360 × 8640", 15360, 8640),
    };

    public MainWindow()
    {
        InitializeComponent();
        Title = "PRender — spectral MLT renderer";
        SystemBackdrop = new MicaBackdrop();
        AppWindow.Resize(new Windows.Graphics.SizeInt32(1640, 980));

        _top = new OrthoViewport(TopCanvas, OrthoAxis.Top);
        _front = new OrthoViewport(FrontCanvas, OrthoAxis.Front);
        _right = new OrthoViewport(RightCanvas, OrthoAxis.Right);
        TopLabel.Text = _top.Title;
        FrontLabel.Text = _front.Title;
        RightLabel.Text = _right.Title;
        foreach (var v in new[] { _top, _front, _right })
            v.HoverChanged += name => HoverText.Text = name ?? "";

        _cameraView = new CameraViewport(CameraCanvas);
        _cameraView.CameraEdited += OnCameraEdited;
        _cameraView.ResetRequested += () => OnResetCameraClick(this, new RoutedEventArgs());

        _rendererExe = RenderJob.LocateRenderer();
        RendererText.Text = _rendererExe is null ? "prender.exe not found (set PRENDER_EXE or build the renderer)" : $"renderer: {_rendererExe}";
        DeviceBox.Items.Add(new ComboBoxItem { Content = "CPU", Tag = "cpu" });
        DeviceBox.SelectedIndex = 0;
        _devicesReady = PopulateDevicesAsync();

        Closed += (_, _) => _job?.Dispose();

        // A scene path on the command line wins over the bundled sample.
        string? fromArgs = Environment.GetCommandLineArgs().Skip(1)
            .FirstOrDefault(a => !a.StartsWith("--", StringComparison.Ordinal) && File.Exists(a) && !a.EndsWith(".png", StringComparison.OrdinalIgnoreCase));
        string? sample = fromArgs ?? RenderJob.LocateSampleScene();
        if (sample is not null) LoadScene(sample, resetUi: true);
        else StatusText.Text = "Open a scene to begin.";
    }

    // ---------------------------------------------------------------------------------------
    // Scene loading

    private List<string> ActiveRigs() =>
        RigPanel.Children.OfType<CheckBox>().Where(c => c.IsChecked == true).Select(c => (string)c.Tag).ToList();

    private void LoadScene(string path, bool resetUi)
    {
        try
        {
            List<string>? rigs = resetUi ? null : ActiveRigs();
            _rendererExe ??= RenderJob.LocateRenderer();
            var doc = SceneDocument.Load(path, rigs, _rendererExe);
            _doc = doc;
            _scenePath = path;
            ScenePathText.Text = path;
            ErrorBar.IsOpen = false;
            foreach (var w in doc.Warnings) AppendLog("warning: " + w);

            _suppressEvents = true;
            if (resetUi)
            {
                _cameras.Clear();
                _editedCameras.Clear();
                RigPanel.Children.Clear();
                foreach (var r in doc.RigNames)
                {
                    var cb = new CheckBox { Content = r, Tag = r, IsChecked = doc.DefaultActiveRigs.Contains(r), MinWidth = 0 };
                    cb.Click += (_, _) => { if (_scenePath is not null) LoadScene(_scenePath, resetUi: false); };
                    RigPanel.Children.Add(cb);
                }
                SelectIntegrator(GpuSelected ? "path" : doc.Integrator);
                ResolutionBox.Items.Clear();
                ResolutionBox.Items.Add(new ComboBoxItem { Content = $"Scene ({doc.FilmWidth} × {doc.FilmHeight})", Tag = (doc.FilmWidth, doc.FilmHeight) });
                foreach (var (label, w, h) in Resolutions)
                    ResolutionBox.Items.Add(new ComboBoxItem { Content = label, Tag = (w, h) });
                ResolutionBox.SelectedIndex = 0;
                SamplesBox.Value = IsMlt() ? doc.MutationsPerPixel : doc.Spp;
            }
            foreach (var (name, cam) in doc.Cameras)
                if (!_editedCameras.Contains(name)) _cameras[name] = cam.Clone();
            string current = (CameraBox.SelectedItem as string) ?? doc.DefaultCamera;
            CameraBox.Items.Clear();
            // Perspective cameras first: they are the ones that can be rendered with MLT/BDPT.
            foreach (var c in _cameras.Values.OrderBy(c => c.IsOrtho).ThenBy(c => c.Name == doc.DefaultCamera ? 0 : 1))
                CameraBox.Items.Add(c.Name);
            CameraBox.SelectedItem = _cameras.ContainsKey(current) ? current : doc.DefaultCamera;
            _suppressEvents = false;

            _top.SetScene(doc, resetUi);
            _front.SetScene(doc, resetUi);
            _right.SetScene(doc, resetUi);
            _cameraView.SetScene(doc);
            ApplyFilmAspect();
            ApplyCamera();
            StatusText.Text = $"Loaded {Path.GetFileName(path)} · {doc.Objects.Count} objects · {doc.Lights.Count} lights";
        }
        catch (Exception ex) when (ex is IOException or JsonException or InvalidDataException or InvalidOperationException or UnauthorizedAccessException)
        {
            ShowError("Could not load scene", ex.Message);
        }
    }

    private void SelectIntegrator(string type)
    {
        foreach (ComboBoxItem item in IntegratorBox.Items)
            if ((string)item.Tag == type) IntegratorBox.SelectedItem = item;
        if (IntegratorBox.SelectedItem is null) IntegratorBox.SelectedIndex = 0;
        UpdateSamplesHeader();
    }

    private string SelectedIntegrator => (IntegratorBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "mmlt";
    private bool IsMlt() => SelectedIntegrator is "mmlt" or "pssmlt";

    private void UpdateSamplesHeader() => SamplesBox.Header = IsMlt() ? "Mutations / pixel" : "Samples / pixel";

    private (int W, int H) SelectedResolution =>
        (ResolutionBox.SelectedItem as ComboBoxItem)?.Tag is ValueTuple<int, int> t ? t : (_doc?.FilmWidth ?? 960, _doc?.FilmHeight ?? 540);

    private void ApplyFilmAspect()
    {
        var (w, h) = SelectedResolution;
        _cameraView.SetFilmAspect(w, h);
    }

    private CameraInfo? SelectedCamera =>
        CameraBox.SelectedItem is string n && _cameras.TryGetValue(n, out var c) ? c : null;

    private void ApplyCamera()
    {
        var cam = SelectedCamera;
        _cameraView.SetCamera(cam);
        _top.SetCamera(cam);
        _front.SetCamera(cam);
        _right.SetCamera(cam);
        ResetCameraButton.Visibility = cam is not null && _editedCameras.Contains(cam.Name) ? Visibility.Visible : Visibility.Collapsed;
    }

    // ---------------------------------------------------------------------------------------
    // UI events

    private async void OnOpenClick(object sender, RoutedEventArgs e)
    {
        var picker = new FileOpenPicker();
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(this));
        picker.FileTypeFilter.Add(".json");
        picker.FileTypeFilter.Add(".gltf");
        picker.FileTypeFilter.Add(".glb");
        picker.FileTypeFilter.Add(".pbrt");
        var file = await picker.PickSingleFileAsync();
        if (file is not null) LoadScene(file.Path, resetUi: true);
    }

    private void OnReloadClick(object sender, RoutedEventArgs e)
    {
        if (_scenePath is not null) LoadScene(_scenePath, resetUi: false);
    }

    private void OnCameraChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_suppressEvents) ApplyCamera();
    }

    // ---------------------------------------------------------------------------------------
    // Render device

    private Task _devicesReady = Task.CompletedTask;

    private string SelectedDevice => (DeviceBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "cpu";
    private bool GpuSelected => SelectedDevice != "cpu";

    private async Task PopulateDevicesAsync()
    {
        if (_rendererExe is null) return;
        var (devices, gpuStatus) = await RenderJob.ListDevicesAsync(_rendererExe);
        App.Log($"devices: {string.Join(", ", devices.Select(d => d.Id))} (gpu: {gpuStatus})");
        DeviceBox.Items.Clear();
        foreach (var d in devices)
        {
            string label = d.IsGpu ? $"{d.Name} ({d.MemoryMb / 1024:0} GB)" : d.Id == "cpu" ? $"CPU · {Environment.ProcessorCount} threads" : d.Name;
            DeviceBox.Items.Add(new ComboBoxItem { Content = label, Tag = d.Id });
        }
        DeviceBox.SelectedIndex = 0;
        // Developer aid: --device <id> preselects a device (e.g. with --autorender).
        string[] argv = Environment.GetCommandLineArgs();
        int di = Array.IndexOf(argv, "--device");
        if (di >= 0 && di + 1 < argv.Length)
            foreach (ComboBoxItem item in DeviceBox.Items)
                if ((string)item.Tag == argv[di + 1] || (argv[di + 1] == "gpu" && (string)item.Tag == "gpu:0")) DeviceBox.SelectedItem = item;
        if (!devices.Any(d => d.IsGpu)) AppendLog($"GPU rendering unavailable: {gpuStatus}");
    }

    private void OnDeviceChanged(object sender, SelectionChangedEventArgs e)
    {
        // The GPU implements the path tracer only: lock the integrator while a GPU is selected.
        bool gpu = GpuSelected;
        foreach (ComboBoxItem item in IntegratorBox.Items)
            item.IsEnabled = !gpu || (string)item.Tag == "path";
        if (gpu && SelectedIntegrator != "path")
        {
            SelectIntegrator("path");
            if (_doc is not null) SamplesBox.Value = _doc.Spp;
        }
    }

    private void OnIntegratorChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressEvents) return;
        UpdateSamplesHeader();
        if (_doc is not null) SamplesBox.Value = IsMlt() ? _doc.MutationsPerPixel : _doc.Spp;
    }

    private void OnResolutionChanged(object sender, SelectionChangedEventArgs e)
    {
        ApplyFilmAspect();
        UpdateResolutionInfo();
    }

    // Film memory estimate (3 doubles per pixel, like the renderer) and a warning when it gets
    // close to the machine's available memory.
    private void UpdateResolutionInfo()
    {
        var (w, h) = SelectedResolution;
        double filmGB = (double)w * h * 3 * sizeof(double) / (1 << 30);
        double totalGB = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (double)(1 << 30);
        double mp = (double)w * h / 1e6;
        string info = $"{mp:0.#} MP · film ≈ {(filmGB < 1 ? $"{filmGB * 1024:0} MB" : $"{filmGB:0.0} GB")}";
        if (filmGB >= 0.5) info += $" · checkpoint ≈ {filmGB:0.0} GB on disk · outputs streamed, preview downscaled";
        if (totalGB > 0 && filmGB > totalGB * 0.5)
        {
            info += $" — warning: this machine has {totalGB:0} GB of memory";
            ResolutionInfo.Foreground = new SolidColorBrush(Microsoft.UI.Colors.OrangeRed);
        }
        else
        {
            ResolutionInfo.ClearValue(TextBlock.ForegroundProperty);
        }
        ResolutionInfo.Text = info;
    }

    private void OnCameraEdited()
    {
        var cam = _cameraView.Camera;
        if (cam is null) return;
        _editedCameras.Add(cam.Name);
        ResetCameraButton.Visibility = Visibility.Visible;
        _top.SetCamera(cam);
        _front.SetCamera(cam);
        _right.SetCamera(cam);
    }

    private void OnResetCameraClick(object sender, RoutedEventArgs e)
    {
        var cam = SelectedCamera;
        if (cam is null || _doc is null || !_doc.Cameras.TryGetValue(cam.Name, out var original)) return;
        _cameras[cam.Name] = original.Clone();
        _editedCameras.Remove(cam.Name);
        ApplyCamera();
    }

    private void OnOpenFolderClick(object sender, RoutedEventArgs e)
    {
        if (_outputDir is not null && Directory.Exists(_outputDir))
            Process.Start(new ProcessStartInfo("explorer.exe", $"\"{_outputDir}\"") { UseShellExecute = true });
    }

    private void OnImageTapped(object sender, RoutedEventArgs e)
    {
        if (_finalImage is not null && File.Exists(_finalImage))
            Process.Start(new ProcessStartInfo(_finalImage) { UseShellExecute = true });
    }

    // ---------------------------------------------------------------------------------------
    // Rendering

    private void OnRenderClick(object sender, RoutedEventArgs e)
    {
        if (_job is { IsRunning: true })
        {
            _job.Cancel();
            StatusText.Text = "Cancelling — writing the current estimate…";
            RenderButton.IsEnabled = false;
            return;
        }
        StartRender();
    }

    private void StartRender()
    {
        if (_scenePath is null || _doc is null) return;
        _rendererExe ??= RenderJob.LocateRenderer();
        if (_rendererExe is null)
        {
            ShowError("Renderer not found", "Build prender.exe (build.cmd) or set the PRENDER_EXE environment variable.");
            return;
        }
        var cam = SelectedCamera;
        if (cam is null) return;
        if (cam.IsOrtho && SelectedIntegrator != "path")
        {
            ShowError("Orthographic camera", "MLT and BDPT need a perspective camera; switching to the path tracer for this render.");
        }

        string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
        string sceneName = Path.GetFileNameWithoutExtension(_scenePath).Replace(".prscene", "");
        _outputDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "PRenderUI", "renders",
                                  $"{sceneName}-{stamp}");
        Directory.CreateDirectory(_outputDir);

        // Camera edits are written as a small override scene that includes the original.
        string scenePath = _scenePath;
        if (_editedCameras.Contains(cam.Name))
        {
            var overrideDoc = new JsonObject
            {
                ["include"] = new JsonArray(Path.GetFullPath(_scenePath)),
                ["cameras"] = new JsonObject { [cam.Name] = CameraJson(cam) },
            };
            scenePath = Path.Combine(_outputDir, "camera-override.prscene.json");
            File.WriteAllText(scenePath, overrideDoc.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        }

        var (w, h) = SelectedResolution;
        string integrator = cam.IsOrtho ? "path" : SelectedIntegrator;
        bool mlt = integrator is "mmlt" or "pssmlt";
        var args = new List<string>
        {
            scenePath, "--camera", cam.Name, "--integrator", integrator, "--res", $"{w}x{h}",
            "--out", Path.Combine(_outputDir, "render.exr"), "--out", Path.Combine(_outputDir, "render.png"),
            "--preview", Path.Combine(_outputDir, "preview.png"), "--preview-interval", "1",
            "--progress", "json", "--control", "stdin",
            "--seed", ((long)SeedBox.Value).ToString(CultureInfo.InvariantCulture),
            "--device", SelectedDevice,
        };
        double samples = double.IsNaN(SamplesBox.Value) ? 64 : SamplesBox.Value;
        args.Add(mlt ? "--mutations" : "--spp");
        args.Add(samples.ToString(CultureInfo.InvariantCulture));
        _lastCheckpoint = Path.Combine(_outputDir, "state.prck");
        args.AddRange(new[] { "--checkpoint", _lastCheckpoint, "--checkpoint-interval", "30" });
        if (AovCheck.IsChecked == true)
            foreach (var aov in new[] { "albedo", "normal", "depth" })
            {
                args.Add("--aov");
                args.Add($"{aov}={Path.Combine(_outputDir, aov + ".exr")}");
            }
        if (!double.IsNaN(TimeBox.Value) && TimeBox.Value > 0)
        {
            args.Add("--time");
            args.Add($"{TimeBox.Value.ToString(CultureInfo.InvariantCulture)}s");
        }
        var rigs = ActiveRigs();
        if (rigs.Count == 0) args.Add("--no-rigs");
        foreach (var r in rigs) { args.Add("--rig"); args.Add(r); }

        _lastArgs = new List<string>(args);
        _lastSamples = samples;
        LaunchRenderer(args);
    }

    // Resume the last render from its checkpoint with twice the sample budget.
    private void OnContinueClick(object sender, RoutedEventArgs e)
    {
        if (_lastArgs is null || _lastCheckpoint is null || !File.Exists(_lastCheckpoint) || _job is { IsRunning: true }) return;
        int i = _lastArgs.FindIndex(a => a is "--mutations" or "--spp");
        _lastSamples *= 2;
        if (i >= 0) _lastArgs[i + 1] = _lastSamples.ToString(CultureInfo.InvariantCulture);
        var args = new List<string>(_lastArgs) { "--resume", _lastCheckpoint };
        LaunchRenderer(args);
    }

    private void LaunchRenderer(List<string> args)
    {
        if (_rendererExe is null || _outputDir is null) return;
        _finalImage = null;
        _previewVersion++;
        LogBox.Text = $"> prender {string.Join(' ', args.Select(a => a.Contains(' ') ? $"\"{a}\"" : a))}\n";
        Progress.Value = 0;
        Progress.IsIndeterminate = true;
        StatusText.Text = "Starting…";
        StatsText.Text = "";
        ErrorBar.IsOpen = false;
        SetRendering(true);
        OpenFolderButton.IsEnabled = true;
        OpenImageButton.IsEnabled = false;

        _job?.Dispose();
        _job = new RenderJob();
        _job.EventReceived += ev => DispatcherQueue.TryEnqueue(() => HandleEvent(ev));
        _job.LogLine += line => DispatcherQueue.TryEnqueue(() => AppendLog(line));
        _job.Exited += code => DispatcherQueue.TryEnqueue(() => OnRenderExited(code));
        try
        {
            _job.Start(_rendererExe, args, _outputDir);
        }
        catch (System.ComponentModel.Win32Exception ex)
        {
            SetRendering(false);
            ShowError("Could not start the renderer", ex.Message);
        }
    }

    private static JsonObject CameraJson(CameraInfo cam)
    {
        var j = (JsonObject)cam.Source.DeepClone();
        j["type"] = cam.Type;
        j["position"] = new JsonArray((double)cam.Position.X, (double)cam.Position.Y, (double)cam.Position.Z);
        j["look_at"] = new JsonArray((double)cam.LookAt.X, (double)cam.LookAt.Y, (double)cam.LookAt.Z);
        j["up"] = new JsonArray((double)cam.Up.X, (double)cam.Up.Y, (double)cam.Up.Z);
        j.Remove("direction");
        if (cam.IsOrtho) j["ortho_height"] = (double)cam.OrthoHeight;
        else
        {
            j["fov_y"] = (double)cam.FovY;
            j.Remove("focal_length_mm");
            // Keep focus on the look-at point after the camera has been moved.
            if (j.ContainsKey("focus_on") || j.ContainsKey("focus_distance"))
            {
                j.Remove("focus_on");
                j["focus_distance"] = (double)Vector3.Distance(cam.Position, cam.LookAt);
            }
        }
        return j;
    }

    private void HandleEvent(JsonObject ev)
    {
        string type = ev["event"]?.GetValue<string>() ?? "";
        switch (type)
        {
            case "stage":
            {
                string name = ev["name"]?.GetValue<string>() ?? "";
                StatusText.Text = name switch
                {
                    "load" => "Loading scene…",
                    "bootstrap" => "Bootstrapping Markov chains…",
                    "render" => "Rendering…",
                    "write" => "Writing images…",
                    _ => name,
                };
                break;
            }
            case "scene":
                AppendLog($"scene: {ev["primitives"]} primitives, {ev["lights"]} lights, {ev["media"]} media, " +
                          $"{ev["width"]}x{ev["height"]}, {ev["integrator"]}, {ev["threads"]} threads");
                break;
            case "progress":
            {
                double f = ev["fraction"]?.GetValue<double>() ?? 0;
                string stage = ev["stage"]?.GetValue<string>() ?? "";
                Progress.IsIndeterminate = false;
                Progress.Value = Math.Clamp(f * 100, 0, 100);
                double elapsed = ev["elapsed"]?.GetValue<double>() ?? 0;
                double spp = ev["samples_per_pixel"]?.GetValue<double>() ?? 0;
                string unit = IsMlt() ? "mutations/px" : "spp";
                string acc = ev["accept_rate"] is JsonNode a ? $" · accept {a.GetValue<double>() * 100:0.0}%" : "";
                StatsText.Text = stage == "bootstrap"
                    ? $"bootstrap {f * 100:0}% · {elapsed:0.0}s"
                    : $"{spp:0.#} {unit}{acc} · {elapsed:0.0}s";
                break;
            }
            case "preview":
            {
                string? file = ev["file"]?.GetValue<string>();
                if (file is not null) _ = ShowImageAsync(file, ++_previewVersion);
                break;
            }
            case "done":
            {
                double elapsed = ev["elapsed"]?.GetValue<double>() ?? 0;
                bool cancelled = ev["cancelled"]?.GetValue<bool>() ?? false;
                var outputs = ev["outputs"] as JsonArray;
                _finalImage = outputs?.Select(o => o?.GetValue<string>()).FirstOrDefault(p => p?.EndsWith(".png", StringComparison.OrdinalIgnoreCase) == true);
                if (_finalImage is not null) _ = ShowImageAsync(_finalImage, ++_previewVersion);
                StatusText.Text = cancelled ? $"Cancelled after {elapsed:0.0}s (partial result saved)" : $"Done in {elapsed:0.0}s";
                Progress.Value = 100;
                OpenImageButton.IsEnabled = _finalImage is not null;
                break;
            }
            case "error":
                ShowError("Render failed", ev["message"]?.GetValue<string>() ?? "unknown error");
                break;
        }
    }

    private void OnRenderExited(int code)
    {
        SetRendering(false);
        Progress.IsIndeterminate = false;
        if (code is not (0 or 3)) StatusText.Text = $"Renderer exited with code {code}";
        AppendLog($"[exit code {code}]");
    }

    private void SetRendering(bool rendering)
    {
        RenderButton.IsEnabled = true;
        RenderButtonText.Text = rendering ? "Cancel" : "Render";
        ContinueButton.IsEnabled = !rendering && _lastCheckpoint is not null && File.Exists(_lastCheckpoint);
        RenderIcon.Glyph = rendering ? "" : "";
        OpenButton.IsEnabled = !rendering;
        ReloadButton.IsEnabled = !rendering;
    }

    private async Task ShowImageAsync(string path, int version)
    {
        try
        {
            byte[] bytes;
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            {
                bytes = new byte[fs.Length];
                await fs.ReadExactlyAsync(bytes);
            }
            if (version != _previewVersion) return;
            using var stream = new InMemoryRandomAccessStream();
            await stream.WriteAsync(bytes.AsBuffer());
            stream.Seek(0);
            // Decode huge results (8K/16K) at display size only.
            var bmp = new BitmapImage { DecodePixelWidth = 2048, DecodePixelType = DecodePixelType.Logical };
            await bmp.SetSourceAsync(stream);
            if (version != _previewVersion) return;
            RenderImage.Source = bmp;
            ImagePlaceholder.Visibility = Visibility.Collapsed;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Runtime.InteropServices.COMException)
        {
            // A preview may be replaced while being read; the next one will succeed.
        }
    }

    public void StartRenderFromCommandLine() => DispatcherQueue.TryEnqueue(async () =>
    {
        await _devicesReady;
        StartRender();
    });

    public async Task SaveScreenshotAndExitAsync(string path, double delaySeconds)
    {
        try
        {
            await Task.Delay(TimeSpan.FromSeconds(delaySeconds));
            var rtb = new RenderTargetBitmap();
            await rtb.RenderAsync(RootGrid);
            var pixels = await rtb.GetPixelsAsync();
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.ReadWrite))
            {
                var encoder = await Windows.Graphics.Imaging.BitmapEncoder.CreateAsync(
                    Windows.Graphics.Imaging.BitmapEncoder.PngEncoderId, stream.AsRandomAccessStream());
                encoder.SetPixelData(Windows.Graphics.Imaging.BitmapPixelFormat.Bgra8, Windows.Graphics.Imaging.BitmapAlphaMode.Premultiplied,
                                     (uint)rtb.PixelWidth, (uint)rtb.PixelHeight, 96, 96, pixels.ToArray());
                await encoder.FlushAsync();
            }
            App.Log($"screenshot saved: {path} ({rtb.PixelWidth}x{rtb.PixelHeight})");
        }
        catch (Exception ex)
        {
            App.Log($"screenshot failed: {ex}");
        }
        Close();
    }

    private void AppendLog(string line)
    {
        LogBox.Text += line + "\n";
        if (LogBox.Text.Length > 60000) LogBox.Text = LogBox.Text[^40000..];
        LogBox.SelectionStart = LogBox.Text.Length;
    }

    private void ShowError(string title, string message)
    {
        ErrorBar.Title = title;
        ErrorBar.Message = message;
        ErrorBar.IsOpen = true;
        AppendLog($"error: {title}: {message}");
    }
}
