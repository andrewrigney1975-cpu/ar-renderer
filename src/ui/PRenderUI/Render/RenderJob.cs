using System.Diagnostics;
using System.Text.Json.Nodes;

namespace PRenderUI.Render;

/// <summary>
/// Runs prender.exe as a child process, parses its line-delimited JSON progress protocol and
/// supports cooperative cancellation through stdin ("cancel"), after which the renderer flushes
/// its outputs and exits with code 3.
/// </summary>
public sealed class RenderJob : IDisposable
{
    private Process? _process;

    public event Action<JsonObject>? EventReceived;
    public event Action<string>? LogLine;
    public event Action<int>? Exited;

    public bool IsRunning => _process is { HasExited: false };

    public void Start(string exe, IEnumerable<string> args, string workingDirectory)
    {
        var psi = new ProcessStartInfo(exe)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            WorkingDirectory = workingDirectory,
        };
        foreach (var a in args) psi.ArgumentList.Add(a);
        _process = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _process.OutputDataReceived += (_, e) =>
        {
            if (string.IsNullOrWhiteSpace(e.Data)) return;
            try
            {
                if (JsonNode.Parse(e.Data) is JsonObject obj) EventReceived?.Invoke(obj);
            }
            catch (System.Text.Json.JsonException)
            {
                LogLine?.Invoke(e.Data);
            }
        };
        _process.ErrorDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data)) LogLine?.Invoke(e.Data);
        };
        _process.Exited += (_, _) =>
        {
            // Let the async readers drain before reporting completion.
            try { _process.WaitForExit(); } catch (InvalidOperationException) { }
            Exited?.Invoke(_process.ExitCode);
        };
        _process.Start();
        _process.BeginOutputReadLine();
        _process.BeginErrorReadLine();
    }

    public void Cancel()
    {
        if (_process is null || _process.HasExited) return;
        try
        {
            _process.StandardInput.WriteLine("cancel");
            _process.StandardInput.Flush();
        }
        catch (IOException)
        {
        }
    }

    public void Kill()
    {
        try
        {
            if (_process is { HasExited: false }) _process.Kill(entireProcessTree: true);
        }
        catch (InvalidOperationException)
        {
        }
    }

    public void Dispose()
    {
        Kill();
        _process?.Dispose();
    }

    /// <summary>Finds prender.exe: $PRENDER_EXE, next to the app, then build\bin in parent directories.</summary>
    public static string? LocateRenderer()
    {
        string? env = Environment.GetEnvironmentVariable("PRENDER_EXE");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string local = Path.Combine(AppContext.BaseDirectory, "prender.exe");
        if (File.Exists(local)) return local;
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "build", "bin", "prender.exe");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    /// <summary>A render device reported by <c>prender --list-devices</c>.</summary>
    public sealed record RenderDevice(string Id, string Name, bool IsGpu, double MemoryMb);

    /// <summary>
    /// Queries the renderer's devices (CPU first). Returns just the CPU when the query fails;
    /// <paramref name="gpuStatus"/> explains why no GPU is listed.
    /// </summary>
    public static async Task<(List<RenderDevice> Devices, string GpuStatus)> ListDevicesAsync(string exe)
    {
        var devices = new List<RenderDevice>();
        string status = "unknown";
        try
        {
            var psi = new ProcessStartInfo(exe)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            foreach (var a in new[] { "--list-devices", "--progress", "json" }) psi.ArgumentList.Add(a);
            using var p = Process.Start(psi)!;
            var stderr = p.StandardError.ReadToEndAsync();
            string output = await p.StandardOutput.ReadToEndAsync();
            await p.WaitForExitAsync();
            await stderr;
            foreach (var line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (JsonNode.Parse(line) is not JsonObject obj || (string?)obj["event"] != "devices") continue;
                status = (string?)obj["gpu_status"] ?? status;
                foreach (var d in obj["devices"]?.AsArray() ?? new JsonArray())
                {
                    if (d is null) continue;
                    devices.Add(new RenderDevice((string?)d["id"] ?? "cpu", (string?)d["name"] ?? "?",
                                                 (bool?)d["gpu"] ?? false, (double?)d["memory_mb"] ?? 0));
                }
            }
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or IOException or InvalidOperationException
                                       or System.Text.Json.JsonException)
        {
            status = ex.Message;
        }
        if (devices.Count == 0) devices.Add(new RenderDevice("cpu", "CPU", false, 0));
        return (devices, status);
    }

    /// <summary>Finds the bundled sample scene by walking up from the app directory.</summary>
    public static string? LocateSampleScene()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "scenes", "sphere-pyramid", "scene.prscene.json");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
}
