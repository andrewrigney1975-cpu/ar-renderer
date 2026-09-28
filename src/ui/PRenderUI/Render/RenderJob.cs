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
