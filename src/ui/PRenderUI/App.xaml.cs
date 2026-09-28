using Microsoft.UI.Xaml;

namespace PRenderUI;

public partial class App : Application
{
    private Window? _window;

    public static string LogPath { get; } = Path.Combine(Path.GetTempPath(), "PRenderUI.log");

    public App()
    {
        InitializeComponent();
        UnhandledException += (_, e) =>
        {
            Log($"unhandled: {e.Exception}");
            e.Handled = true;
        };
    }

    public static void Log(string message)
    {
        try { File.AppendAllText(LogPath, $"{DateTime.Now:HH:mm:ss.fff} {message}{Environment.NewLine}"); }
        catch (IOException) { }
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        try
        {
            var window = new MainWindow();
            _window = window;
            _window.Activate();
            // Developer aids: --autorender starts a render on launch; --screenshot out.png [--screenshot-delay s]
            // renders the UI to an image and exits.
            string[] argv = Environment.GetCommandLineArgs();
            if (argv.Contains("--autorender")) window.StartRenderFromCommandLine();
            int i = Array.IndexOf(argv, "--screenshot");
            int di = Array.IndexOf(argv, "--screenshot-delay");
            double delay = di >= 0 && di + 1 < argv.Length && double.TryParse(argv[di + 1], out double dv) ? dv : 2.5;
            if (i >= 0 && i + 1 < argv.Length) _ = window.SaveScreenshotAndExitAsync(argv[i + 1], delay);
        }
        catch (Exception ex)
        {
            Log($"startup failed: {ex}");
            throw;
        }
    }
}
