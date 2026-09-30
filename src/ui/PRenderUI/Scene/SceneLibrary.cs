namespace PRenderUI.Scene;

/// <summary>A scene the picker can offer: a display label and the file to load.</summary>
public sealed record SceneEntry(string Label, string Path);

/// <summary>
/// Finds scenes in a scenes folder: every scene file directly inside each of its subfolders (one
/// scene per folder is the convention, e.g. scenes/sphere-pyramid/scene.prscene.json) and any
/// scene files at its top level. Rescanning is cheap, so the picker rescans whenever it opens.
/// </summary>
public static class SceneLibrary
{
    private static readonly string[] Patterns = { "*.prscene.json", "*.gltf", "*.glb", "*.pbrt" };

    public static List<SceneEntry> Find(string root)
    {
        var result = new List<SceneEntry>();
        if (!Directory.Exists(root)) return result;
        try
        {
            foreach (var dir in Directory.GetDirectories(root).OrderBy(d => d, StringComparer.OrdinalIgnoreCase))
            {
                var files = SceneFiles(dir);
                string folder = System.IO.Path.GetFileName(dir);
                foreach (var f in files)
                    result.Add(new SceneEntry(files.Count == 1 ? folder : $"{folder} / {Stem(f)}", f));
            }
            foreach (var f in SceneFiles(root)) result.Add(new SceneEntry(Stem(f), f));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // An unreadable folder simply contributes no scenes.
        }
        return result;
    }

    /// <summary>The folder whose subfolders hold scenes: the folder above the scene's own folder.</summary>
    public static string? RootFor(string scenePath)
    {
        string? dir = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(scenePath));
        return dir is null ? null : Directory.GetParent(dir)?.FullName;
    }

    private static List<string> SceneFiles(string dir)
    {
        var files = new List<string>();
        foreach (var p in Patterns)
        {
            try { files.AddRange(Directory.GetFiles(dir, p)); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }
        return files.Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(f => f, StringComparer.OrdinalIgnoreCase).ToList();
    }

    private static string Stem(string file)
    {
        string name = System.IO.Path.GetFileName(file);
        return name.EndsWith(".prscene.json", StringComparison.OrdinalIgnoreCase) ? name[..^".prscene.json".Length]
                                                                                   : System.IO.Path.GetFileNameWithoutExtension(file);
    }
}
