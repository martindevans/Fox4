using CommandLine;
using System.Reflection;
using System.Runtime.InteropServices;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4;

public class Fox4Provider
    : IAIPProvider
{
    private Fox4? _pilot;

    private bool _setup;
    private bool _stopped;

    static Fox4Provider()
    {
        // Load onnxruntime.dll from a location relative to the currently executing assembly
        var assemblyPath = Assembly.GetExecutingAssembly().Location;
        var onnxRuntimePath = Path.Join(Path.GetDirectoryName(assemblyPath), @"runtimes\win-x64\native", "onnxruntime.dll");
        NativeLibrary.Load(onnxRuntimePath);
    }

    public override SetupActions Start(SetupInfo info)
    {
        // Parse arguments as if they were commandline args
        var opts = new Options();
        Parser.Default.ParseArguments<Options>(string.Join(" ", info.args).Split(' '))
              .WithParsed(o => opts = o)
              .WithNotParsed(errs =>
               {
                   Log("Failed to parse setup args:");
                   foreach (var error in errs)
                       Log(error.ToString() ?? "");
               });
        
        // Create inner pilot, responsible for running model
        _pilot = new Fox4(this, info.id, "model.onnx", opts.LogTensors, opts.OutputRandDev, opts.RunId);

        // Request a gun and no other equipment
        return new SetupActions
        {
            hardpoints = ["HPEquips/AFighter/fa26_gun"],
            name = _pilot.Name
        };
    }

    public override void Stop()
    {
        if (_stopped)
            return;
        _stopped = true;

        base.Stop();

        _pilot?.Dispose();
    }

    public override InboundState Update(OutboundState state)
    {
        // Don't do anything after stop is called
        if (_stopped)
            return default;

        // Skip the very first step of the sim to avoid weird issues on the first frame
        if (!_setup)
        {
            _setup = true;
            return new InboundState { throttle = 1 };
        }

        return _pilot!.Update(state);
    }
}

public class Options
{
    [Option("log-tensors")]
    public bool LogTensors { get; set; }

    [Option("output-rand-dev")]
    public float OutputRandDev { get; set; } = 0;

    [Option("runid")]
    public string RunId { get; set; } = "";
}