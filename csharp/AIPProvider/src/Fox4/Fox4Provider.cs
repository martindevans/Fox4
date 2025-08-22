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

    public override SetupActions Start(SetupInfo info)
    {
        // Locate the assembly that is cirrently executing
        var assemblyPath = Assembly.GetExecutingAssembly().Location;
        Log($"Assembly Path: {assemblyPath}");

        // Load onnxruntime.dll from a known relative path
        var onnxRuntimePath = Path.Join(Path.GetDirectoryName(assemblyPath), @"runtimes\win-x64\native", "onnxruntime.dll");
        Log($"Loading onnxruntime: {onnxRuntimePath}");
        NativeLibrary.Load(onnxRuntimePath);

        // Create inner pilot, responsible for running model
        _pilot = new Fox4(this, info.id, "model.onnx", info.args.Contains("--log-tensors"));

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

        // Skip the very first step of the sim to avoid weird setup issues on the first frame
        if (!_setup)
        {
            _setup = true;
            return new InboundState { throttle = 1 };
        }

        return _pilot!.Update(state);
    }
}