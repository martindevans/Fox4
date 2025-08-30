using CommandLine;
using System.Reflection;
using System.Runtime.InteropServices;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4;

public class Fox4Provider
    : IAIPProvider
{
    private static readonly List<Fox4Provider> _providers = [ ];
    private static float _lastUpdatedMagic;
    private AircraftState _magicSensorState;

    private Fox4? _pilot;

    private bool _setup;
    private bool _stopped;

    private InboundState _lastOutputs;
    private OutboundState _prevState;
    private OutboundState _state;

    static Fox4Provider()
    {
        // Load onnxruntime.dll from a location relative to the currently executing assembly
        var assemblyPath = Assembly.GetExecutingAssembly().Location;
        var onnxRuntimePath = Path.Join(Path.GetDirectoryName(assemblyPath), @"runtimes\win-x64\native", "onnxruntime.dll");
        NativeLibrary.Load(onnxRuntimePath);
    }

    public override SetupActions Start(SetupInfo info)
    {
        // Store this provider in the list of all providers. This allows planes to find each other "by magic"
        _providers.Add(this);

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

        // Whichever aircraft updates first sets up all of the sensor data for all aircraft. This means that _none_ of
        // the aircraft have update yet, so we know everyone is getting consistent sensor data.
        // ReSharper disable once CompareOfFloatsByEqualityOperator
        if (_lastUpdatedMagic != state.time)
        {
            _lastUpdatedMagic = state.time;

            foreach (var provider in _providers)
                provider.UpdateMagicSensor();
        }

        // Update gamestate every frame with current state. Doing this even before setup
        // is important, it means that when the pilot is called we have seen 2 states and
        // can calculate deltas
        _prevState = _state;
        _state = state;

        // Skip the very first step of the sim to avoid weird issues on the first frame
        if (!_setup)
        {
            _setup = true;

            _lastOutputs = new InboundState { throttle = 1 };
            return _lastOutputs;
        }

        // Call the actual pilot impl
        var aircraftState = new AircraftState(
            prev: _prevState,
            state: _state
        );
        _lastOutputs = _pilot!.Update(aircraftState, _lastOutputs, _magicSensorState);
        return _lastOutputs;
    }

    private void UpdateMagicSensor()
    {
        var other = _providers.Single(a => !ReferenceEquals(a, this));
        _magicSensorState = new AircraftState(other._prevState, other._state);
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