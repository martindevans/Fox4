using AIPProvider.Extensions;
using AIPProvider.Fox4.ONNX.Tensors.Input;
using AIPProvider.Fox4.ONNX.Tensors.Output;
using Microsoft.ML.OnnxRuntime;
using System.Numerics;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4;

public sealed class Fox4
    : IDisposable
{
    private readonly RunOptions _options;
    private readonly InferenceSession _session;

    private readonly string _inputName;
    private readonly IInputTensorBuilder _inputBuilder;
    private readonly string[] _outputNames;
    private readonly IOutputTensorReader _outputReader;

    private float? _previousTime;
    private Vector3? _previousEulerAngles;

    public string Name { get; init; }

    public Fox4(string modelPath)
    {
        _session = new InferenceSession(modelPath);
        _options = new RunOptions();

        _inputName = _session.InputMetadata.Keys.First();
        _outputNames = _session.OutputNames.ToArray();

        Name = $"Fox4 v{_session.ModelMetadata.Version}";

        _inputBuilder = InputTensorBuilderFactory.Get(_session.ModelMetadata.CustomMetadataMap.GetValueOrDefault("input_tensor_version") ?? "v1");
        _outputReader = OutputTensorReaderFactory.Get(_session.ModelMetadata.CustomMetadataMap.GetValueOrDefault("output_tensor_version") ?? "v1");
    }

    public void Dispose()
    {
        _session.Dispose();
        _options.Dispose();
    }

    public InboundState Update(OutboundState state)
    {
        // Calculate delta time
        if (!_previousTime.HasValue)
            _previousTime = state.time;
        var dt = state.time - _previousTime.Value;
        _previousTime = state.time;

        // Calculate change in euler angle since last frame
        var euler = state.kinematics.rotation.quat.eulerAngles.To();
        var angleRate = Vector3.Zero;
        if (!_previousEulerAngles.HasValue || dt == 0)
        {
            _previousEulerAngles = euler;
        }
        else
        {
            var deltaAngle = _previousEulerAngles.Value - euler;
            angleRate = deltaAngle / dt;
        }

        // Build inputs
        var inputTensor = _inputBuilder.Build(ref state, angleRate, Map.instance);
        using var inputTensorOrt = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, inputTensor.Buffer, [ 1, inputTensor.Length ]);
        var inputs = new Dictionary<string, OrtValue>
        {
            { _inputName, inputTensorOrt }
        };

        // Inference forward pass
        //using var outputs = _session.Run(_options, inputs, _outputNames);

        // HACK!! This works around the issue
        using var session = new InferenceSession("model.onnx");
        using var outputs = session.Run(_options, inputs, _outputNames);

        // Read outputs
        return _outputReader.Read(outputs[0].GetTensorDataAsSpan<float>(), state);
    }
}