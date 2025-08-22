using AIPProvider.Extensions;
using AIPProvider.Fox4.ONNX.Tensors;
using AIPProvider.Fox4.ONNX.Tensors.Input;
using AIPProvider.Fox4.ONNX.Tensors.Output;
using Microsoft.ML.OnnxRuntime;
using System.Numerics;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4;

public sealed class Fox4
    : IDisposable
{
    private const string INPUT_TENSOR = "input";
    private const string OUTPUT_TENSOR = "output";

    private readonly IAIPProvider _provider;

    private readonly RunOptions _runOptions;
    private readonly InferenceSession _session;
    private readonly IInputTensorBuilder _inputBuilder;
    private readonly IOutputTensorReader _outputReader;

    private float? _previousTime;
    private Vector3? _previousEulerAngles;

    private readonly DatasetLogger? _logDataset;

    public string Name { get; }

    public Fox4(IAIPProvider provider, int id, string modelPath, bool logDataset = false)
    {
        _provider = provider;

        _runOptions = new RunOptions();
        _session = new InferenceSession(modelPath);

        Name = $"Fox4 v{_session.ModelMetadata.Version}";

        _inputBuilder = InputTensorBuilderFactory.Get(_session.ModelMetadata.CustomMetadataMap.GetValueOrDefault("input_tensor_version") ?? "v1");
        _outputReader = OutputTensorReaderFactory.Get(_session.ModelMetadata.CustomMetadataMap.GetValueOrDefault("output_tensor_version") ?? "v1");

        if (logDataset)
            _logDataset = DatasetLogger.Create(id, _inputBuilder, _outputReader);
    }

    public void Dispose()
    {
        _runOptions.Dispose();
        _session.Dispose();
        _logDataset?.Dispose();
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
            { INPUT_TENSOR, inputTensorOrt }
        };

        // Inference forward pass
        using var outputs = _session.Run(_runOptions, inputs, [OUTPUT_TENSOR]);
        var outputsSpan = outputs[0].GetTensorDataAsSpan<float>();

        // Log tensor data to CSV
        _logDataset?.Log(inputTensor.Buffer.Span, outputsSpan);

        // Read outputs
        return _outputReader.Read(outputsSpan, state);
    }
}