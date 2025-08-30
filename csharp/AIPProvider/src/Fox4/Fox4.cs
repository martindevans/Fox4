using AIPProvider.Extensions;
using AIPProvider.Fox4.ONNX.Tensors;
using AIPProvider.Fox4.ONNX.Tensors.Input;
using AIPProvider.Fox4.ONNX.Tensors.Output;
using Microsoft.ML.OnnxRuntime;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4;

public sealed class Fox4
    : IDisposable
{
    private const string INPUT_TENSOR = "input";

    private readonly IAIPProvider _provider;

    private readonly RunOptions _runOptions;
    private readonly InferenceSession _session;
    private readonly IInputTensorBuilder _inputBuilder;
    private readonly IOutputTensorReader _outputReader;
    private readonly IReadOnlyList<string> _outputs;

    private readonly DatasetLogger? _logDataset;

    private readonly Random _random;
    private readonly float _outputRandStd;
    private readonly float[] _prevOutputsArr;

    public string Name { get; }

    public Fox4(IAIPProvider provider, int id, string modelPath, bool logDataset, float outputRandStd, string runid)
    {
        _provider = provider;

        _random = new Random(id * 3452346 + (int)DateTime.UtcNow.Ticks);
        _outputRandStd = outputRandStd;

        _runOptions = new RunOptions();

        modelPath = modelPath.Replace("{{PILOT_ID}}", id.ToString());

        _session = new InferenceSession(modelPath);
        _outputs = _session.OutputNames;

        Name = $"Fox4 v{_session.ModelMetadata.CustomMetadataMap.GetValueOrDefault("version", "unknown")}";

        _inputBuilder = InputTensorBuilderFactory.Get(_session.ModelMetadata.CustomMetadataMap.GetValueOrDefault("input_tensor_version") ?? "v1");
        _outputReader = OutputTensorReaderFactory.Get(_session.ModelMetadata.CustomMetadataMap.GetValueOrDefault("output_tensor_version") ?? "v1");

        var outputs = _outputReader.Columns.Count;
        _prevOutputsArr = new float[outputs];

        if (logDataset)
            _logDataset = DatasetLogger.Create($"{id}{runid}", _inputBuilder, _outputReader);
    }

    public void Dispose()
    {
        _runOptions.Dispose();
        _session.Dispose();
        _logDataset?.Dispose();
    }

    public InboundState Update(AircraftState state)
    {
        // Build inputs
        var inputTensor = _inputBuilder.Build(state, Map.instance);
        using var inputTensorOrt = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, inputTensor.Buffer, [ 1, inputTensor.Length ]);
        var inputs = new Dictionary<string, OrtValue>
        {
            { INPUT_TENSOR, inputTensorOrt }
        };

        // Inference forward pass
        using var outputs = _session.Run(_runOptions, inputs, _outputs);

        // Read outputs
        var outputsArr = ReadOutputs(outputs);

        // Log tensor data to CSV
        _logDataset?.Log(inputTensor.Buffer.Span, outputsArr);

        // Convert outputs to game actions
        return _outputReader.Read(outputsArr, state);
    }

    private float[] ReadOutputs(IDisposableReadOnlyCollection<OrtValue> outputs)
    {
        var outputsArr = outputs[0].GetTensorDataAsSpan<float>().ToArray();

        // Apply output randomisation
        if (_outputRandStd > 0)
        {
            // Try to find a deviation tensor
            var devTensorIdx = _outputs.IndexOf("output_deviation");

            // Apply noise
            if (devTensorIdx < 0)
                outputsArr.AddGaussianNoise(_random, _outputRandStd);
            else
                outputsArr.AddGaussianNoise(_random, _outputRandStd, outputs[devTensorIdx]);
        }

        // Copy outputs to prev outputs
        Array.Copy(outputsArr, _prevOutputsArr, outputsArr.Length);

        return outputsArr;
    }
}