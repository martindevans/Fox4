using Microsoft.ML.OnnxRuntime.Tensors;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Input;

public interface IInputTensorBuilder
{
    IReadOnlyList<string> Columns { get; }

    DenseTensor<float> Build(AircraftState state, InboundState previousOutputs, Map map, AircraftState enemy);
}

public static class InputTensorBuilderFactory
{
    public static IInputTensorBuilder Get(string version)
    {
        return (version) switch
        {
            "v1" => new InputTensorV1(),
            "v2" => new InputTensorV2(),
            "v3" => new InputTensorV3(),
            "v4" => new InputTensorV4(),

            _ => throw new ArgumentException($"Unknown input tensor type: {version}")
        };
    }
}