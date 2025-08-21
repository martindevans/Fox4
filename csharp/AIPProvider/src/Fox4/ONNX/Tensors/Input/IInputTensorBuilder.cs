using System.Numerics;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Input;

public interface IInputTensorBuilder
{
    IReadOnlyList<string> Columns { get; }

    DenseTensor<float> Build(ref OutboundState state, Vector3 angleRate, Map map);
}

public static class InputTensorBuilderFactory
{
    public static IInputTensorBuilder Get(string version)
    {
        return (version) switch
        {
            "v1" => new InputTensorV1(),
            "v2" => new InputTensorV2(),

            _ => throw new ArgumentException($"Unknown input tensor type: {version}")
        };
    }
}