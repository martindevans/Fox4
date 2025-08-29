using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Output;

public interface IOutputTensorReader
{
    IReadOnlyList<string> Columns { get; }

    InboundState Read(ReadOnlySpan<float> tensor, GameState state);
}

public static class OutputTensorReaderFactory
{
    public static IOutputTensorReader Get(string version)
    {
        return (version) switch
        {
            "v1" => new OutputTensorV1(),
            "v2" => new OutputTensorV2(),

            _ => throw new ArgumentException($"Unknown output tensor type: {version}")
        };
    }
}