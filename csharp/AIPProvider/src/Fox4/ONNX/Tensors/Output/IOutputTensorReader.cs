using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Output;

public interface IOutputTensorReader
{
    IReadOnlyList<string> Columns { get; }

    InboundState Read(ReadOnlySpan<float> tensor, OutboundState state);
}

public static class OutputTensorReaderFactory
{
    public static IOutputTensorReader Get(string version)
    {
        return (version) switch
        {
            "v1" => new OutputTensorV1(),

            _ => throw new ArgumentException($"Unknown output tensor type: {version}")
        };
    }
}