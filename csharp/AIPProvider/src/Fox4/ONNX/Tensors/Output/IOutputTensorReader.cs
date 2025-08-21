using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Output;

public interface IOutputTensorReader
{
    InboundState Read(ReadOnlySpan<float> tensor, OutboundState state);
}

public static class OutputTensorReaderFactory
{
    public static IOutputTensorReader Get(string version)
    {
        return (version) switch
        {
            "v1" => new OutputTensorV1(),

            _ => throw new ArgumentException($"Unknown input tensor type: {version}")
        };
    }
}