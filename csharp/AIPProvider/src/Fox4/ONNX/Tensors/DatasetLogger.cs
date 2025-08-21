using System.Text;
using AIPProvider.Extensions;

namespace AIPProvider.Fox4.ONNX.Tensors;

public sealed class DatasetLogger
    : IDisposable
{
    private readonly StreamWriter _inputs;
    private readonly StreamWriter _outputs;

    public DatasetLogger()
    {
        _inputs = File.CreateText("input_tensors.csv");
        _outputs = File.CreateText("output_tensors.csv");
    }

    public void Log(ReadOnlySpan<float> inputTensor, ReadOnlySpan<float> outputTensor)
    {
        var builder = new StringBuilder();
        builder.AppendJoin(", ", inputTensor);
        _inputs.WriteLine(builder);

        builder.Clear();
        builder.AppendJoin(", ", outputTensor);
        _outputs.WriteLine(builder);
    }

    public void Dispose()
    {
        _inputs.Dispose();
        _outputs.Dispose();
    }
}