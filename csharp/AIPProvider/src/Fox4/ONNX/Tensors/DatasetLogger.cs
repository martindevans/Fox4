using System.Text;
using AIPProvider.Extensions;
using AIPProvider.Fox4.ONNX.Tensors.Input;
using AIPProvider.Fox4.ONNX.Tensors.Output;

namespace AIPProvider.Fox4.ONNX.Tensors;

public sealed class DatasetLogger
    : IDisposable
{
    private readonly StreamWriter _inputs;
    private readonly StreamWriter _outputs;

    private DatasetLogger(StreamWriter inputs, StreamWriter outputs)
    {
        _inputs = inputs;
        _outputs = outputs;
    }

    public static DatasetLogger Create(int id, IInputTensorBuilder inputs, IOutputTensorReader outputs)
    {
        var inputWriter = File.CreateText($"input_tensors-{id}.csv");
        var outputWriter = File.CreateText($"output_tensors-{id}.csv");

        inputWriter.WriteLine(string.Join(", ", inputs.Columns));
        outputWriter.WriteLine(string.Join(", ", outputs.Columns));

        return new DatasetLogger(inputWriter, outputWriter);
    }

    public void Log(ReadOnlySpan<float> inputTensor, ReadOnlySpan<float> outputTensor)
    {
        var builder = new StringBuilder();
        builder.AppendJoin(", ", inputTensor);
        _inputs.WriteLine(builder);

        builder.Clear();
        builder.AppendJoin(", ", outputTensor);
        _outputs.WriteLine(builder);

        _inputs.Flush();
        _outputs.Flush();
    }

    public void Dispose()
    {
        _inputs.Flush();
        _outputs.Flush();
    }
}