using Microsoft.ML.OnnxRuntime;

namespace AIPProvider.Fox4;

public readonly ref struct OutputTensorReader
{
    private readonly ReadOnlySpan<float> _data;

    public OutputTensorReader(OrtValue ort)
    {
        _data = ort.GetTensorDataAsSpan<float>();
    }

    public OutputTensorReader(ReadOnlySpan<float> data)
    {
        _data = data;
    }

    /// <summary>
    /// Trigger, any value > 0 is true
    /// </summary>
    public bool Trigger => _data[0] > 0;

    /// <summary>
    /// Raw throttle value, [0, 1] range.
    /// </summary>
    /// <remarks>Remapped from [-1, 1] range</remarks>
    public float RawThrottle => Math.Clamp(_data[1] / 2 + 0.5f, 0, 1);

    /// <summary>
    /// Indicates if afterburner should be engaged, any value > 0 is true
    /// </summary>
    public bool Afterburner => _data[2] > 0;

    /// <summary>
    /// True throttle value to return to sim, combining <see cref="RawThrottle"/> and <see cref="Afterburner"/>
    /// </summary>
    public float Throttle => Afterburner ? 1 : RawThrottle * 0.749f;

    /// <summary>
    /// Yaw, [-1, 1] range
    /// </summary>
    public float Yaw => Math.Clamp(_data[3], -1, 1);

    /// <summary>
    /// Pitch, [-1, 1] range
    /// </summary>
    public float Pitch => Math.Clamp(_data[4], -1, 1);

    /// <summary>
    /// Roll, [-1, 1] range
    /// </summary>
    public float Roll => Math.Clamp(_data[5], -1, 1);
}