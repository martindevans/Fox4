using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Output;

public class OutputTensorV1
    : IOutputTensorReader
{
    public IReadOnlyList<string> Columns { get; } =
    [
        "trigger",

        "raw_throttle",
        "afterburner",

        "yaw",
        "pitch",
        "roll",
    ];

    public InboundState Read(ReadOnlySpan<float> tensor, OutboundState state)
    {
        var builder = new ActionsBuilder(state);

        var trigger = tensor[0] > 0;
        if (trigger)
            builder.TryFire(WeaponType.Guns);

        var rawThrottle = Math.Clamp(tensor[1] / 2 + 0.5f, 0, 1);
        var afterburner = tensor[2] > 0;

        var yaw = Math.Clamp(tensor[3], -1, 1);
        var pitch = Math.Clamp(tensor[4], -1, 1);
        var roll = Math.Clamp(tensor[5], -1, 1);

        return new InboundState
        {
            events = builder.Build(),
            pyr = new NetVector(pitch, yaw, roll),
            throttle = afterburner ? 1 : rawThrottle * 0.749f
        };
    }
}