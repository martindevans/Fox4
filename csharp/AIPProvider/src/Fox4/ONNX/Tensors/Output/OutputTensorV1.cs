using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Output;

public class OutputTensorV1
    : IOutputTensorReader
{
    public const float RECENTERING = 0.5f;

    public IReadOnlyList<string> Columns { get; } =
    [
        "trigger",

        "raw_throttle",
        "afterburner",

        "pitch",
        "yaw",
        "roll",
    ];

    public InboundState Read(ReadOnlySpan<float> tensor, GameState state)
    {
        var builder = new ActionsBuilder(state.RawGameState);

        var trigger = tensor[0] > 0;
        if (trigger)
            builder.TryFire(WeaponType.Guns);

        var rawThrottle = Math.Clamp(tensor[1], -1, 1) / 2 + 0.5f;
        var afterburner = tensor[2] > 0;

        var recenter = MathF.Pow(1 - RECENTERING, state.DeltaTime);
        var pitch = Math.Clamp(state.PreviousOutputs.pyr.x * recenter + tensor[3] * 2 * state.DeltaTime, -1, 1);
        var yaw = Math.Clamp(state.PreviousOutputs.pyr.y * recenter + tensor[4] * state.DeltaTime, -1, 1);
        var roll = Math.Clamp(state.PreviousOutputs.pyr.z * recenter + tensor[5] * 2 * state.DeltaTime, -1, 1);

        return new InboundState
        {
            events = builder.Build(),
            pyr = new(pitch, yaw, roll),
            throttle = afterburner ? 1 : rawThrottle * 0.749f
        };
    }
}