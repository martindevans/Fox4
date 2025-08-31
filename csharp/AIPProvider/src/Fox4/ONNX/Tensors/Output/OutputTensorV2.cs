using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Output;

public class OutputTensorV2(bool _trigger)
    : IOutputTensorReader
{
    /// <summary>
    /// The stick will return this factor towards zero every second
    /// </summary>
    public const float RECENTERING = 0.5f;

    public IReadOnlyList<string> Columns { get; } =
    [
        "trigger",

        "throttle",

        "pitch",
        "yaw",
        "roll",
    ];

    public InboundState Read(ReadOnlySpan<float> tensor, InboundState previousOutputs, AircraftState state)
    {
        var builder = new ActionsBuilder(state);

        // Gun trigger
        var trigger = tensor[0] > 0;
        if (_trigger && trigger)
            builder.TryFire(WeaponType.Guns);

        // Delta throttle control
        var throttle = Math.Clamp(previousOutputs.throttle + tensor[1] * 2 * state.DeltaTime, 0, 1);

        // Delta stick control with auto recentering
        var recenter = MathF.Pow(1 - RECENTERING, state.DeltaTime);
        var pitch    = Math.Clamp(previousOutputs.pyr.x * recenter + tensor[2] * 2 * state.DeltaTime, -1, 1);
        var yaw      = Math.Clamp(previousOutputs.pyr.y * recenter + tensor[3] * 1 * state.DeltaTime, -1, 1);
        var roll     = Math.Clamp(previousOutputs.pyr.z * recenter + tensor[4] * 2 * state.DeltaTime, -1, 1);

        return new InboundState
        {
            events = builder.Build(),
            pyr = new(pitch, yaw, roll),
            throttle = throttle
        };
    }
}