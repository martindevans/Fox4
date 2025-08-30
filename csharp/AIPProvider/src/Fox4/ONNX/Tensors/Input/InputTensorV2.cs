using AIPProvider.Extensions;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Input;

public class InputTensorV2
    : IInputTensorBuilder
{
    private const float ANGLE_RATE_NORM = MathF.Tau;    // Full rotation per second
    private const float ALT_NORM = 15240;               // 50,000ft
    private const float SPEED_NORM = 686;               // Mach 2

    public IReadOnlyList<string> Columns { get; } =
    [
        "delta_time",

        "altitude",
        "radar_altitude",

        "speed",
        "local_dir.x", "local_dir.y", "local_dir.z",
        "vert_speed",

        "sin(aoa)", "cos(aoa)",
        "sin(slip)", "cos(slip)",
        "sin(pitch)", "cos(pitch)",
        "sin(roll)", "cos(roll)",

        "pitch_rate",
        "roll_rate",

        "prev_pitch_in",
        "prev_yaw_in",
        "prev_roll_in",
    ];

    public DenseTensor<float> Build(AircraftState state, Map map)
    {
        var speed = state.Speed;
        var localDir = state.LocalVelocity / speed;

        var radarAltitude = map.GetRadarAltitude(state.WorldPosition);

        var aoa = state.AoA;
        var slip = state.Slip;
        var pitch = state.Pitch;
        var roll = state.Roll;

        float[] data =
        [
            state.DeltaTime,

            // Altitude
            state.WorldPosition.Y / ALT_NORM,
            radarAltitude / ALT_NORM,

            // Velocity
            speed / SPEED_NORM,
            localDir.X, localDir.Y, localDir.Z,
            state.VerticalSpeed / SPEED_NORM,

            // Attitude
            MathF.Sin(aoa), MathF.Cos(aoa),
            MathF.Sin(slip), MathF.Cos(slip),
            MathF.Sin(pitch), MathF.Cos(pitch),
            MathF.Sin(roll), MathF.Cos(roll),

            // Attitude rate
            state.DeltaPitch / ANGLE_RATE_NORM,
            state.DeltaRoll / ANGLE_RATE_NORM,

            // Previous inputs
            state.PreviousOutputs.pyr.x,
            state.PreviousOutputs.pyr.y,
            state.PreviousOutputs.pyr.z,
        ];

        return new DenseTensor<float>(data, [1, data.Length]);
    }
}