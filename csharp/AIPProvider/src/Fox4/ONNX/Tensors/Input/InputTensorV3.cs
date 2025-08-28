using System.Numerics;
using AIPProvider.Extensions;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Input;

public class InputTensorV3
    : IInputTensorBuilder
{
    private const float ANGLE_RATE_NORM = MathF.Tau;    // Full rotation per second
    private const float ALT_NORM = 15240;               // 50,000ft
    private const float SPEED_NORM = 686;               // Mach 2
    private const float ENERGY_NORM = 250000;           // Energy at (roughly) mach 1.5 at 50,000ft

    public IReadOnlyList<string> Columns { get; } =
    [
        "delta_time",

        "altitude",
        "radar_altitude",

        "speed",
        "local_dir.x", "local_dir.y", "local_dir.z",
        "vert_speed",
        "energy",

        "sin(aoa)", "cos(aoa)",
        "sin(slip)", "cos(slip)",
        "sin(pitch)", "cos(pitch)",
        "sin(yaw)", "cos(yaw)",
        "sin(roll)", "cos(roll)",
        "local_grav.x", "local_grav.y", "local_grav.z",

        "pitch_rate",
        "yaw_rate",
        "roll_rate",

        "prev_pitch_in",
        "prev_yaw_in",
        "prev_roll_in",
        "prev_throttle_in",
    ];

    public DenseTensor<float> Build(GameState state, Map map)
    {
        var speed = state.Speed;
        var localDir = state.LocalVelocity / speed;

        var localGrav = Vector3.Transform(new Vector3(0, -1f, 0), Quaternion.Inverse(state.Rotation));

        var radarAltitude = map.GetRadarAltitude(state.WorldPosition);

        float energy = (0.5f * speed * speed) + (9.8f * state.WorldPosition.Y);
        energy /= ENERGY_NORM;

        var aoa = state.AoA;
        var slip = state.Slip;
        var pitch = state.Pitch;
        var yaw = state.Yaw;
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
            energy,

            // Attitude
            MathF.Sin(aoa), MathF.Cos(aoa),
            MathF.Sin(slip), MathF.Cos(slip),
            MathF.Sin(pitch), MathF.Cos(pitch),
            MathF.Sin(yaw), MathF.Cos(yaw),
            MathF.Sin(roll), MathF.Cos(roll),
            localGrav.X, localGrav.Y, localGrav.Z,

            // Attitude rate
            state.DeltaPitch / ANGLE_RATE_NORM,
            state.DeltaYaw / ANGLE_RATE_NORM,
            state.DeltaRoll / ANGLE_RATE_NORM,

            // Previous inputs
            state.PreviousOutputs.pyr.x,
            state.PreviousOutputs.pyr.y,
            state.PreviousOutputs.pyr.z,
            state.PreviousOutputs.throttle,
        ];

        return new DenseTensor<float>(data, [1, data.Length]);
    }
}