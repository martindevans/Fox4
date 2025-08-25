using AIPProvider.Extensions;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Input;

public class InputTensorV2
    : IInputTensorBuilder
{
    private const float PITCH_RATE_NORM = 360;  // 1 full rotation per second
    private const float YAW_RATE_NORM = 360;    // 1 full rotation per second
    private const float ROLL_RATE_NORM = 360;   // 1 full rotation per second
    private const float ALT_NORM = 15240;       // 50,000ft
    private const float SPEED_NORM = 686;       // Mach 2

    public IReadOnlyList<string> Columns { get; } =
    [
        "fwd.x", "fwd.y", "fwd.z",
        "right.x", "right.y", "right.z",
        "up.x", "up.y", "up.z",

        "pitch",
        "yaw",
        "roll",

        "pitch_rate",
        "yaw_rate",
        "roll_rate",

        "altitude",
        "radar_altitude",

        "speed",
        "dir.x", "dir.y", "dir.z",
    ];

    public DenseTensor<float> Build(ref OutboundState state, Vector3 angleRate, Map map)
    {
        var speed = state.kinematics.velocity.vec3.magnitude;
        var dir = state.kinematics.velocity.vec3 / Math.Max(0.001f, speed);
        var radar_altitude = state.kinematics.position.y - map.GetHeightAtSubpoint(state.kinematics.position);
        var euler_angles = state.kinematics.rotation.quat.eulerAngles;

        float[] data =
        [
            // Rotation frame of reference
            //..state.kinematics.rotation.EncodeFrameOfReference(),
            0,0,0,
            0,0,0,
            0,0,0,

            // Rotation
            euler_angles.x / 360,
            euler_angles.y / 360,
            euler_angles.z / 360,

            // Rotation rate
            angleRate.X / PITCH_RATE_NORM,
            angleRate.Y / YAW_RATE_NORM,
            angleRate.Z / ROLL_RATE_NORM,

            // Altitude
            state.kinematics.position.y / ALT_NORM,
            radar_altitude / ALT_NORM,

            // Speed & Direction (world space)
            speed / SPEED_NORM,
            dir.x, dir.y, dir.z,
        ];

        return new DenseTensor<float>(data, [1, data.Length]);
    }
}