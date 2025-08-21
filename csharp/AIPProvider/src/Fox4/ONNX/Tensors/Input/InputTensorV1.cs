using AIPProvider.Extensions;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Input;

public class InputTensorV1
    : IInputTensorBuilder
{
    private const float FUEL_NORM = 10000;      // Max fuel
    private const float BULLET_NORM = 800;      // Max bullets
    private const float PITCH_RATE_NORM = 360;  // 1 full rotation per second
    private const float YAW_RATE_NORM = 360;    // 1 full rotation per second
    private const float ROLL_RATE_NORM = 360;   // 1 full rotation per second
    private const float ALT_NORM = 15240;       // 50,000ft
    private const float SPEED_NORM = 686;       // Mach 2
    private const float ENERGY_NORM = 100_000;  // Arbitrary choice

    public IReadOnlyList<string> Columns { get; } =
    [
        "fuel",
        "gun_ammo",
        "gun_ammo_available",

        "fwd.x", "fwd.y", "fwd.z",
        "right.x", "right.y", "right.z",
        "up.x", "up.y", "up.z",

        "pitch_rate",
        "yaw_rate",
        "roll_rate",

        "radar_altitude",
        "altitude",

        "speed",
        "dir.x", "dir.y", "dir.z",

        "localDir.x", "localDir.y", "localDir.z",
        "target.closure",

        "tgt_fwd.x", "tgt_fwd.y", "tgt_fwd.z",
        "tgt_right.x", "tgt_right.y", "tgt_right.z",
        "tgt_up.x", "tgt_up.y", "tgt_up.z",

        "kinetic_energy",
        "potential_energy",
    ];

    public DenseTensor<float> Build(ref OutboundState state, Vector3 angleRate, Map map)
    {
        var radar_altitude = state.kinematics.position.y - map.GetHeightAtSubpoint(state.kinematics.position);

        var speed = state.kinematics.velocity.vec3.magnitude;
        var dir = state.kinematics.velocity.vec3 / speed;

        var target = state.visualTargets.Length == 0 ? default : state.visualTargets[0];
        var localDir = state.kinematics.rotation.quat * target.direction;

        var targetLocalRot = Quaternion.Inverse(state.kinematics.rotation.To()) * target.orientation.To();

        var mass = 1;
        var g = 9.8f;
        var ke_energy = 0.5f * mass * MathF.Pow(state.kinematics.velocity.vec3.magnitude, 2);
        var pe_energy = mass * g * state.kinematics.position.y;

        float[] data =
        [
            // Fuel
            state.fuel / FUEL_NORM,

            // Ammo
            state.gunAmmo / BULLET_NORM,
            state.gunAmmo == 0 ? -1 : 1,

            // Rotation
            ..state.kinematics.rotation.EncodeFrameOfReference(),

            // Rotation rate
            angleRate.X / PITCH_RATE_NORM,
            angleRate.Y / YAW_RATE_NORM,
            angleRate.Z / ROLL_RATE_NORM,

            // Altitude
            radar_altitude / ALT_NORM,
            state.kinematics.position.y / ALT_NORM,

            // Speed & Direction (world space)
            speed / SPEED_NORM,
            dir.x, dir.y, dir.z,

            // Target info
            localDir.x, localDir.y, localDir.z,
            target.closure / SPEED_NORM,
            ..targetLocalRot.EncodeFrameOfReference(),

            // Energy state
            ke_energy / ENERGY_NORM,
            pe_energy / ENERGY_NORM,
        ];

        return new DenseTensor<float>(data, [1, data.Length]);
    }
}