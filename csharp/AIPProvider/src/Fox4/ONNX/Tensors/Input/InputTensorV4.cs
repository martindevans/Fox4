using System.Numerics;
using AIPProvider.Extensions;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4.ONNX.Tensors.Input;

public class InputTensorV4
    : IInputTensorBuilder
{
    private const float ANGLE_RATE_NORM = MathF.Tau;    // Full rotation per second
    private const float ALT_NORM = 15240;               // 50,000ft
    private const float DIST_NORM = 10000;              // 10km
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

        "enemy_dist",
        "local_enemy_dir.x", "local_enemy_dir.y", "local_enemy_dir.z",
        "enemy_speed",
        "local_enemy_vel.x", "local_enemy_vel.y", "local_enemy_vel.z",
    ];

    public DenseTensor<float> Build(AircraftState state, InboundState previousOutputs, Map map, AircraftState enemy)
    {
        var speed = state.Speed;
        var localDir = state.LocalVelocity / speed;
        var localGrav = Vector3.Transform(new Vector3(0, -1f, 0), Quaternion.Inverse(state.Orientation));

        // Get dist and direction to enemy
        var enemyDir = enemy.WorldPosition - state.WorldPosition;
        var enemyDist = enemyDir.Length();
        enemyDir /= enemyDist;
        enemyDir = Vector3.Transform(enemyDir, Quaternion.Inverse(state.Orientation));

        var enemyVel = Vector3.Transform(enemy.WorldVelocity, Quaternion.Inverse(state.Orientation));
        var enemySpeed = enemyVel.Length();
        enemyVel /= enemySpeed;

        float[] data =
        [
            state.DeltaTime,

            // Altitude
            state.WorldPosition.Y / ALT_NORM,
            map.GetRadarAltitude(state.WorldPosition) / ALT_NORM,

            // Velocity
            speed / SPEED_NORM,
            localDir.X, localDir.Y, localDir.Z,
            state.VerticalSpeed / SPEED_NORM,
            ((0.5f * speed * speed) + (9.8f * state.WorldPosition.Y)) / ENERGY_NORM,

            // Attitude
            MathF.Sin(state.AoA), MathF.Cos(state.AoA),
            MathF.Sin(state.Slip), MathF.Cos(state.Slip),
            MathF.Sin(state.Pitch), MathF.Cos(state.Pitch),
            MathF.Sin(state.Yaw), MathF.Cos(state.Yaw),
            MathF.Sin(state.Roll), MathF.Cos(state.Roll),
            localGrav.X, localGrav.Y, localGrav.Z,

            // Attitude rate
            state.DeltaPitch / ANGLE_RATE_NORM,
            state.DeltaYaw / ANGLE_RATE_NORM,
            state.DeltaRoll / ANGLE_RATE_NORM,

            // Previous inputs
            previousOutputs.pyr.x,
            previousOutputs.pyr.y,
            previousOutputs.pyr.z,
            previousOutputs.throttle,

            // Magic sensor
            enemyDist / DIST_NORM,
            enemyDir.X, enemyDir.Y, enemyDir.Z,
            enemySpeed / SPEED_NORM,
            enemyVel.X, enemyVel.Y, enemyVel.Z,
        ];

        return new DenseTensor<float>(data, [1, data.Length]);
    }
}