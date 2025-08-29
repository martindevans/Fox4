using UnityGERunner.UnityApplication;
using Vector3 = System.Numerics.Vector3;

namespace AIPProvider.Extensions;

public static class OutboundStateExtensions
{
    public static (Vector3 forward, Vector3 right, Vector3 up) GetFrameOfReferences(this OutboundState state)
    {
        var forward = Vector3.Transform(new Vector3(0f, 0f, 1f), state.kinematics.rotation.To());
        var right = Vector3.Transform(new Vector3(1f, 0f, 0f), state.kinematics.rotation.To());
        var up = Vector3.Transform(new Vector3(0f, 1f, 0f), state.kinematics.rotation.To());

        return (forward, right, up);
    }

    /// <summary>
    /// Pitch in radians
    /// </summary>
    /// <param name="state"></param>
    /// <returns></returns>
    public static float GetPitch(this OutboundState state)
    {
        var (forward, _, _) = GetFrameOfReferences(state);
        return MathF.Atan2(-forward.Y, MathF.Sqrt(forward.X * forward.X + forward.Z * forward.Z));
    }

    /// <summary>
    /// Roll in radians
    /// </summary>
    /// <param name="state"></param>
    /// <returns></returns>
    public static float GetRoll(this OutboundState state)
    {
        var (_, right, up) = GetFrameOfReferences(state);
        return MathF.Atan2(right.Y, up.Y);
    }

    /// <summary>
    /// Yaw in radians
    /// </summary>
    /// <param name="state"></param>
    /// <returns></returns>
    public static float GetYaw(this OutboundState state)
    {
        var (forward, _, _) = GetFrameOfReferences(state);
        return MathF.Atan2(forward.X, forward.Z);
    }
}