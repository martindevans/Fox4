using AIPProvider.Extensions;
using UnityGERunner;
using UnityGERunner.UnityApplication;
using Quaternion = System.Numerics.Quaternion;
using Vector3 = System.Numerics.Vector3;

namespace AIPProvider.Fox4;

public record AircraftState
{
    public readonly OutboundState PreviousRawState;
    public readonly OutboundState RawState;

    public AircraftState(OutboundState prev, OutboundState state)
    {
        PreviousRawState = prev;
        RawState = state;
    }

    public float Time => RawState.time;
    public float DeltaTime => RawState.time - PreviousRawState.time;

    public float Fuel => RawState.fuel;
    public float DeltaFuel => (RawState.fuel - PreviousRawState.fuel) / DeltaTime;

    public Quaternion Orientation => RawState.kinematics.rotation.To();

    public Vector3 WorldPosition => RawState.kinematics.position.To();
    public Vector3 WorldVelocity => RawState.kinematics.velocity.To();
    public float Speed => WorldVelocity.Length();
    public Vector3 LocalVelocity => Vector3.Transform(WorldVelocity, Quaternion.Inverse(Orientation));

    public float VerticalSpeed => (RawState.kinematics.position.y - PreviousRawState.kinematics.position.y) / DeltaTime;

    public int GunAmmo => RawState.gunAmmo;
    public int ChaffCount => RawState.chaffCount;
    public int FlareCount => RawState.flareCount;

    public float AoA
    {
        get
        {
            var speed = Speed;
            var localDir = LocalVelocity / speed;
            return speed < 0.1f ? 0 : Mathf.Atan2(-localDir.Y, localDir.Z);
        }
    }

    public float Slip
    {
        get
        {
            var speed = Speed;
            var localDir = LocalVelocity / speed;
            return speed < 0.1f ? 0 : Mathf.Atan2(-localDir.X, localDir.Z);
        }
    }

    public float Pitch => RawState.GetPitch();
    public float DeltaPitch => MathHelper.DeltaAngleRadians(RawState.GetPitch(), PreviousRawState.GetPitch()) / DeltaTime;

    public float Roll => RawState.GetRoll();
    public float DeltaRoll => MathHelper.DeltaAngleRadians(RawState.GetRoll(), PreviousRawState.GetRoll()) / DeltaTime;

    public float Yaw => RawState.GetYaw();
    public float DeltaYaw => MathHelper.DeltaAngleRadians(RawState.GetYaw(), PreviousRawState.GetYaw()) / DeltaTime;
}