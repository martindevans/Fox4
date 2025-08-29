using AIPProvider.Extensions;
using UnityGERunner;
using UnityGERunner.UnityApplication;
using Quaternion = System.Numerics.Quaternion;
using Vector3 = System.Numerics.Vector3;

namespace AIPProvider.Fox4;

public class GameState
{
    private OutboundState _prevState;
    private OutboundState _state;

    public InboundState PreviousOutputs { get; private set; }

    public OutboundState RawGameState => _state;

    public float Time => _state.time;
    public float DeltaTime => _state.time - _prevState.time;

    public float Fuel => _state.fuel;
    public float DeltaFuel => (_state.fuel - _prevState.fuel) / DeltaTime;

    public Quaternion Rotation => _state.kinematics.rotation.To();
    public Vector3 WorldPosition => _state.kinematics.position.To();
    public Vector3 WorldVelocity => _state.kinematics.velocity.To();
    public float Speed => WorldVelocity.Length();
    public Vector3 LocalVelocity => Vector3.Transform(WorldVelocity, Quaternion.Inverse(Rotation));

    public float VerticalSpeed => (_state.kinematics.position.y - _prevState.kinematics.position.y) / DeltaTime;

    public int GunAmmo => _state.gunAmmo;
    public int ChaffCount => _state.chaffCount;
    public int FlareCount => _state.flareCount;

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

    public float Pitch => _state.GetPitch();
    public float DeltaPitch => MathHelper.DeltaAngleRadians(_state.GetPitch(), _prevState.GetPitch()) / DeltaTime;

    public float Roll => _state.GetRoll();
    public float DeltaRoll => MathHelper.DeltaAngleRadians(_state.GetRoll(), _prevState.GetRoll()) / DeltaTime;

    public float Yaw => _state.GetYaw();
    public float DeltaYaw => MathHelper.DeltaAngleRadians(_state.GetYaw(), _prevState.GetYaw()) / DeltaTime;

    public void Update(OutboundState state)
    {
        _prevState = _state;
        _state = state;
    }

    public InboundState SetOutputs(InboundState outputs)
    {
        PreviousOutputs = outputs;
        return outputs;
    }
}