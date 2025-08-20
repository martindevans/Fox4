using System.Numerics;
using AIPProvider.Extensions;
using Microsoft.ML.OnnxRuntime;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4;

public sealed class Fox4
    : IDisposable
{
    private const string MODEL_PATH = "model.onnx";

    private const float FUEL_NORM = 10000;      // Max fuel
    private const float BULLET_NORM = 800;      // Max bullets
    private const float PITCH_RATE_NORM = 360;  // 1 full rotation per second
    private const float YAW_RATE_NORM = 360;    // 1 full rotation per second
    private const float ROLL_RATE_NORM = 360;   // 1 full rotation per second
    private const float ALT_NORM = 15240;       // 50,000ft
    private const float SPEED_NORM = 686;       // Mach 2

    private readonly IAIPProvider _logger;
    private readonly InferenceSession _session;
    private readonly RunOptions _options;

    private float? _previousTime;
    private Vector3? _previousEulerAngles;

    public string Name { get; init; }

    public Fox4(IAIPProvider logger)
    {
        _logger = logger;
        _session = new InferenceSession(MODEL_PATH);
        _options = new RunOptions();

        Name = $"Fox4 v{_session.ModelMetadata}";
    }

    public void Dispose()
    {
        _session.Dispose();
        _options.Dispose();
    }

    public InboundState Update(OutboundState state)
    {
        // Calculate delta time
        if (!_previousTime.HasValue)
            _previousTime = state.time;
        var dt = state.time - _previousTime.Value;
        _previousTime = state.time;

        // Calculate change in euler angle since last frame
        var euler = state.kinematics.rotation.quat.eulerAngles.To();
        if (!_previousEulerAngles.HasValue)
            _previousEulerAngles = euler;
        var deltaAngle = _previousEulerAngles.Value - euler;
        var angleRate = deltaAngle / dt;

        // Create input tensor
        var inputTensor = BuildInputTensor(ref state, angleRate);
        var inputs = new Dictionary<string, OrtValue>
        {
            { "input", inputTensor }
        };

        // Inference forward pass
        using var outputs = _session.Run(_options, inputs, _session.OutputNames);

        // Read outputs
        using var output = outputs[0];
        var reader = new OutputTensorReader(output);

        // Pull trigger
        var builder = new ActionsBuilder(state);
        if (reader.Trigger && builder.TryFire(WeaponType.Guns))
            _logger.Log($"Guns guns guns: {state.gunAmmo}");

        return new InboundState
        {
            throttle = reader.Throttle,
            pyr = new NetVector(reader.Pitch, reader.Yaw, reader.Roll),
            events = builder.Build(),
        };
    }

    private OrtValue BuildInputTensor(ref OutboundState state, Vector3 angleRate)
    {
        var radar_altitude = state.kinematics.position.y - _logger.HeightAt(state.kinematics.position);

        var speed = state.kinematics.velocity.vec3.magnitude;
        var dir = state.kinematics.velocity.vec3 / speed;

        var target = state.visualTargets[0];
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
            ke_energy,
            pe_energy,

            //todo: aiming/lead indicator info
        ];

        return OrtValue.CreateTensorValueFromMemory<float>(OrtMemoryInfo.DefaultInstance, data, [1, data.Length]);
    }
}