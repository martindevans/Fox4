using AIPProvider.Extensions;
using UnityGERunner;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4;

public class AngleRateCalculator
{
    private Quaternion? _previous;

    public System.Numerics.Vector3 Update(NetQuaternion rotation, float dt)
    {
        if (!_previous.HasValue || dt == 0)
        {
            _previous = rotation;
            return System.Numerics.Vector3.Zero;
        }

        var prev = _previous.Value;
        var current = rotation.quat;
        _previous = rotation.quat;

        var delta = Quaternion.Inverse(prev) * current;
        var deltaEuler = delta.eulerAngles;

        deltaEuler.x = Mathf.DeltaAngle(0, deltaEuler.x);
        deltaEuler.y = Mathf.DeltaAngle(0, deltaEuler.y);
        deltaEuler.z = Mathf.DeltaAngle(0, deltaEuler.z);

        // Compute per-axis rates (degs/s)
        return (deltaEuler / dt).To();
    }
}