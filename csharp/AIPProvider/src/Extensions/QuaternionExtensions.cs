using System.Numerics;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Extensions;

public static class QuaternionExtensions
{
    public static Quaternion To(this NetQuaternion nq)
    {
        return new Quaternion(nq.x, nq.y, nq.z, nq.w);
    }

    public static float[] EncodeFrameOfReference(this NetQuaternion q)
    {
        return q.quat.EncodeFrameOfReference();
    }

    public static float[] EncodeFrameOfReference(this System.Numerics.Quaternion q)
    {
        return new UnityGERunner.Quaternion(q.X, q.Y, q.Z, q.W).EncodeFrameOfReference();
    }

    public static float[] EncodeFrameOfReference(this UnityGERunner.Quaternion q)
    {
        var fwd = q * UnityGERunner.Vector3.forward;
        var right = q * UnityGERunner.Vector3.right;
        var up = q * UnityGERunner.Vector3.up;

        return
        [
            fwd.x, fwd.y, fwd.z,
            right.x, right.y, right.z,
            up.x, up.y, up.z,
        ];
    }
}