using System.Numerics;

namespace AIPProvider.Extensions;

public static class Vector3Extensions
{
    public static Vector3 To(this UnityGERunner.Vector3 v)
    {
        return new Vector3(v.x, v.y, v.z);
    }
}