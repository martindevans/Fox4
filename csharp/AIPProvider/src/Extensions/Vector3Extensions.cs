using System.Numerics;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Extensions;

public static class Vector3Extensions
{
    public static Vector3 To(this UnityGERunner.Vector3 v)
    {
        return new Vector3(v.x, v.y, v.z);
    }

    public static Vector3 To(this NetVector v)
    {
        return new Vector3(v.x, v.y, v.z);
    }
}