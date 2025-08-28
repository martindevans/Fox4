using System.Numerics;
using UnityGERunner.UnityApplication;

namespace AIPProvider.Extensions;

public static class MapExtensions
{
    public static float GetRadarAltitude(this Map map, Vector3 position)
    {
        var floor = map.GetHeightAtSubpoint(new UnityGERunner.Vector3(position.X, position.Y, position.Z));
        var alt = position.Y;
        return alt - floor;
    }
}