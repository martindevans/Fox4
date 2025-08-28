using UnityGERunner;

namespace AIPProvider.Fox4;

public class MathHelper
{
    public static float DeltaAngleRadians(float a, float b)
    {
        var ad = a / MathF.Tau * 360;
        var bd = b / MathF.Tau * 360;
        var delta = Mathf.DeltaAngle(ad, bd);
        return delta / 360 * MathF.Tau;
    }
}