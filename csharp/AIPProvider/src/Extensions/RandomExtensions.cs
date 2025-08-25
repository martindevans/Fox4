namespace AIPProvider.Extensions;

public static class RandomExtensions
{
    public static float NextGaussian(this Random random, float mean, float dev)
    {
        // Irwin–Hall distribution, approaches a normal distribution as you sum more uniform samples.
        // Taken from Ephemeris/Kessler Orbital Warfare

        var val = 2 * (random.NextSingle() + random.NextSingle() + random.NextSingle() - 1.5f);
        return mean + val * dev;
    }
}