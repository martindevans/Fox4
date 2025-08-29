namespace AIPProvider.Extensions;

public static class RandomExtensions
{
    /// <summary>
    /// Sample a rand value with normal distribution (mean=0, stddev=1)
    /// </summary>
    /// <param name="random"></param>
    /// <returns></returns>
    public static double SampleNormal(this Random random)
    {
        return random.SampleGaussian(0, 1);
    }

    /// <summary>
    /// Generates a sample from a normal (Gaussian) distribution.
    /// </summary>
    /// <param name="random"></param>
    /// <param name="mean">The mean (μ) of the distribution.</param>
    /// <param name="stddev">The standard deviation (σ) of the distribution.</param>
    /// <returns>A random variate from the normal distribution.</returns>
    public static double SampleGaussian(this Random random, double mean, double stddev)
    {
        // Box-Muller transform to generate a standard normal random variable.

        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();
        var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

        // Scale and shift to the desired mean and standard deviation.
        return mean + stddev * randStdNormal;
    }
}

public static class GammaDistribution
{
    public static double SampleGamma(this Random random, double shape, double scale)
    {
        // Marsaglia and Tsang's method for sampling from a Gamma distribution

        if (shape < 1)
            return random.SampleGamma(shape + 1, scale) * Math.Pow(random.NextDouble(), 1.0 / shape);

        var d = shape - 1.0 / 3.0;
        var c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double x;
            double v;
            do
            {
                x = random.SampleNormal();
                v = 1.0 + c * x;
            }
            while (v <= 0);
            v = v * v * v;
            var u = random.NextDouble();
            if (u < 1.0 - 0.0331 * (x * x) * (x * x) || Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v)))
                return (d * v * scale);
        }
    }
}

public static class BetaDistribution
{
    /// <summary>
    /// Generates a sample from a Beta distribution.
    /// </summary>
    /// <param name="random"></param>
    /// <param name="alpha">The alpha (α) shape parameter.</param>
    /// <param name="beta">The beta (β) shape parameter.</param>
    /// <returns>A random variate from the Beta distribution.</returns>
    public static double SampleBeta(this Random random, double alpha, double beta)
    {
        if (alpha <= 0)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha parameter must be positive.");
        if (beta <= 0)
            throw new ArgumentOutOfRangeException(nameof(beta), "Beta parameter must be positive.");

        var x = random.SampleGamma(alpha, 1.0);
        var y = random.SampleGamma(beta, 1.0);

        return x / (x + y);
    }

    /// <summary>
    /// Calculate the mean value of a beta distribution. This is the average if you sampled the distribution an infinite number of times.
    /// </summary>
    /// <param name="alpha"></param>
    /// <param name="beta"></param>
    /// <returns></returns>
    public static double BetaMean(double alpha, double beta)
    {
        return alpha / (alpha + beta);
    }

    /// <summary>
    /// Calculate the most likely value of a beta distribution. The point where the probability density function is at it's maximum.
    /// </summary>
    /// <param name="alpha"></param>
    /// <param name="beta"></param>
    /// <returns></returns>
    public static double BetaMode(double alpha, double beta)
    {
        if (alpha <= 1)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Beta mode is only defined for alpha > 1");
        if (beta <= 1)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Beta mode is only defined for beta > 1");

        return (alpha - 1) / (alpha + beta - 2);
    }
}