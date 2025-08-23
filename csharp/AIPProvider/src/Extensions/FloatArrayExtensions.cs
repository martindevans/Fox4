using Microsoft.ML.OnnxRuntime;

namespace AIPProvider.Extensions;

public static class FloatArrayExtensions
{
    public static void AddGaussianNoise(this float[] arr, Random rng, float dev)
    {
        for (var i = 0; i < arr.Length; i++)
            arr[i] += rng.NextGaussian(mean: 0, dev: dev);
    }

    public static void AddGaussianNoise(this float[] arr, Random rng, float stdGain, OrtValue deviations)
    {
        var deviationsSpan = deviations.GetTensorDataAsSpan<float>();

        for (var i = 0; i < arr.Length; i++)
            arr[i] += rng.NextGaussian(mean: 0, dev: stdGain * deviationsSpan[i]);
    }
}