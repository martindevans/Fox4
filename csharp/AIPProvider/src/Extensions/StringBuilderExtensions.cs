using System.Text;

namespace AIPProvider.Extensions;

public static class StringBuilderExtensions
{
    public static void AppendJoin(this StringBuilder builder, string join, ReadOnlySpan<float> values)
    {
        for (var i = 0; i < values.Length; i++)
        {
            builder.Append(values[i]);
            if (i != values.Length - 1)
                builder.Append(join);
        }
    }
}