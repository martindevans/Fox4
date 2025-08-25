namespace AIPProvider.Extensions;

public static class IReadOnlyListExtensions
{
    public static int IndexOf<T>(this IReadOnlyList<T> list, T item)
        where T : IEquatable<T>
    {
        for (var i = 0; i < list.Count; i++)
            if (list[i].Equals(item))
                return i;

        return -1;
    }
}