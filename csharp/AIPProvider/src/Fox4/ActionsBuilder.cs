using UnityGERunner.UnityApplication;

namespace AIPProvider.Fox4;

public class ActionsBuilder
{
    private static readonly IReadOnlyDictionary<WeaponType, string> weaponClassifications = new Dictionary<WeaponType, string>
    {
        { WeaponType.Radar, "Weapons/Missiles/AIM-120" },
        { WeaponType.Heat, "Weapons/Missiles/AIRS-T" }
    };

    private readonly OutboundState _state;
    private readonly List<int> _events = [ ];

    public ActionsBuilder(OutboundState state)
    {
        _state = state;
    }

    public int[] Build()
    {
        return _events.ToArray();
    }

    public bool TrySelectWeapon(WeaponType type)
    {
        if (type == WeaponType.Guns)
        {
            _events.Add((int)InboundAction.SelectHardpoint);
            _events.Add(-1);
            return true;
        }

        var idx = Array.IndexOf(_state.weapons, weaponClassifications[type]);
        if (idx == -1)
            return false;

        _events.Add((int)InboundAction.SelectHardpoint);
        _events.Add(idx);
        return true;
    }

    public void Fire()
    {
       _events.Add((int)InboundAction.Fire);
    }

    public bool TryFire(WeaponType type)
    {
        if (!TrySelectWeapon(type))
            return false;

        Fire();
        return true;
    }

    public void Chaff()
    {
        _events.Add((int)InboundAction.Chaff);
    }

    public void Flare()
    {
        _events.Add((int)InboundAction.Flare);
    }

    public void Uncage(bool uncaged)
    {
        _events.Add((int)InboundAction.SetUncage);
        _events.Add(Convert.ToInt32(uncaged));
    }
}

public enum WeaponType
{
    Guns,
    Heat,
    Radar
}