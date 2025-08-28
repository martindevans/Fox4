import numpy as np

def remap(value, from_low, from_high, to_low, to_high, clamp=True):
    """
    Remaps a value from one numerical range to another, with clamping.

    Args:
        value: The input number to remap.
        from_low: The lower bound of the original range.
        from_high: The upper bound of the original range.
        to_low: The lower bound of the target range.
        to_high: The upper bound of the target range.
        clamp (bool): If True, the output will be clamped to the [to_low, to_high]
                      range. Defaults to True.

    Returns:
        The remapped number as a float.
    """
    # Handle the case where the input range has zero width to avoid division by zero
    if from_high == from_low:
        return (to_low + to_high) / 2

    # Calculate the remapped value
    remapped_value = to_low + (value - from_low) * (to_high - to_low) / (from_high - from_low)

    if clamp:
        # Clamp the output to the target range
        # This handles both normal (e.g., 0 to 100) and inverted (e.g., 100 to 0) ranges
        if to_low < to_high:
            return max(to_low, min(remapped_value, to_high))
        else:
            return max(to_high, min(remapped_value, to_low))

    return remapped_value

def altitude_target(state, target, weight):
    alt_err = state['altitude'] - target

    # Return max weight if near target altitude
    if (abs(alt_err) < 0.02):
        return weight * 2

    # Reward/penalise based on if we're moving towards or away from the target
    vspeed_up = state['vert_speed'] > 0
    if (target > state['altitude']):
        if vspeed_up:
            return weight
        else:
            return -weight / 2
    else:
        if vspeed_up:
            return -weight / 2
        else:
            return weight

def low_angle_rates(state, weight):
    # Absolute rates
    p = abs(state['pitch_rate'])
    y = abs(state['yaw_rate'])
    r = abs(state['roll_rate'])

    # Any rate below 0.05 is good
    p = p - 0.05
    y = y - 0.05
    r = r - 0.05

    return -(p + y + r) / 3 * weight

def stay_alive(state, is_last):
    if (state['radar_altitude'] < 0.05):
        if is_last:
            return -10
        else:
            return -1
    else:
        return 0.1

def upright_roll(state, weight):
    # Roll vector (none)
    roll = np.array([
        state["sin(roll)"],
        state["cos(roll)"]
    ])

    # Dot target and actual (-1 to 1)
    dot = np.dot(roll, np.array([0, 1]))

    # Remap to (0 to 1)
    dot = dot / 2 + 0.5

    # Make reward sharper
    return pow(dot, 3) * weight

def low_aoa(state, weight):
    # aoa vector
    aoa = np.array([
        state["sin(aoa)"],
        state["cos(aoa)"]
    ])

    # Dot target and actual (-1 to 1)
    dot = np.dot(aoa, np.array([0, 1]))

    # Remap range, anything over 0.99 (8 degrees) gets no penalty, anything under 0.9 (25 degrees)
    # gets max penalty
    mapped = remap(dot, 0.9, 0.99, 1, 0, clamp=True)
    return -mapped * weight

class BaseScore():
    def __init__(self):
        pass

    def attach_score(self, df_input, df_output, df_extra):
        df_extra["score"] = df_input.apply(lambda row: 0, axis=1)

class StableAltitudeScore(BaseScore):
    def __init__(self, alt_weight=4, rate_weight=2, roll_weight=1, aoa_weight=0.5, target=7000):
        super().__init__()

        self.alt_weight = alt_weight
        self.rate_weight = rate_weight
        self.roll_weight = roll_weight
        self.aoa_weight = aoa_weight

        # 7000m == 22965ft
        self.alt_target = target / 15240
    
    def attach_score(self, df_input, df_output, df_extra):
        df_extra["score_rates"] = df_input.apply(lambda row: low_angle_rates(row, self.rate_weight), axis=1)
        df_extra["score_alt_tgt"] = df_input.apply(lambda row: altitude_target(row, self.alt_target, self.alt_weight), axis=1)
        df_extra["score_upright"] = df_input.apply(lambda row: upright_roll(row, self.roll_weight), axis=1)
        df_extra["aoa_penalty"] = df_input.apply(lambda row: low_aoa(row, self.aoa_weight), axis=1)

        last_idx = df_input.index[-1]
        df_extra["score_alive"] = df_input.apply(lambda row: stay_alive(row, row.name == last_idx), axis=1)

        df_extra["score"] = (
              df_extra["score_rates"]
            + df_extra["score_alt_tgt"]
            + df_extra["score_upright"]
            + df_extra["score_alive"]
            + df_extra["aoa_penalty"]
        )