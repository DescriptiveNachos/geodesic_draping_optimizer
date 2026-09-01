import numpy as np


def objective(result, shear, vertices, faces, seed_xy, angle):
    """Example custom objective: mean shear weighted near locking angle."""
    lock_angle = 35.0
    penalty = 5.0
    exponent = 4.0
    shear = np.asarray(shear, dtype=float)
    shear = shear[np.isfinite(shear)]
    if not len(shear):
        return 90.0
    return float(np.mean(shear * (1.0 + penalty * (shear / lock_angle) ** exponent)))
