import numpy as np
from scipy.interpolate import interp1d

def calculate_relative_thickness(coords):
    """
    Calculate the maximum relative thickness of an airfoil from its coordinates.
    coords: (N, 2) array of coordinates (Selig format: 1->0->1)
    returns: rel_thickness (float)
    """
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Identify the leading edge (min x)
    idx_le = np.argmin(x)
    x_le, x_max = np.min(x), np.max(x)
    chord = x_max - x_le
    
    if chord <= 0:
        return 0.0
        
    # Split into upper and lower surfaces
    surf1_x, surf1_y = x[:idx_le+1], y[:idx_le+1]
    surf2_x, surf2_y = x[idx_le:], y[idx_le:]
    
    # Create common x-stations for interpolation
    x_test = np.linspace(x_le, x_max, 101)
    
    # Sort surfaces to ensure monotonic x for interpolation
    s1 = np.argsort(surf1_x)
    s2 = np.argsort(surf2_x)
    
    # Interpolate
    y1_interp = np.interp(x_test, surf1_x[s1], surf1_y[s1])
    y2_interp = np.interp(x_test, surf2_x[s2], surf2_y[s2])
    
    # Max difference
    rel_thickness = np.max(np.abs(y1_interp - y2_interp)) / chord
    return float(rel_thickness)
