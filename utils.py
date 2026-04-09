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

def check_intersection(coords):
    """
    Check if the airfoil curve self-intersects.
    coords: (N, 2) array of coordinates.
    returns: True if self-intersects, False otherwise.
    """
    N = len(coords)
    if N < 4:
        return False
        
    A = coords[:-1]
    B = coords[1:]
    
    def ccw(A, B, C):
        return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) - (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])
    
    A_exp = A[:, None, :]
    B_exp = B[:, None, :]
    C_exp = A[None, :, :]
    D_exp = B[None, :, :]
    
    ccw1 = ccw(A_exp, C_exp, D_exp)
    ccw2 = ccw(B_exp, C_exp, D_exp)
    ccw3 = ccw(A_exp, B_exp, C_exp)
    ccw4 = ccw(A_exp, B_exp, D_exp)
    
    intersect = ((ccw1 * ccw2) < 0) & ((ccw3 * ccw4) < 0)
    
    # Check only non-adjacent segments: index j > i + 1
    mask = np.triu(np.ones((N-1, N-1), dtype=bool), k=2)
    
    return np.any(intersect & mask)

def check_shape_intersections(coords):
    """
    Check if the airfoil shape is valid based on ray intersections.
    - Any vertical line should intersect the curve at most 2 times.
    - Any horizontal line should intersect the curve at most 4 times.
    Returns True if the shape is INVALID (fails the check), False if valid.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    
    # vertical lines
    x_sorted = np.unique(x)
    x_test = (x_sorted[:-1] + x_sorted[1:]) / 2.0
    x1 = x[:-1]
    x2 = x[1:]
    
    for c in x_test:
        intersections = np.sum((x1 - c) * (x2 - c) < 0)
        if intersections > 2:
            return True
            
    # horizontal lines
    y_sorted = np.unique(y)
    y_test = (y_sorted[:-1] + y_sorted[1:]) / 2.0
    y1 = y[:-1]
    y2 = y[1:]
    
    for c in y_test:
        intersections = np.sum((y1 - c) * (y2 - c) < 0)
        if intersections > 4:
            return True
            
    return False

