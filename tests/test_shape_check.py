import numpy as np
import matplotlib.pyplot as plt

def check_shape_intersections(coords):
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
            return True, f"Vertical line at {c:.4f} has {intersections} intersections (>2)"
            
    # horizontal lines
    y_sorted = np.unique(y)
    y_test = (y_sorted[:-1] + y_sorted[1:]) / 2.0
    y1 = y[:-1]
    y2 = y[1:]
    
    for c in y_test:
        intersections = np.sum((y1 - c) * (y2 - c) < 0)
        if intersections > 4:
            return True, f"Horizontal line at {c:.4f} has {intersections} intersections (>4)"
            
    return False, "Valid"

# Test with NACA 2412
dat_path = 'tests/naca2412.dat'
try:
    coords = np.loadtxt(dat_path, skiprows=1)
    print("NACA 2412:", check_shape_intersections(coords))
except Exception as e:
    print(f"Error loading {dat_path}: {e}")

# Create a folded airfoil in X
coords_fold = coords.copy()
coords_fold[10, 0] -= 0.1 # create a fold
print("Folded X:", check_shape_intersections(coords_fold))

# Create a wiggly airfoil in Y
t = np.linspace(0, 1, 100)
coords_wiggle = np.zeros((100, 2))
coords_wiggle[:, 0] = np.cos(t * np.pi) * 0.5 + 0.5
coords_wiggle[:, 1] = np.sin(t * np.pi * 5) * 0.1 # 5 half-waves, so > 4 intersections
print("Wiggly Y:", check_shape_intersections(coords_wiggle))
