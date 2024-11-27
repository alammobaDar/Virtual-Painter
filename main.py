import math

def ndc_to_cartesian(z_ndc, z_near, z_far):
    """Convert Z from NDC to Cartesian coordinates."""
    return ((z_ndc * (z_far - z_near)) + (z_near + z_far)) / 2

def calculate_position(x1, y1, x2, y2):
    """Calculate 2D position (distance between two points)."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_distance_3d(x1, y1, z1, x2, y2, z2):
    """Calculate 3D distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def calculate_weighted_distance(x1, y1, z1, x2, y2, z2, weight=1.0):
    """Calculate weighted 3D distance with Z contribution scaled by weight."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + weight * (z2 - z1)**2)

# Input data
x1, y1 = 0, 0  # Point 1 (2D)
x2, y2 = 249.56161563830284, 0  # Point 2 (2D)
z_ndc_1, z_ndc_2 = -0.1569426953792572, -0.1569426953792572  # Z values in NDC

# Projection parameters
z_near, z_far = 1, 100  # Near and far planes

# Convert Z from NDC to Cartesian
z1_cartesian = ndc_to_cartesian(z_ndc_1, z_near, z_far)
z2_cartesian = ndc_to_cartesian(z_ndc_2, z_near, z_far)

# Compute position and distances
position_2d = calculate_position(x1, y1, x2, y2)
distance_3d = calculate_distance_3d(x1, y1, z1_cartesian, x2, y2, z2_cartesian)
weighted_distance = calculate_weighted_distance(x1, y1, z1_cartesian, x2, y2, z2_cartesian, weight=0.5)

# Print results
print("Position (2D):", position_2d)
print("Distance (3D):", distance_3d)
print("Weighted Distance (w=0.5):", weighted_distance)
