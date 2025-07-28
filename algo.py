import math

# Triangle vertices (in degrees)
A_deg = (19.180517, 72.941430)
B_deg = (19.180841, 72.973848)
C_deg = (19.1617, 72.95051)

# === Earth radius in meters ===
R = 6371000

# === Reference latitude for projection (approx center of triangle) ===
ref_lat = math.radians((A_deg[0] + B_deg[0] + C_deg[0]) / 3)

def project_to_cartesian(lat_deg, lon_deg):
    """Project lat/lon to x/y using equirectangular approximation"""
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    x = R * (lon_rad) * math.cos(ref_lat)
    y = R * (lat_rad)
    return (x, y)

# Project triangle vertices to Cartesian coordinates
A = project_to_cartesian(*A_deg)
B = project_to_cartesian(*B_deg)
C = project_to_cartesian(*C_deg)

def barycentric_weights(x, y, A, B, C):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C

    d1 = (x3 - x2)*(y1 - y2) - (y3 - y2)*(x1 - x2)
    d2 = (x1 - x3)*(y2 - y3) - (y1 - y3)*(x2 - x3)
    d3 = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
    w1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / d1
    w2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / d2
    w3 = ((x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)) / d3

    return w1, w2, w3

def test_barycentric_with_curvature(user_lat_deg, user_lon_deg):
    x, y = project_to_cartesian(user_lat_deg, user_lon_deg)
    w1, w2, w3 = barycentric_weights(x, y, A, B, C)

    print(f"\n[TEST] For point ({user_lat_deg}, {user_lon_deg}):")
    print(f"  → Weights: w1 = {w1:.6f}, w2 = {w2:.6f}, w3 = {w3:.6f}")
    print(f"  → Sum = {w1 + w2 + w3:.20f}")

    classification = classify_point(w1, w2, w3)
    return classification

def classify_point(w1, w2, w3, epsilon=1e-2):  # ±(approx. 45m) tolerance
    # If any weight is negative beyond epsilon, clearly outside
    if w1 < -epsilon or w2 < -epsilon or w3 < -epsilon:
        return "outside"
    
    # If any weight is within the ±epsilon margin of zero, treat as boundary
    elif abs(w1) < epsilon or abs(w2) < epsilon or abs(w3) < epsilon:
        return "on_boundary"
    
    return "inside"

# === Try some test points ===
test_points = [
    (19.177, 72.95089), #clearly inside P1
    (19.16705, 72.95182), #clearly inside P2
    (19.17858, 72.966), #clear inside P3
    (19.18123, 72.95276), #clear outside P4
    (19.18063, 72.97282), #inside near, vertex P5
    (19.18078, 72.97366), #inside more close to vertex P6
    (19.16736, 72.95862), #82 meter away from border P7
    (19.16763, 72.95829), # 35m away P8
    (19.16714, 72.95783), # 50m away P9
    (19.16674, 72.95757), # around 65m away P10
    (19.16651, 72.95765), # around 80-90m away P11
    (19.16668, 72.9576), #around 75m away P12
    (19.16763, 72.95833), # 40m away 
    (19.16759, 72.95838), # 48m away
    (19.17178, 72.96231), #40m inside
    (19.176872, 72.968993), # on boundary
]

for lat, lon in test_points:
    result = test_barycentric_with_curvature(lat, lon)
    print(f"  → Classification: {result.upper()}")






# === Try some test points ===
#test_points = [
#    (19.18043, 72.94356), #inside
#    (19.16363, 72.95382), #outside
#    (19.18063, 72.97282), #outside near border
#    (19.17858, 72.966), #clearly inside
#    (19.16965, 72.94635) #outside
