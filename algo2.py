import math
from pyproj import Proj, Transformer

# === Initialize UTM projection for Mumbai region (Zone 43N) ===
proj = Proj(proj='utm', zone=43, ellps='WGS84')
transformer = Transformer.from_crs("epsg:4326", proj.srs, always_xy=True)

def project_to_cartesian(lat_deg, lon_deg):
    """Use UTM projection to project (lat, lon) to (x, y) in meters."""
    x, y = transformer.transform(lon_deg, lat_deg)
    return (x, y)

# Triangle vertices (in degrees)
A_deg = (19.180517, 72.941430)
B_deg = (19.180841, 72.973848)
C_deg = (19.162681, 72.952579)

# Project triangle vertices to Cartesian coordinates
A = project_to_cartesian(*A_deg)
B = project_to_cartesian(*B_deg)
C = project_to_cartesian(*C_deg)

def barycentric_weights(x, y, A, B, C):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C

    denominator = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    w1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denominator
    w2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denominator
    w3 = 1 - w1 - w2

    return w1, w2, w3

def test_barycentric_with_curvature(user_lat_deg, user_lon_deg):
    x, y = project_to_cartesian(user_lat_deg, user_lon_deg)
    w1, w2, w3 = barycentric_weights(x, y, A, B, C)

    print(f"\n[TEST] For point ({user_lat_deg}, {user_lon_deg}):")
    print(f"  → Weights: w1 = {w1:.6f}, w2 = {w2:.6f}, w3 = {w3:.6f}")
    print(f"  → Sum = {w1 + w2 + w3:.6f}")

    classification = classify_point(w1, w2, w3)
    return classification

def classify_point(w1, w2, w3, epsilon=1e-4):
    # 1. Clearly outside: any weight < -epsilon
    if w1 < -epsilon or w2 < -epsilon or w3 < -epsilon:
        return "outside"
    
    # 2. On boundary: any weight in the range [-epsilon, +epsilon]
    elif (abs(w1) <= epsilon and w1 < 0) or \
         (abs(w2) <= epsilon and w2 < 0) or \
         (abs(w3) <= epsilon and w3 < 0):
        return "outside"  # Still negative, just near-zero
    elif abs(w1) <= epsilon or abs(w2) <= epsilon or abs(w3) <= epsilon:
        return "on_boundary"
    
    # 3. Inside: All weights are in [0, 1]
    elif 0 <= w1 <= 1 and 0 <= w2 <= 1 and 0 <= w3 <= 1:
        return "inside"

# === Try some test points ===
test_points = [
    (19.177, 72.95089),
    (19.16705, 72.95182),
    (19.17858, 72.966),
    (19.18123, 72.95276),
    (19.18063, 72.97282),
    (19.18078, 72.97366),
    (19.16736, 72.95862),
    (19.16763, 72.95829),
    (19.16714, 72.95783),
    (19.16674, 72.95757),
    (19.16651, 72.95765),
    (19.16668, 72.9576),
    (19.16763, 72.95833),
    (19.16759, 72.95838),
    (19.17178, 72.96231),
    (19.170482, 72.947692)
]

for lat, lon in test_points:
    result = test_barycentric_with_curvature(lat, lon)
    print(f"  → Classification: {result.upper()}")