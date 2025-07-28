import requests
import tenseal as ts
import math
import time
import base64
import json

def fetch_triangle_vertices():
    url = "http://localhost:5001/get-triangle"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        A = tuple(data["A"])
        B = tuple(data["B"])
        C = tuple(data["C"])
        return A, B, C
    except Exception as e:
        print(f"[ERROR] Failed to fetch triangle from Geofence Service: {e}")
        return None, None, None

# === Earth radius in meters ===
R = 6371000


def project_to_cartesian(lat_deg, lon_deg, ref_lat_rad):
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    x = R * lon_rad * math.cos(ref_lat_rad)
    y = R * lat_rad
    return x, y

def get_carer_public_key():
    url = "http://localhost:5002/get-public-key"
    try:
        response = requests.get(url)
        response.raise_for_status()

        with open("context.tenseal", "wb") as f:
            f.write(response.content)
        with open("context.tenseal", "rb") as f:
            context_bytes = f.read()

        context = ts.context_from(context_bytes)
        print("Context loaded from Carer Device:")
        print("  ➤ Is private?", context.is_private())   # should be False
        return context

    except Exception as e:
        print(f"[ERROR] Failed to fetch public key from Carer Device: {e}")
        return None

def compute_and_encrypt_user_location_terms(user_lat_deg, user_lon_deg, A_deg, B_deg, C_deg, ref_lat_rad, context):
    

    # Project user and triangle vertices to x/y in meters
    x_user, y_user = project_to_cartesian(user_lat_deg, user_lon_deg, ref_lat_rad)
    x1, y1 = project_to_cartesian(*A_deg, ref_lat_rad)
    x2, y2 = project_to_cartesian(*B_deg, ref_lat_rad)
    x3, y3 = project_to_cartesian(*C_deg, ref_lat_rad)
    
    start = time.perf_counter()
    c1 = y_user - y1
    c2 = y_user - y2
    c3 = y_user - y3
    c4 = x_user - x1
    c5 = x_user - x2
    c6 = x_user - x3
    """
    print("[DEBUG] User Plaintext Terms:")
    print(f"c1 = y - y1 = {c1}")
    print(f"c2 = y - y2 = {c2}")
    print(f"c3 = y - y3 = {c3}")
    print(f"c4 = x - x1 = {c4}")
    print(f"c5 = x - x2 = {c5}")
    print(f"c6 = x - x3 = {c6}")
    """
    enc_c1 = ts.ckks_vector(context, [c1])
    enc_c2 = ts.ckks_vector(context, [c2])
    enc_c3 = ts.ckks_vector(context, [c3])
    enc_c4 = ts.ckks_vector(context, [c4])
    enc_c5 = ts.ckks_vector(context, [c5])
    enc_c6 = ts.ckks_vector(context, [c6])

    end = time.perf_counter()
    enc_time = end - start 
    #print("[DEBUG] Encryption time:", round(end - start, 4), "s")

    return [enc_c1, enc_c2, enc_c3, enc_c4, enc_c5, enc_c6], enc_time

def send_encrypted_location(enc_list, user_lat, user_lon, enc_time):
    try:
        enc_payload = {
            f"encrypted_c{i+1}": base64.b64encode(enc.serialize()).decode("utf-8")
            for i, enc in enumerate(enc_list)
        }

        payload = {
            **enc_payload,
            "user_lat": user_lat,
            "user_lon": user_lon,
            "enc_time": enc_time,
        }

        print("[DEBUG] Payload sent to geofencing service.")
        start_request = time.time()
        response = requests.post("http://localhost:5001/submit-user-location", json=payload)
        end_request = time.time()

        response.raise_for_status()
        print("[DEBUG] Round-trip time:", round(end_request - start_request, 4), "s")
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request to Geofencing service failed: {e}")
        return None

def main():
    user_lat_deg = 19.177440373193384
    user_lon_deg = 72.94551696961442

    A_deg, B_deg, C_deg = fetch_triangle_vertices()
    if not A_deg:
        print("[ERROR] Could not fetch triangle. Exiting.")
        return 
    
    # === Reference latitude for projection (approx center of triangle) ===

    ref_lat_rad = math.radians((A_deg[0] + B_deg[0] + C_deg[0]) / 3)
  

    context = get_carer_public_key()
    if not context:
        print("[ERROR] Could not load encryption context. Exiting.")
        return

    print("[DEBUG] Global scale:", context.global_scale)

    enc_list, enc_time = compute_and_encrypt_user_location_terms(user_lat_deg, user_lon_deg, A_deg, B_deg, C_deg, ref_lat_rad, context)
    result = send_encrypted_location(enc_list, user_lat_deg, user_lon_deg, enc_time)
    print("[DEBUG] Raw response from Geofencing Service:")
    print(json.dumps(result, indent=2))

    if result:
        print("\n===== FINAL RESULT =====")
        print(f"User Latitude: {user_lat_deg}")
        print(f"User Longitude: {user_lon_deg}")
        status = result.get('result')
        if status:
            print(f"User location status: The user is **{status.upper()}** the geofence.\n")
        else:
            print("User location status: Unable to determine geofence result.\n")
        print("========================\n")
    else:
        print("[DEBUG] No response received or 'result' key missing in response.")

if __name__ == "__main__":
    main()
