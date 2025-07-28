from flask import Flask, request, jsonify
import math
import tenseal as ts
import os
import base64
import requests
import time

app = Flask(__name__)

# === Earth radius in meters ===
R = 6371000

# Default triangle (can be updated via /set-triangle)
A_deg = (19.18051, 72.94143)
B_deg = (19.18084, 72.97384)
C_deg = (19.1617, 72.95051)

@app.route("/set-triangle", methods=["POST"])
def set_triangle():
    global A_deg, B_deg, C_deg
    try:
        data = request.get_json()
        A = tuple(data.get("A"))
        B = tuple(data.get("B"))
        C = tuple(data.get("C"))

        if not (A and B and C and len(A) == 2 and len(B) == 2 and len(C) == 2):
            return jsonify({"error": "Invalid triangle format"}), 400

        A_deg, B_deg, C_deg = A, B, C
        print("[DEBUG] Triangle updated by external request.")
        return jsonify({"message": "Triangle updated successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-triangle", methods=["GET"])
def get_triangle():
    return jsonify({
        "A": A_deg,
        "B": B_deg,
        "C": C_deg
    }), 200


def project_to_cartesian(lat_deg, lon_deg, ref_lat):
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    x = R * lon_rad * math.cos(ref_lat)
    y = R * lat_rad
    return (x, y)

CONTEXT_PATH = "/app/shared/context.tenseal"

def load_context():
    if os.path.exists(CONTEXT_PATH):
        with open(CONTEXT_PATH, "rb") as f:
            return ts.context_from(f.read())
    raise FileNotFoundError("CKKS context file not found.")

@app.route("/submit-user-location", methods=["POST"])
def submit_user_location():
    try:
        data = request.get_json()
        required = ["encrypted_c1", "encrypted_c2", "encrypted_c3", "encrypted_c4", "encrypted_c5", "encrypted_c6"]
        if not data or not all(k in data for k in required):
            return jsonify({"status": "error", "message": "Missing encrypted inputs."}), 400

        context = load_context()

        # Deserialize encrypted inputs
        c1 = ts.ckks_vector_from(context, base64.b64decode(data["encrypted_c1"]))
        c2 = ts.ckks_vector_from(context, base64.b64decode(data["encrypted_c2"]))
        c3 = ts.ckks_vector_from(context, base64.b64decode(data["encrypted_c3"]))
        c4 = ts.ckks_vector_from(context, base64.b64decode(data["encrypted_c4"]))
        c5 = ts.ckks_vector_from(context, base64.b64decode(data["encrypted_c5"]))
        c6 = ts.ckks_vector_from(context, base64.b64decode(data["encrypted_c6"]))
        
        encryption_time = float(data.get("enc_time", 0.0))  # Capture user encryption time
        # Recalculate reference latitude from current triangle
        ref_lat = math.radians((A_deg[0] + B_deg[0] + C_deg[0]) / 3)

        # Re-project triangle using latest coordinates
        x1, y1 = project_to_cartesian(*A_deg, ref_lat)
        x2, y2 = project_to_cartesian(*B_deg, ref_lat)
        x3, y3 = project_to_cartesian(*C_deg, ref_lat)

        print("[DEBUG] Projected Triangle Coordinates:")
        print(f"        A_proj: ({x1:.2f}, {y1:.2f})")
        print(f"        B_proj: ({x2:.2f}, {y2:.2f})")
        print(f"        C_proj: ({x3:.2f}, {y3:.2f})")

        # Encrypted f values
        start_f = time.time()
        f1 = c2 * (x3 - x2) - c5 * (y3 - y2)
        f2 = c3 * (x1 - x3) - c6 * (y1 - y3)
        f3 = c1 * (x2 - x1) - c4 * (y2 - y1)
        end_f = time.time()
        f_time = end_f - start_f

        # Plaintext denominators
        start_d = time.time()
        d1 = (x3 - x2)*(y1 - y2) - (y3 - y2)*(x1 - x2)
        d2 = (x1 - x3)*(y2 - y3) - (y1 - y3)*(x2 - x3)
        d3 = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
        end_d = time.time()
        d_time = end_d - start_d

        return send_geofence_result_to_carer(f1, f2, f3, d1, d2, d3, encryption_time, f_time, d_time)

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"status": "error", "message": str(e)}), 500

def send_geofence_result_to_carer(f1, f2, f3, d1, d2, d3, encryption_time, f_time, d_time):
    try:
        payload = {
            "f1": base64.b64encode(f1.serialize()).decode("utf-8"),
            "f2": base64.b64encode(f2.serialize()).decode("utf-8"),
            "f3": base64.b64encode(f3.serialize()).decode("utf-8"),
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "encryption_time": encryption_time,
            "computation_time": f_time,
            "denominator_time": d_time
        }

        print("[DEBUG] Sending encrypted f values to Carer Device...")
        response = requests.post("http://carerdevice:5002/submit-geofence-result", json=payload)
        response.raise_for_status()
        return response.json(), 200

    except Exception as e:
        print("[ERROR] Failed to contact Carer Device:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
