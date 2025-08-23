from flask import Flask, jsonify, Response, request
import tenseal as ts
import os
import time
import base64

app = Flask(__name__)
CONTEXT_PATH = "/app/shared/context.tenseal"

# === Create and save CKKS context (run once if not already created) ===
def create_and_save_context():
    if os.path.exists(CONTEXT_PATH):
        print("[INFO] CKKS context already exists. Skipping creation.")
        return

    print("[INFO] Creating new TenSEAL CKKS context...")
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()

    with open(CONTEXT_PATH, "wb") as f:
        f.write(context.serialize(save_secret_key=True))

    print("[INFO] CKKS context created and saved to disk.")

# === Load TenSEAL context with secret key ===
def load_context():
    if not os.path.exists(CONTEXT_PATH):
        raise FileNotFoundError(f"[ERROR] Context file not found at {CONTEXT_PATH}")
    
    with open(CONTEXT_PATH, "rb") as f:
        context = ts.context_from(f.read())

    print("\n[DEBUG] Carer Device: Context loaded.")
    print(f"         ➤ Can decrypt? {context.is_private()}")
    print(f"         ➤ Global scale: {context.global_scale:.1e}", flush=True)

    return context

# === Public key sharing endpoint for user device to get the public context ===
@app.route("/get-public-key", methods=["GET"])
def get_public_key():
    try:
        if not os.path.exists(CONTEXT_PATH):
            raise FileNotFoundError("Context file not found.")

        with open(CONTEXT_PATH, "rb") as f:
            context = ts.context_from(f.read())

        context.make_context_public()
        public_bytes = context.serialize()

        print("[INFO] Public CKKS context sent to client.")
        return Response(public_bytes, mimetype="application/octet-stream")

    except Exception as e:
        print("[ERROR] Failed to provide public key:", e)
        return jsonify({"error": str(e)}), 500
    
# === Classification logic ===
def classify(w1, w2, w3, epsilon=1e-3):
    if any(abs(w) < epsilon for w in [w1, w2, w3]):
        return "ON_BOUNDARY"
    elif all(w > 0 for w in [w1, w2, w3]):
        return "INSIDE"
    else:
        return "OUTSIDE"

# === Utility: Decode and parse encrypted vector ===
def parse_encrypted_vector(base64_data, context):
    try:
        encrypted_bytes = base64.b64decode(base64_data)
        return ts.ckks_vector_from(context, encrypted_bytes)
    except Exception as e:
        print("[ERROR] Failed to parse encrypted vector:", e)
        raise ValueError("Invalid base64-encoded encrypted data.")

# === Utility: Decrypt vector ===
def decrypt_encrypted_vector(enc_vec):
    try:
        decrypted_values = enc_vec.decrypt()
        print("[DEBUG] Decrypted CKKS vector:", decrypted_values)
        return decrypted_values
    except Exception as e:
        print("[ERROR] Decryption failed:", e)
        return None

# === Core Evaluate function ===
def evaluate_f_values(f1_enc, f2_enc, f3_enc, d1, d2, d3):
    start_dec = time.time()

    # Decrypt and normalize weights
    w1 = decrypt_encrypted_vector(f1_enc)[0] / d1
    w2 = decrypt_encrypted_vector(f2_enc)[0] / d2
    w3 = decrypt_encrypted_vector(f3_enc)[0] / d3

    end_dec = time.time()
    dec_time = end_dec - start_dec
    print(f"[DEBUG] Decryption started at: {start_dec:.6f}")
    print(f"[DEBUG] Decryption ended at: {end_dec:.6f}")
    print(f"[DEBUG] Total decryption duration: {end_dec - start_dec:.6f} seconds")

    print(f"[DEBUG] Decrypted Weights → w1: {w1}, w2: {w2}, w3: {w3}")
    result = classify(w1, w2, w3)
    return result, (w1, w2, w3), dec_time

# === Endpoint: Receive encrypted f1, f2, f3 and denominators ===
@app.route("/submit-geofence-result", methods=["POST"])
def submit_geofence_result():
    try:
        data = request.get_json()
        enc_time = float(data.get("encryption_time", 0))
        f_time = float(data.get("computation_time", 0))
        d_time = float(data.get("denominator_time", 0))
        required_fields = ["f1", "f2", "f3", "d1", "d2", "d3"]
        if not all(field in data for field in required_fields):
            return jsonify({"status": "error", "message": "Missing required fields."}), 400

        context = load_context()

        f1_enc = parse_encrypted_vector(data["f1"], context)
        f2_enc = parse_encrypted_vector(data["f2"], context)
        f3_enc = parse_encrypted_vector(data["f3"], context)

        d1, d2, d3 = float(data["d1"]), float(data["d2"]), float(data["d3"])

        result, weights, dec_time = evaluate_f_values(f1_enc, f2_enc, f3_enc, d1, d2, d3)
        # Print Result at carer logs
        print("[CARER] Final classification:", result, flush=True)
        print("[CARER] Barycentric weights (w1, w2, w3):", weights, flush=True)

        return jsonify({
            "status": "success",
            "result": result,
            "weights": {
                "w1": f"{weights[0]:.10f}",
                "w2": f"{weights[1]:.10f}",
                "w3": f"{weights[2]:.10f}"
            },
            "timing": {
                "encryption_time": float(data.get("encryption_time", enc_time)),
                "computation_time": float(data.get("computation_time", f_time)),
                "denominator_time": float(data.get("denominator_time", d_time)),
                "decryption_time": round(dec_time, 6)
            }
        })

    except Exception as e:
        print("[ERROR] submit-geofence-result failed:", e, flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# === Run Flask app ===
if __name__ == "__main__":
    create_and_save_context()
    app.run(host="0.0.0.0", port=5002, debug=True)

