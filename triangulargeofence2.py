import math
import random
import tenseal as ts
import csv
import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from shapely.geometry import Point, Polygon, LineString
import os

from UserDevice import compute_and_encrypt_user_location_terms
from CarerDevice.app import classify

def create_result_dirs():
    folders = [
        "results/accuracy",
        "results/runtime",
        "results/scalability_results",
        "results/security_overhead"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

ACCURACY_FILE = "results/accuracy/triangular_geofence_accuracy_export.csv"
WEIGHT_DIFF_FILE = "results/accuracy/weight_diff_summary.csv"
WEIGHT_DIFF_PLOT = "results/accuracy/grouped_weight_diff_plot.png"

RUNTIME_FILE = "results/runtime/runtime.csv"
RUNTIME_PLOT = "results/runtime/ckks_vs_groundtruth_runtime_labeled.png"

SECURITY_OVERHEAD_PLOT = "results/security_overhead/security_overhead_with_means.png"

SCALABILITY_FOLDER = "results/scalability_results"
SCALABILITY_SUMMARY_FILE = "results/scalability_results/scalability_summary.csv"

R = 6371000

T1 = [
    (19.18051, 72.94143),
    (19.18084, 72.97384),
    (19.1617, 72.95051)
]

ACCURACY_HEADERS = [
    "Triangle", "Class", "Latitude", "Longitude",
    "GT_w1", "GT_w2", "GT_w3",
    "CKKS_w1", "CKKS_w2", "CKKS_w3",
    "Δw1", "Δw2", "Δw3",
    "GT_Class", "CKKS_Class"
]

RUNTIME_HEADERS = [
    "Triangle", "Class", "Latitude", "Longitude", "Encryption Time (s)",
    "Computation Time (s)", "Decryption Time (s)", "Denominator Time (s)",
    "Ground Truth Time (s)", "Total CKKS Time (s)"
]

def scale_triangle(triangle, scale_factor):
    lat_c = sum([v[0] for v in triangle]) / 3
    lon_c = sum([v[1] for v in triangle]) / 3
    return [(lat_c + scale_factor * (lat - lat_c),
             lon_c + scale_factor * (lon - lon_c)) for lat, lon in triangle]
  

def send_triangle_to_server(triangle):
    url = "http://localhost:5001/set-triangle"
    payload = {
        "A": list(triangle[0]),
        "B": list(triangle[1]),
        "C": list(triangle[2])
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print(f"[✓] Triangle set on server.")
    else:
        print(f"[✗] Failed to set triangle: {response.json()}")

def get_context():
    r = requests.get("http://localhost:5002/get-public-key")
    r.raise_for_status()
    return ts.context_from(r.content)

def project_to_cartesian(lat, lon, ref_lat_rad):
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    x = R * lon_r * math.cos(ref_lat_rad)
    y = R * lat_r
    return x, y

def barycentric_weights(x, y, A, B, C):
    x1, y1 = A; x2, y2 = B; x3, y3 = C
    start = time.time()
    d1 = (x3 - x2)*(y1 - y2) - (y3 - y2)*(x1 - x2)
    d2 = (x1 - x3)*(y2 - y3) - (y1 - y3)*(x2 - x3)
    d3 = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
    w1 = ((x3 - x2)*(y - y2) - (y3 - y2)*(x - x2)) / d1
    w2 = ((x1 - x3)*(y - y3) - (y1 - y3)*(x - x3)) / d2
    w3 = ((x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)) / d3
    end = time.time()
    return w1, w2, w3, end - start



def generate_points(triangle, ref_lat_rad, n=30):
    A, B, C = [project_to_cartesian(lat, lon, ref_lat_rad) for lat, lon in triangle]
    polygon = Polygon([A, B, C])
    edges = [LineString([A, B]), LineString([B, C]), LineString([C, A])]
    lat_min = min(v[0] for v in triangle) - 0.01
    lat_max = max(v[0] for v in triangle) + 0.01
    lon_min = min(v[1] for v in triangle) - 0.01
    lon_max = max(v[1] for v in triangle) + 0.01

    inside, edge, outside = [], [], []
    while len(inside) < n or len(edge) < n or len(outside) < n:
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        x, y = project_to_cartesian(lat, lon, ref_lat_rad)
        pt = Point(x, y)
        if polygon.contains(pt) and len(inside) < n:
            inside.append(("INSIDE", lat, lon, x, y))
        elif any(e.distance(pt) < 1 for e in edges) and len(edge) < n:
            edge.append(("ON_BOUNDARY", lat, lon, x, y))
        elif not polygon.contains(pt) and len(outside) < n:
            outside.append(("OUTSIDE", lat, lon, x, y))
    return inside + edge + outside

def evaluate_point(label, lat, lon, x, y, triangle, ref_lat_rad, context):
    w1_gt, w2_gt, w3_gt, gt_time = barycentric_weights(x, y, *[project_to_cartesian(*v, ref_lat_rad) for v in triangle])
    gt_class = classify(w1_gt, w2_gt, w3_gt)
    enc_terms, enc_time = compute_and_encrypt_user_location_terms(lat, lon, triangle[0], triangle[1], triangle[2], ref_lat_rad, context)
    payload = {
        f"encrypted_c{i+1}": base64.b64encode(enc.serialize()).decode("utf-8")
        for i, enc in enumerate(enc_terms)
    }
    payload.update({"user_lat": lat, "user_lon": lon, "enc_time": enc_time})
    r = requests.post("http://localhost:5001/submit-user-location", json=payload)
    r.raise_for_status()
    result_json = r.json()

    w1 = float(result_json["weights"]["w1"])
    w2 = float(result_json["weights"]["w2"])
    w3 = float(result_json["weights"]["w3"])
    result = result_json["result"].upper()
    timing = result_json.get("timing", {})
    enc_time = timing.get("encryption_time", 0.0)
    comp_time = timing.get("computation_time", 0.0)
    dec_time = timing.get("decryption_time", 0.0)
    denom_time = timing.get("denominator_time", 0.0)
    total_ckks_time = enc_time + comp_time + dec_time + denom_time

    return {
        "label": label, "lat": lat, "lon": lon,
        "gt_weights": (w1_gt, w2_gt, w3_gt),
        "ckks_weights": (w1, w2, w3),
        "gt_class": gt_class, "ckks_class": result,
        "timing": (enc_time, comp_time, dec_time, denom_time, gt_time, total_ckks_time)
    }

def run_accuracy_all(context, triangles, write_header=False, write_csv=True):
    """
    Runs accuracy for each triangle, prints class-wise & overall accuracy,
    and (optionally) writes the detailed rows to CSV.
    """
    print("\n[ACCURACY EXPERIMENT]")

    CLASSES = ("INSIDE", "ON_BOUNDARY", "OUTSIDE")

    # Prepare CSV ONLY if requested
    writer = None
    f = None
    if write_csv:
        mode = "w" if write_header else "a"
        f = open(ACCURACY_FILE, mode, newline="")
        writer = csv.writer(f)
        if write_header:
            writer.writerow(ACCURACY_HEADERS)

    try:
        for label, triangle in triangles.items():
            print(f"[•] Running {label}...")

            # per-triangle counters
            per_class_total   = {c: 0 for c in CLASSES}
            per_class_correct = {c: 0 for c in CLASSES}
            correct_total = 0

            ref_lat_rad = math.radians(sum(v[0] for v in triangle) / 3)
            points = generate_points(triangle, ref_lat_rad)

            for p in points:
                result = evaluate_point(*p, triangle, ref_lat_rad, context)

                gt_cls   = result["gt_class"].strip().upper()
                ckks_cls = result["ckks_class"].strip().upper()

                is_match = (gt_cls == ckks_cls)
                correct_total += is_match

                if gt_cls in per_class_total:
                    per_class_total[gt_cls] += 1
                    if is_match:
                        per_class_correct[gt_cls] += 1

                # Only write CSV if requested
                if writer:
                    w1_gt, w2_gt, w3_gt = result["gt_weights"]
                    w1_ckks, w2_ckks, w3_ckks = result["ckks_weights"]
                    delta_w1 = abs(w1_ckks - w1_gt)
                    delta_w2 = abs(w2_ckks - w2_gt)
                    delta_w3 = abs(w3_ckks - w3_gt)

                    writer.writerow([
                        label, result["label"], result["lat"], result["lon"],
                        f"{w1_gt:.10f}", f"{w2_gt:.10f}", f"{w3_gt:.10f}",
                        f"{w1_ckks:.10f}", f"{w2_ckks:.10f}", f"{w3_ckks:.10f}",
                        f"{delta_w1:.10f}", f"{delta_w2:.10f}", f"{delta_w3:.10f}",
                        result["gt_class"], result["ckks_class"]
                    ])

            # pretty print per-class + overall for this triangle
            def pct(corr, tot): return (100.0 * corr / tot) if tot else 0.0

            overall_acc = pct(correct_total, len(points))
            inside_acc  = pct(per_class_correct["INSIDE"],      per_class_total["INSIDE"])
            onb_acc     = pct(per_class_correct["ON_BOUNDARY"], per_class_total["ON_BOUNDARY"])
            outside_acc = pct(per_class_correct["OUTSIDE"],     per_class_total["OUTSIDE"])

            print(f"{label} → Overall: {overall_acc:.2f}% ({correct_total}/{len(points)})")
            print(f"   INSIDE:      {inside_acc:.2f}% ({per_class_correct['INSIDE']}/{per_class_total['INSIDE']})")
            print(f"   ON_BOUNDARY: {onb_acc:.2f}% ({per_class_correct['ON_BOUNDARY']}/{per_class_total['ON_BOUNDARY']})")
            print(f"   OUTSIDE:     {outside_acc:.2f}% ({per_class_correct['OUTSIDE']}/{per_class_total['OUTSIDE']})")

    finally:
        if f:
            f.close()

def generate_weight_diff_summary():
    df = pd.read_csv(ACCURACY_FILE)
    df["Class"] = df["Class"].astype(str).str.upper()

    grouped = df.groupby(["Triangle", "Class"])[["Δw1", "Δw2", "Δw3"]].mean().reset_index()
    grouped[["Δw1", "Δw2", "Δw3"]] = grouped[["Δw1", "Δw2", "Δw3"]].round(10)

    grouped.to_csv("weight_diff_summary.csv", index=False)
    print("[✓] Combined weight difference summary saved to: weight_diff_summary.csv")

def plot_weight_differences_grouped(csv_path=WEIGHT_DIFF_FILE, save_path=WEIGHT_DIFF_PLOT):
    """
    Generate and save a grouped bar plot of Δw1, Δw2, and Δw3 by Triangle and Class.

    Parameters:
        csv_path (str): Path to the weight_diff_summary CSV file.
        save_path (str): Path to save the output PNG image.
    """
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Create grouped X-axis labels
    df["Group"] = df["Triangle"] + "\n" + df["Class"]
    x = np.arange(len(df))  # Each triangle-class pair is one group
    width = 0.25  # Width of each bar

    # Plot setup
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, df["Δw1"], width, label="Δw1", color='skyblue')
    ax.bar(x, df["Δw2"], width, label="Δw2", color='orange')
    ax.bar(x + width, df["Δw3"], width, label="Δw3", color='seagreen')

    # Labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels(df["Group"], rotation=45, ha='right')
    ax.set_ylabel("Mean Δw (Absolute Difference)")
    ax.set_title("Δw1, Δw2, Δw3 by Triangle and Class")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Finalize
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"[✓] Grouped bar plot saved as {save_path}")

def run_runtime_all(context, triangles):
    print("\n[RUNTIME EXPERIMENT]")
    with open(RUNTIME_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(RUNTIME_HEADERS)

        for label, triangle in triangles.items():
            print(f"[•] Evaluating runtime for {label}...")
            ref_lat_rad = math.radians(sum(v[0] for v in triangle) / 3)
            points = generate_points(triangle, ref_lat_rad)

            for i, p in enumerate(points):
                result = evaluate_point(*p, triangle, ref_lat_rad, context)
                enc, comp, dec, denom, gt, total = result["timing"]

                writer.writerow([
                    label, result["label"], result["lat"], result["lon"],
                    round(enc, 6), round(comp, 6), round(dec, 6),
                    round(denom, 6), round(gt, 6), round(total, 6)
                ])

            print(f"[✓] {label} completed. {len(points)} points processed.\n")

def generate_runtime_detailed_summary(csv_path="results/runtime/runtime.csv", summary_path="results/runtime/runtime_detailed_summary.csv"):
    df = pd.read_csv(csv_path)
    df["Class"] = df["Class"].astype(str).str.upper()

    # Define relevant timing columns
    time_columns = [
        "Encryption Time (s)",
        "Computation Time (s)",
        "Decryption Time (s)",
        "Denominator Time (s)"
    ]

    # Group by Triangle and Class, compute mean, min, max
    grouped = df.groupby(["Triangle", "Class"])[time_columns].agg(['mean', 'min', 'max'])

    # Reformat: MultiIndex to single-level and melt for long-format
    grouped.columns = ['_'.join(col).replace(" ", "_") for col in grouped.columns]
    grouped.reset_index(inplace=True)

    # Convert to long format for Word-friendly table
    long_format = pd.melt(
        grouped,
        id_vars=["Triangle", "Class"],
        var_name="Metric",
        value_name="Time (s)"
    )

    # Split metric into Phase and Stat (e.g., Encryption_Time_mean → Encryption, mean)
    long_format[["Phase", "Stat"]] = long_format["Metric"].str.extract(r"(.+)_([a-z]+)$")
    long_format.drop(columns=["Metric"], inplace=True)

    # Pivot to wide format: One row per Triangle, Class, Phase with Min, Mean, Max columns
    final = long_format.pivot_table(
        index=["Triangle", "Class", "Phase"],
        columns="Stat",
        values="Time (s)"
    ).reset_index()

    # Optional rounding
    final = final.round(6)

    # Save as clean CSV for Word usage
    final.to_csv(summary_path, index=False)
    print(f"[✓] Word-friendly runtime summary saved to: {summary_path}")

def plot_runtime_graph():
    df = pd.read_csv(RUNTIME_FILE)
    df["Class"] = df["Class"].str.upper()

    grouped = df.groupby(["Triangle", "Class"]).agg({
        "Total CKKS Time (s)": "mean",
        "Ground Truth Time (s)": "mean"
    }).reset_index()

    triangle_order = ["T1 (Default)", "T2 (Scaled ×1.5)", "T3 (Scaled ×2.0)"]
    class_order = ["INSIDE", "ON_BOUNDARY", "OUTSIDE"]

    labels, ckks_vals, gt_vals = [], [], []

    for t in triangle_order:
        for c in class_order:
            row = grouped[(grouped["Triangle"] == t) & (grouped["Class"] == c)]
            labels.append(f"{t}\n{c}")
            if not row.empty:
                ckks_vals.append(row["Total CKKS Time (s)"].values[0])
                gt_vals.append(row["Ground Truth Time (s)"].values[0])
            else:
                ckks_vals.append(0)
                gt_vals.append(0)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width/2, ckks_vals, width, label='CKKS Time', color='skyblue')
    bars2 = ax.bar(x + width/2, gt_vals, width, label='Ground Truth Time', color='salmon')

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.4f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.6f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Average Time (s)')
    ax.set_title('PriTriGeo vs TriGeo Runtime by Triangle and Class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("ckks_vs_groundtruth_runtime_labeled.png", dpi=300)
    plt.show()
    print("[✓] Runtime plot saved as ckks_vs_groundtruth_runtime_labeled.png")

def compute_and_plot_security_overhead():
    df = pd.read_csv(RUNTIME_FILE)
    df["Class"] = df["Class"].str.upper()

    # Skip invalid rows
    df = df[df["Ground Truth Time (s)"] > 0].copy()
    df["Security Overhead"] = df["Total CKKS Time (s)"] / df["Ground Truth Time (s)"]

    triangle_order = ["T1 (Default)", "T2 (Scaled ×1.5)", "T3 (Scaled ×2.0)"]
    class_order = ["INSIDE", "ON_BOUNDARY", "OUTSIDE"]
    colors = {
        "T1 (Default)": "tab:blue",
        "T2 (Scaled ×1.5)": "tab:orange",
        "T3 (Scaled ×2.0)": "tab:green"
    }

    # Group and pivot for plotting
    grouped = df.groupby(["Triangle", "Class"])["Security Overhead"].mean().unstack().reindex(triangle_order)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    x = np.arange(len(class_order))

    for idx, triangle in enumerate(triangle_order):
        offsets = x + (idx - 1) * width
        bar = ax.bar(offsets, grouped.loc[triangle], width=width,
                     label=triangle, color=colors[triangle], alpha=0.85)
        # Annotate each bar
        for rect in bar:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}×", xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Mean lines for each triangle
    mean_df = df.groupby("Triangle")[["Total CKKS Time (s)", "Ground Truth Time (s)"]].mean()
    mean_df["Mean Overhead"] = mean_df["Total CKKS Time (s)"] / mean_df["Ground Truth Time (s)"]

    for idx, triangle in enumerate(triangle_order):
        mean_val = mean_df.loc[triangle, "Mean Overhead"]
        ax.axhline(y=mean_val, linestyle='--', linewidth=1.2,
                   label=f"{triangle} Mean: {mean_val:.2f}×",
                   color=colors[triangle], alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(class_order)
    ax.set_ylabel("Security Overhead (Encrypted / Plaintext)")
    ax.set_title("Security Overhead per Triangle and Class")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, grouped.values.max() * 1.2)

    plt.tight_layout()
    plt.savefig(SECURITY_OVERHEAD_PLOT, dpi=300)
    plt.show()
    print(f"[✓] Security Overhead plot saved → {SECURITY_OVERHEAD_PLOT}")

def run_scalability_experiment(context, triangles, point_counts):
    print("\n[📈 SCALABILITY EXPERIMENT]")
    output_dir = SCALABILITY_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    for label, triangle in triangles.items():
        send_triangle_to_server(triangle)
        ref_lat_rad = math.radians(sum(v[0] for v in triangle) / 3)

        filename = os.path.join(output_dir, f"{label.replace(' ', '_').replace('(', '').replace(')', '').replace('×', 'x')}_scalability.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Triangle", "Total Points", "Class", "Latitude", "Longitude",
                "Encryption Time (s)", "Computation Time (s)", "Decryption Time (s)",
                "Total CKKS Time (s)"
            ])

            for count in point_counts:
                print(f"[•] {label} → {count} points")

                # Get `count` number of points randomly (any class)
                all_points = generate_points(triangle, ref_lat_rad, n=count)
                sampled_points = random.sample(all_points, count)

                for p in sampled_points:
                    result = evaluate_point(*p, triangle, ref_lat_rad, context)
                    enc, comp, dec, _, _, total = result["timing"]
                    writer.writerow([
                        label, count, result["label"], result["lat"], result["lon"],
                        round(enc, 6), round(comp, 6), round(dec, 6), round(total, 6)
                    ])

        print(f"[✓] Finished {label} — saved to {filename}")

def generate_scalability_summary(folder="results/scalability_results", summary_file="results/scalability_results/detailed_summary.csv"):
    summary_rows = []

    for filename in os.listdir(folder):
        #if filename in ["scalability_summary.csv", "scalability_detailed_summary.csv"]:
        #   continue
        if filename.endswith(".csv") and "scalability" in filename:
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath)

            # Ensure required columns exist
            if "Total CKKS Time (s)" not in df.columns or "Total Points" not in df.columns:
                print(f"[!] Skipping file (missing columns): {filename}")
                continue

            # Group by Total Points to calculate mean, min, and max CKKS time for each batch
            grouped = df.groupby("Total Points")["Total CKKS Time (s)"].agg(['mean', 'min', 'max']).reset_index()

            for _, row in grouped.iterrows():
                summary_rows.append({
                    "Triangle": filename.replace("_scalability.csv", "").replace(".csv", ""),
                    "Total Points": int(row["Total Points"]),
                    "Mean CKKS Time (s)": round(row["mean"], 6),
                    "Min CKKS Time (s)": round(row["min"], 6),
                    "Max CKKS Time (s)": round(row["max"], 6)
                })

    # Save combined summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(by=["Triangle", "Total Points"], inplace=True)
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    summary_df.to_csv(summary_file, index=False)
    return summary_df

def plot_scalability_summary(csv_path="results/scalability_results/detailed_summary.csv",
                              save_path="results/scalability_results/ckks_scalability_plot.png"):
    df = pd.read_csv(csv_path)

    # Unique triangle names
    triangles = df["Triangle"].unique()
    
    plt.figure(figsize=(10, 6))

    for triangle in triangles:
        subset = df[df["Triangle"] == triangle]
        plt.plot(subset["Total Points"], subset["Mean CKKS Time (s)"], marker='o', label=triangle)

    plt.title("CKKS Runtime Scalability Across Triangles")
    plt.xlabel("Number of Points")
    plt.ylabel("Mean CKKS Time (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✓] Scalability plot saved to: {save_path}")

def run_all():
    create_result_dirs()
    context = get_context()
    triangles = {
        "T1 (Default)": T1,
        "T2 (Scaled ×1.5)": scale_triangle(T1, 1.5),
        "T3 (Scaled ×2.0)": scale_triangle(T1, 2.0)
    }
    for i, (label, triangle) in enumerate(triangles.items()):
        print(f"\n[🚀] Running experiments for: {label}")
        send_triangle_to_server(triangle)
        #run_accuracy_all(context, {label: triangle}, write_header=(i == 0), write_csv=True)


    #generate_weight_diff_summary()
    #plot_weight_differences_grouped()
    #run_runtime_all(context, triangles)
    #generate_runtime_detailed_summary()
    plot_runtime_graph()
    #compute_and_plot_security_overhead()
    point_counts = [50, 100, 200, 400, 500, 700, 1000]
    #run_scalability_experiment(context, triangles, point_counts)
    #generate_scalability_summary()
    #plot_scalability_summary()

if __name__ == "__main__":
    run_all()