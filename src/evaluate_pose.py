
import cv2
import numpy as np
import csv
from pathlib import Path

# --------------------------------------------------
# 1. Chargement des paramètres de calibration
# --------------------------------------------------

calibration = np.load("results/calibration_without_outlier_18.npz")

K = calibration["K"]
dist = calibration["dist"]

# --------------------------------------------------
# 2. Chargement des points détectés
# --------------------------------------------------

data = np.load("results/image_points_original_2040.npz")

image_points = data["image_points"]
image_names = data["image_names"]

# --------------------------------------------------
# 3. Chargement des poses estimées
# --------------------------------------------------

poses = np.load("results/poses/camera_poses.npz")

pose_names = poses["image_names"]
rvecs = poses["rvecs"]
tvecs = poses["tvecs"]

# --------------------------------------------------
# 4. Construction des points 3D du damier
# --------------------------------------------------

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 15.0  # mm

objp = np.zeros((6 * 9, 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# --------------------------------------------------
# 5. Calcul des erreurs de reprojection
# --------------------------------------------------

results = []

for i, name in enumerate(pose_names):

    # Retrouver l'image correspondante
    indices = np.where(image_names == name)[0]

    if len(indices) == 0:
        print(f"Image introuvable : {name}")
        continue

    image_index = indices[0]

    observed_points = image_points[image_index].reshape(-1, 2)

    projected_points, _ = cv2.projectPoints(
        objp,
        rvecs[i],
        tvecs[i],
        K,
        dist
    )

    projected_points = projected_points.reshape(-1, 2)

    # Erreur euclidienne pour chaque coin
    errors = np.linalg.norm(
        observed_points - projected_points,
        axis=1
    )

    mean_error = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)

    results.append([
        str(name),
        mean_error,
        rmse,
        max_error
    ])

    print(f"\n{str(name)}")
    print(f"Erreur moyenne : {mean_error:.4f} px")
    print(f"RMSE : {rmse:.4f} px")
    print(f"Erreur maximale : {max_error:.4f} px")

# --------------------------------------------------
# 6. Statistiques globales
# --------------------------------------------------

results_array = np.array([r[1:] for r in results], dtype=float)

print("\n========== BILAN GLOBAL ==========")
print(f"Nombre d'images : {len(results)}")
print(f"Erreur moyenne : {results_array[:, 0].mean():.4f} px")
print(f"RMSE moyen par image : {results_array[:, 1].mean():.4f} px")
print(f"Erreur maximale observée : {results_array[:, 2].max():.4f} px")

# --------------------------------------------------
# 7. Sauvegarde des résultats
# --------------------------------------------------

output_dir = Path("results/poses")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "reprojection_errors.csv"

with open(output_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Image",
        "Erreur moyenne (px)",
        "RMSE (px)",
        "Erreur maximale (px)"
    ])

    writer.writerows(results)

print(f"\nRésultats enregistrés dans : {output_path}")
