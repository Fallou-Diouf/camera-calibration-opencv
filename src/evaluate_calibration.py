import cv2
import numpy as np

# Charger les paramètres de calibration
calibration = np.load("results/calibration_original_2040.npz")

K = calibration["K"]
dist = calibration["dist"]
rvecs = calibration["rvecs"]
tvecs = calibration["tvecs"]
rms = float(calibration["rms"])

# Charger les points détectés
data = np.load("results/image_points_original_2040.npz")
image_points = data["image_points"]
image_names = data["image_names"]

# Construire les points 3D du damier
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 15.0  # mm

objp = np.zeros((6 * 9, 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp *= SQUARE_SIZE

errors = []

print("\n===== ERREURS PAR IMAGE =====")

for i in range(len(image_points)):
    projected_points, _ = cv2.projectPoints(
        objp,
        rvecs[i],
        tvecs[i],
        K,
        dist
    )

    observed = image_points[i].reshape(-1, 2)
    projected = projected_points.reshape(-1, 2)

    # Erreur euclidienne moyenne par coin
    distances = np.linalg.norm(observed - projected, axis=1)
    mean_error = np.mean(distances)

    errors.append(mean_error)

    print(f"{image_names[i]} : {mean_error:.4f} px")

errors = np.array(errors)

print("\n===== BILAN =====")
print(f"RMS global OpenCV : {rms:.4f} px")
print(f"Erreur moyenne par coin : {errors.mean():.4f} px")
print(f"Erreur maximale par image : {errors.max():.4f} px")
print(f"Erreur minimale par image : {errors.min():.4f} px")
print(f"Écart-type : {errors.std():.4f} px")