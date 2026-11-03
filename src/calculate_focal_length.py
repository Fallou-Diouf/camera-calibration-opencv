
import cv2
import numpy as np

# Charger la calibration finale
calibration = np.load(
    "results/calibration_without_outlier_18.npz"
)

K = calibration["K"]

# Résolution utilisée pour la calibration
image_size = (2040, 1530)

# Dimensions estimées du capteur physique (mm)
sensor_width = 5.2224
sensor_height = 3.9168

# Calcul des caractéristiques physiques
fov_x, fov_y, focal_length, principal_point, aspect_ratio = (
    cv2.calibrationMatrixValues(
        K,
        image_size,
        sensor_width,
        sensor_height
    )
)

print("\n===== CARACTÉRISTIQUES DE LA CAMÉRA =====")

print(f"Focale physique : {focal_length:.4f} mm")
print(f"Focale physique : {focal_length / 1000:.6f} m")

print(f"Champ de vision horizontal : {fov_x:.2f} degrés")
print(f"Champ de vision vertical : {fov_y:.2f} degrés")

print("Point principal :", principal_point)
print(f"Rapport d'aspect : {aspect_ratio:.6f}")

print("\nFocale EXIF : 4.0 mm")
