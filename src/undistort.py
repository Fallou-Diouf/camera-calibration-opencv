
import cv2
import numpy as np
from pathlib import Path

# 1. Charger la calibration retenue
calibration = np.load(
    "results/calibration_original_2040.npz"
)

K = calibration["K"]
dist = calibration["dist"]

# 2. Charger une image
image_path = "images_original_2040/20261028_155614.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image introuvable : {image_path}")

# 3. Normaliser l'orientation comme dans detect_corners.py
h, w = image.shape[:2]

if h > w:
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

h, w = image.shape[:2]

# Vérifier les dimensions avec celles de la calibration
data = np.load("results/image_points_original_2040.npz")
expected_size = tuple(data["image_size"])

assert (w, h) == expected_size, (
    f"Dimensions incompatibles : {(w, h)} != {expected_size}"
)

print("Dimensions :", (w, h))
print("Calibration RMS :", calibration["rms"])

# 4. Calculer la nouvelle matrice intrinsèque
new_K, roi = cv2.getOptimalNewCameraMatrix(
    K, dist, (w, h), alpha=1, newImgSize=(w, h)
)

# 5. Corriger la distorsion
undistorted = cv2.undistort(
    image, K, dist, None, new_K
)

# 6. Sauvegarder les résultats
output_dir = Path("results/undistortion")
output_dir.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output_dir / "original_normalized.png"), image)
cv2.imwrite(str(output_dir / "corrected.png"), undistorted)

print("Correction terminée.")
print("ROI :", roi)



# Comparaison avant / après correction
comparison = np.hstack((image, undistorted))

cv2.imwrite(
    "results/undistortion/comparison.png",
    comparison
)

print("Comparaison sauvegardée.")
