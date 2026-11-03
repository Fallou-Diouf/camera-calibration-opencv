
import cv2
import numpy as np
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
# 3. Chargement des poses
# --------------------------------------------------

poses = np.load("results/poses/camera_poses.npz")

pose_names = poses["image_names"]
rvecs = poses["rvecs"]
tvecs = poses["tvecs"]

# --------------------------------------------------
# 4. Construction des points 3D du damier
# --------------------------------------------------

objp = np.zeros((6 * 9, 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp *= 15.0

# --------------------------------------------------
# 5. Sélection de l'image
# --------------------------------------------------

target_name = "20261028_155747.png"

def normalize_name(name):
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)

image_index = next(
    i for i, name in enumerate(image_names)
    if normalize_name(name) == target_name
)

pose_index = next(
    i for i, name in enumerate(pose_names)
    if normalize_name(name) == target_name
)

# --------------------------------------------------
# 6. Chargement de l'image originale
# --------------------------------------------------

image_path = Path("images_original_2040") / target_name

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Image introuvable : {image_path}")

# --------------------------------------------------
# 7. Reprojection des points 3D
# --------------------------------------------------

projected_points, _ = cv2.projectPoints(
    objp,
    rvecs[pose_index],
    tvecs[pose_index],
    K,
    dist
)

projected_points = projected_points.reshape(-1, 2)

observed_points = image_points[image_index].reshape(-1, 2)

# --------------------------------------------------
# 8. Dessin des points et des écarts
# --------------------------------------------------

for observed, projected in zip(observed_points, projected_points):

    p_obs = tuple(np.round(observed).astype(int))
    p_proj = tuple(np.round(projected).astype(int))

    # Coin détecté : vert
    cv2.circle(image, p_obs, 5, (0, 255, 0), -1)

    # Coin reprojeté : rouge
    cv2.drawMarker(
        image,
        p_proj,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=2
    )

    # Écart entre les deux positions
    cv2.line(image, p_obs, p_proj, (255, 0, 0), 1)

# --------------------------------------------------
# 9. Ajout d'une légende
# --------------------------------------------------

cv2.putText(
    image, "Detected: green | Projected: red",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7, (255, 255, 255), 2
)

# --------------------------------------------------
# 10. Sauvegarde
# --------------------------------------------------

output_dir = Path("results/poses")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "reprojection_visualization.png"

cv2.imwrite(str(output_path), image)

print(f"Image utilisée : {target_name}")
print(f"Visualisation enregistrée : {output_path}")

# Calcul de l'erreur pour cette image
errors = np.linalg.norm(
    observed_points - projected_points,
    axis=1
)

print(f"Erreur moyenne : {errors.mean():.4f} px")
print(f"RMSE : {np.sqrt(np.mean(errors**2)):.4f} px")
print(f"Erreur maximale : {errors.max():.4f} px")

# Affichage
cv2.imshow("Detected vs projected corners", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
