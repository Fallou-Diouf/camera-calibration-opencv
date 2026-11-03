
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
# 2. Chargement des points image détectés
# --------------------------------------------------

data = np.load("results/image_points_original_2040.npz")

image_points = data["image_points"]
image_names = data["image_names"]

# --------------------------------------------------
# 3. Construction des points 3D du damier
# --------------------------------------------------

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 15.0  # millimètres

objp = np.zeros((6 * 9, 3), dtype=np.float32)

objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# --------------------------------------------------
# 4. Estimation de la pose pour chaque image
# --------------------------------------------------

results = []

for i, corners in enumerate(image_points):

    # Exclure l'image écartée de la calibration
    if "155639" in str(image_names[i]):
        continue

    image_points_2d = corners.reshape(-1, 1, 2).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        objp,
        image_points_2d,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:

        # Conversion du vecteur de rotation en matrice
        R, _ = cv2.Rodrigues(rvec)

        # Position de la caméra dans le repère du damier
        camera_position = -R.T @ tvec

        print(f"\n===== {image_names[i]} =====")
        print("Vecteur de rotation rvec :")
        print(rvec)

        print("Matrice de rotation R :")
        print(R)

        print("Vecteur de translation tvec (mm) :")
        print(tvec)

        print("Position de la caméra dans le repère du damier (mm) :")
        print(camera_position)

        results.append({
            "image_name": image_names[i],
            "rvec": rvec,
            "tvec": tvec,
            "R": R,
            "camera_position": camera_position
        })

    else:
        print(f"Échec de l'estimation pour {image_names[i]}")

# --------------------------------------------------
# 5. Sauvegarde des résultats
# --------------------------------------------------

output_dir = Path("results/poses")
output_dir.mkdir(parents=True, exist_ok=True)

np.savez(
    output_dir / "camera_poses.npz",
    image_names=np.array([r["image_name"] for r in results]),
    rvecs=np.array([r["rvec"] for r in results]),
    tvecs=np.array([r["tvec"] for r in results]),
    rotations=np.array([r["R"] for r in results]),
    camera_positions=np.array([r["camera_position"] for r in results])
)

print(f"\n{len(results)} poses sauvegardées dans {output_dir}")
