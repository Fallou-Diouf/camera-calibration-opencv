
import cv2
import glob
import os
import numpy as np

CHECKERBOARD = (9, 6)
IMAGE_DIR = "images_original_2040"
OUTPUT_DIR = "results/corners_original_2040"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))

# Critères d'arrêt du raffinement sous-pixel
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

all_image_points = []
image_names = []
image_size = None

print(f"Nombre d'images : {len(image_paths)}")

for image_path in image_paths:

    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERREUR] Impossible de lire {image_path}")
        continue

    # Normaliser l'orientation : toutes les images en paysage
    height, width = image.shape[:2]

    if height > width:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    height, width = image.shape[:2]
    current_size = (width, height)

    if image_size is None:
        image_size = current_size
    elif current_size != image_size:
        print(f"[ERREUR] Dimension différente : {image_path}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Détection initiale des coins
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None
    )

    if ret:
        # 2. Raffinement sous-pixel
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        all_image_points.append(corners_refined)
        image_names.append(os.path.basename(image_path))

        # Afficher les coins affinés
        cv2.drawChessboardCorners(
            image,
            CHECKERBOARD,
            corners_refined,
            ret
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            os.path.basename(image_path)
        )
        cv2.imwrite(output_path, image)

        print(
            f"[OK] {os.path.basename(image_path)} : "
            f"{len(corners_refined)} coins affinés"
        )

    else:
        print(f"[ÉCHEC] {os.path.basename(image_path)}")

# Sauvegarde des coordonnées pour la calibration
if all_image_points:
    np.savez(
        "results/image_points_original_2040.npz",
        image_points=np.array(all_image_points, dtype=np.float32),
        image_names=np.array(image_names),
        image_size=np.array(image_size)
    )

print("\n===== RÉSULTAT =====")
print(f"Images traitées : {len(image_paths)}")
print(f"Images réussies : {len(all_image_points)}")
print(f"Points affinés : {len(all_image_points) * 54}")
print(f"Dimensions normalisées : {image_size}")
