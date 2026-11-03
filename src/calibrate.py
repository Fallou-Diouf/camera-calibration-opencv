
import cv2
import numpy as np

# Géométrie du damier
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 15.0  # millimètres

# Charger les coordonnées des coins détectés et affinés
data = np.load("results/image_points_original_2040.npz")

image_points = data["image_points"]
image_size = tuple(data["image_size"])

number_of_images = len(image_points)

# Construire les coordonnées réelles du damier
objp = np.zeros((6 * 9, 3), dtype=np.float32)

objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# Associer les mêmes coordonnées 3D à chaque image
object_points = [
    objp.copy() for _ in range(number_of_images)
]

# Calibration de la caméra
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    list(image_points),
    image_size,
    None,
    None
)

# Afficher les résultats
print("\n===== RÉSULTATS DE LA CALIBRATION =====")

print(f"Nombre d'images : {number_of_images}")
print(f"Nombre de coins par image : {len(objp)}")
print(f"Dimensions des images : {image_size}")

print("\nErreur RMS de reprojection :")
print(ret)

print("\nMatrice intrinsèque K :")
print(K)

print("\nCoefficients de distorsion :")
print(dist)

print("\nNombre de poses estimées :")
print(len(rvecs))

# Sauvegarder les paramètres
np.savez(
    "results/calibration_original_2040.npz",
    K=K,
    dist=dist,
    rvecs=np.array(rvecs),
    tvecs=np.array(tvecs),
    rms=ret
)

print(
    "\nCalibration sauvegardée dans "
    "results/calibration_original_2040.npz"
)