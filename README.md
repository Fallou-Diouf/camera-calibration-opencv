# Calibration d'une caméra avec OpenCV — Samsung Galaxy A14 4G

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.10%2B-red.svg)](https://cmake.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x%20%7C%205.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Présentation du projet

Ce projet consiste à réaliser la calibration d'une caméra de smartphone à partir d'une série de photographies d'un damier (*checkerboard*).

L'objectif est d'estimer les paramètres intrinsèques de la caméra, ses coefficients de distorsion optique et les paramètres extrinsèques correspondant aux différentes prises de vue. Les paramètres obtenus sont ensuite utilisés pour estimer les poses de la caméra, évaluer la qualité de la calibration à travers l'erreur de reprojection et corriger la distorsion des images.

Le projet a été réalisé avec Python et OpenCV dans le cadre d'un travail pratique de Vision par Ordinateur.

## Objectifs

- Détecter les coins intérieurs d'un damier dans plusieurs images.
- Raffiner la localisation des coins avec une précision subpixel.
- Estimer la matrice intrinsèque de la caméra.
- Estimer les coefficients de distorsion radiale et tangentielle.
- Calculer la longueur focale physique de la caméra.
- Estimer les poses de la caméra pour différentes prises de vue.
- Évaluer la calibration à l'aide de l'erreur de reprojection.
- Corriger la distorsion des images.

## Environnement expérimental

### Matériel utilisé

| Caractéristique | Valeur |
|---|---|
| Appareil | Samsung Galaxy A14 4G |
| Caméra | Caméra arrière principale |
| Zoom | 1× |
| Résolution originale | 4080 × 3060 pixels |
| Focale indiquée dans les métadonnées EXIF | 4,0 mm |
| Ouverture | f/1.8 |

### Damier utilisé

Le damier utilisé pour la calibration présente les caractéristiques suivantes :

- Nombre de coins intérieurs : 9 × 6.
- Nombre total de points détectés par image : 54.
- Taille d'un carré : 15 mm.
- Nombre de photographies acquises : 19.
- Nombre d'images retenues pour la calibration finale : 18.

Le damier a été affiché sur un écran d'ordinateur et sa taille a été mesurée à l'aide d'une règle.

## Méthodologie

Le projet est organisé autour des étapes suivantes.

### 1. Acquisition et préparation des images

Dix-neuf photographies du damier ont été prises avec différentes positions et orientations de la caméra.

Les images originales ont été converties au format PNG, puis redimensionnées à une résolution de 2040 × 1530 pixels. Les photographies en orientation portrait ont également été réorientées afin d'obtenir des dimensions homogènes.

### 2. Détection des coins du damier

La détection des 54 coins intérieurs est réalisée avec :

- `cv2.findChessboardCorners()` : détection initiale des coins.
- `cv2.cornerSubPix()` : raffinement de leur localisation à une précision subpixel.

Les coordonnées des points détectés sont sauvegardées pour être réutilisées lors de la calibration.

### 3. Calibration de la caméra

La calibration est effectuée avec `cv2.calibrateCamera()`.

Cette étape permet d'estimer :

- La matrice intrinsèque de la caméra.
- Les coefficients de distorsion optique.
- Les paramètres extrinsèques associés à chaque image.
- L'erreur quadratique moyenne de reprojection (RMS).

### 4. Estimation de la longueur focale physique

La fonction `cv2.calibrationMatrixValues()` est utilisée pour convertir les paramètres focaux exprimés en pixels en une longueur focale physique.

Cette conversion nécessite de renseigner les dimensions physiques du capteur.

### 5. Estimation des poses de la caméra

La fonction `cv2.solvePnP()` est utilisée pour estimer la rotation et la translation de la caméra par rapport au repère du damier pour chacune des 18 images.

Les positions des centres des caméras et leurs orientations sont ensuite représentées dans une visualisation tridimensionnelle.

### 6. Évaluation par erreur de reprojection

Les points 3D du damier sont reprojetés dans le plan image à partir des paramètres estimés.

Les positions reprojetées sont comparées aux coins initialement détectés afin de mesurer les écarts géométriques.

### 7. Correction de la distorsion

Enfin, les paramètres estimés sont utilisés pour corriger les images avec :

- `cv2.getOptimalNewCameraMatrix()`
- `cv2.undistort()`

Une comparaison avant/après correction permet de visualiser les transformations géométriques obtenues.

## Résultats expérimentaux

### Paramètres intrinsèques

La matrice intrinsèque estimée est :

\[
K =
\begin{bmatrix}
1638.6939 & 0 & 1020.4124 \\
0 & 1640.5535 & 762.8702 \\
0 & 0 & 1
\end{bmatrix}
\]

Les coefficients de distorsion obtenus sont :

\[
D =
\begin{bmatrix}
0.2061 & -0.6083 & 0.000596 &
0.000585 & -0.2928
\end{bmatrix}
\]

### Longueur focale physique

| Mesure | Valeur |
|---|---:|
| Focale estimée | 4,1951 mm |
| Focale estimée en mètres | 0,0041951 m |
| Focale EXIF | 4,0 mm |
| Champ de vision horizontal | 63,80° |
| Champ de vision vertical | 50,00° |

### Évaluation de la calibration et des poses

| Indicateur | Résultat |
|---|---:|
| Nombre d'images utilisées | 18 |
| RMS de calibration | 0,2678 pixel |
| Erreur moyenne de reprojection | 0,2146 pixel |
| Moyenne des RMSE par image | 0,2484 pixel |
| Erreur maximale observée | 1,2359 pixel |

Ces résultats montrent une bonne cohérence entre les points détectés dans les images et leurs positions reprojetées par le modèle de caméra.

## Technologies utilisées

- **Python** : langage de programmation.
- **OpenCV** : traitement d'images, calibration, estimation de pose et correction de distorsion.
- **NumPy** : manipulation des matrices et des données numériques.
- **Matplotlib** : visualisation des résultats et représentation 3D des poses.
- **LaTeX** : rédaction du compte rendu expérimental.

## Installation

Cloner le dépôt :

```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_DEPOT>
```

Créer un environnement virtuel :

```bash
python3 -m venv .venv
```

Activer l'environnement :

Sur macOS / Linux :

```bash
source .venv/bin/activate
```

Sur Windows :

```bash
.venv\Scripts\activate
```

Installer les dépendances :

```bash
pip install numpy opencv-python matplotlib
```

## Exécution

Les scripts sont exécutés depuis la racine du projet.

**1. Détection des coins :**

```bash
python src/detect_corners.py
```

**2. Calibration de la caméra :**

```bash
python src/calibrate.py
```

**3. Calcul de la longueur focale physique :**

```bash
python src/calculate_focal_length.py
```

**4. Estimation des poses :**

```bash
python src/estimate_pose.py
```

**5. Évaluation des erreurs de reprojection :**

```bash
python src/evaluate_pose.py
```

**6. Correction de la distorsion :**

```bash
python src/undistort.py
```

Chaque script utilise les données et les résultats générés lors des étapes précédentes.

## Organisation du projet

```text
camera_calibration/
│
├── images_png/
│   └── Images originales converties en PNG
│
├── images_original_2040/
│   └── Images redimensionnées à 2040 × 1530
│
├── src/
│   ├── detect_corners.py
│   ├── calibrate.py
│   ├── calculate_focal_length.py
│   ├── estimate_pose.py
│   ├── evaluate_pose.py
│   └── undistort.py
│
├── results/
│   ├── corners_original_2040/
│   ├── poses/
│   ├── image_points_original_2040.npz
│   └── calibration_without_outlier_18.npz
│
├── README.md
└── calibrationFallouDioufVMI2026.tex
```

*Cette arborescence présente les principaux fichiers du projet. Les noms et emplacements des autres fichiers de résultats peuvent varier selon les scripts utilisés.*

## Limites expérimentales

Quelques précautions doivent être prises dans l'interprétation des résultats :

- Le damier a été affiché sur un écran plutôt qu'imprimé. La précision de sa géométrie et de la mesure de ses carrés peut donc introduire des incertitudes.
- Les dimensions physiques du capteur utilisées pour calculer la longueur focale sont des estimations.
- L'erreur de reprojection mesure la cohérence entre les observations et le modèle estimé, mais ne garantit pas à elle seule une précision physique absolue.

## Auteur

**Fallou Diouf**

Master 2 — Vision et Machine Intelligente  
Université Paris Cité

Projet réalisé dans le cadre d'un travail pratique de Vision par Ordinateur.