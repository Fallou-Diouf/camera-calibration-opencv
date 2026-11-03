
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 0. Chargement des poses estimées
# --------------------------------------------------

data = np.load("results/poses/camera_poses.npz")

image_names = data["image_names"]
rotations = data["rotations"]
camera_positions = data["camera_positions"]


# Chargement des poses
data = np.load("results/poses/camera_poses.npz")

rotations = data["rotations"]
camera_positions = data["camera_positions"].reshape(-1, 3)

# Dimensions du damier : étendue des coins intérieurs
board_width = 8 * 15.0
board_height = 5 * 15.0

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# --------------------------------------------------
# 1. Damier dans le plan Z = 0
# --------------------------------------------------

board = np.array([
    [0, 0, 0],
    [board_width, 0, 0],
    [board_width, board_height, 0],
    [0, board_height, 0],
    [0, 0, 0]
])

ax.plot(
    board[:, 0], board[:, 1], board[:, 2],
    color="black", linewidth=2.5, label="Plan du damier"
)

# --------------------------------------------------
# 2. Positions des caméras
# --------------------------------------------------

ax.scatter(
    camera_positions[:, 0],
    camera_positions[:, 1],
    camera_positions[:, 2],
    color="tab:blue",
    s=30,
    label="Positions des caméras"
)

# --------------------------------------------------
# 3. Repères locaux des caméras
# --------------------------------------------------

axis_length = 35.0

axis_labels = [
    ("Axe X caméra", "red"),
    ("Axe Y caméra", "green"),
    ("Axe Z caméra", "blue")
]

for i, (R, C) in enumerate(zip(rotations, camera_positions)):

    # Axes de la caméra exprimés dans le repère du damier
    camera_axes = R.T

    for j, (label, color) in enumerate(axis_labels):

        direction = camera_axes[:, j]

        ax.quiver(
            C[0], C[1], C[2],
            direction[0], direction[1], direction[2],
            length=axis_length,
            normalize=True,
            color=color,
            arrow_length_ratio=0.18,
            linewidth=1.3,
            label=label if i == 0 else None
        )

    # Numérotation discrète des positions
    ax.text(
        C[0] + 5,
        C[1] + 5,
        C[2] + 5,
        str(i + 1),
        fontsize=7
    )

# --------------------------------------------------
# 4. Mise en forme
# --------------------------------------------------

ax.set_title(
    "Estimation des 18 poses de la caméra",
    fontsize=14,
    pad=20
)

ax.set_xlabel("X (mm)", labelpad=10)
ax.set_ylabel("Y (mm)", labelpad=10)
ax.set_zlabel("Z (mm)", labelpad=10)

ax.legend(loc="upper right", fontsize=9)
ax.grid(True)

# Même échelle sur les trois axes
all_points = np.vstack([camera_positions, board])
center = all_points.mean(axis=0)
max_range = np.ptp(all_points, axis=0).max() / 2

ax.set_xlim(center[0] - max_range, center[0] + max_range)
ax.set_ylim(center[1] - max_range, center[1] + max_range)
ax.set_zlim(center[2] - max_range, center[2] + max_range)

ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=25, azim=-60)

# --------------------------------------------------
# 5. Enregistrement
# --------------------------------------------------

output_dir = Path("results/poses")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "camera_poses_3d.png"

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Figure enregistrée : {output_path}")

plt.show()
