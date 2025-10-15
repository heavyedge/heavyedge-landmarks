import matplotlib.pyplot as plt
import numpy as np
from heavyedge import ProfileData, get_sample_path

from heavyedge_landmarks import (
    landmarks_type2,
    landmarks_type3,
    plateau_type2,
    plateau_type3,
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    x = data.x()
    Ys, Ls, _ = data[:]
lm = landmarks_type2(x, Ys, Ls, 32)
peaks, knees = lm[:, 0, 1:].T
plateau = plateau_type2(x, Ys, peaks, knees)
plateau_x = np.stack([plateau[:, 2], np.zeros(len(plateau))])
plateau_y = plateau[:, 0] + plateau_x * plateau[:, 1]
plateau = np.stack([plateau_x, plateau_y]).transpose(2, 0, 1)
lm = np.concatenate([lm[..., :-1], plateau], axis=-1)

axes[0].plot(x, Ys.T, color="gray")
axes[0].plot(*lm.transpose(1, 2, 0))

with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    x = data.x()
    Ys, Ls, _ = data[:]
lm = landmarks_type3(x, Ys, Ls, 32)
troughs, knees = lm[:, 0, 2:].T
plateau = plateau_type3(x, Ys, troughs, knees)
plateau_x = np.stack([plateau[:, 2], np.zeros(len(plateau))])
plateau_y = plateau[:, 0] + plateau_x * plateau[:, 1]
plateau = np.stack([plateau_x, plateau_y]).transpose(2, 0, 1)
lm = np.concatenate([lm[..., :-1], plateau], axis=-1)

axes[1].plot(x, Ys.T, color="gray")
axes[1].plot(*lm.transpose(1, 2, 0))

axes[0].axis("off")
axes[1].axis("off")
fig.tight_layout()
