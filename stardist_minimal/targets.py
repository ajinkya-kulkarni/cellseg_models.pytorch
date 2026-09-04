# Radial-distance target generation adapted from StarDist (BSD-3-Clause).
import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt


@njit
def gen_stardist_maps(inst_map: np.ndarray, n_rays: int) -> np.ndarray:
    dist = np.empty(inst_map.shape + (n_rays,), np.float32)
    step = np.float32(2 * np.pi / n_rays)

    for i in range(inst_map.shape[0]):
        for j in range(inst_map.shape[1]):
            label = inst_map[i, j]
            if label == 0:
                dist[i, j] = 0
                continue

            for k in range(n_rays):
                angle = np.float32(k * step)
                dy, dx = np.cos(angle), np.sin(angle)
                x = np.float32(0)
                y = np.float32(0)
                while True:
                    x += dx
                    y += dy
                    ii = int(round(i + x))
                    jj = int(round(j + y))
                    if (
                        ii < 0
                        or ii >= inst_map.shape[0]
                        or jj < 0
                        or jj >= inst_map.shape[1]
                        or inst_map[ii, jj] != label
                    ):
                        correction = 1 - 0.5 / max(np.abs(dx), np.abs(dy))
                        x -= correction * dx
                        y -= correction * dy
                        dist[i, j, k] = np.sqrt(x * x + y * y)
                        break

    return dist.transpose(2, 0, 1)


def gen_dist_map(inst_map: np.ndarray) -> np.ndarray:
    """Normalized Euclidean distance transform computed independently per instance."""
    inst_map = np.asarray(inst_map)
    out = np.zeros(inst_map.shape, dtype=np.float32)
    for label in np.unique(inst_map):
        if label == 0:
            continue
        ys, xs = np.where(inst_map == label)
        if ys.size == 0:
            continue
        y0, y1 = max(0, int(ys.min()) - 1), min(inst_map.shape[0], int(ys.max()) + 2)
        x0, x1 = max(0, int(xs.min()) - 1), min(inst_map.shape[1], int(xs.max()) + 2)
        obj = inst_map[y0:y1, x0:x1] == label
        d = distance_transform_edt(obj).astype(np.float32)
        maximum = float(d.max())
        if maximum > 0:
            d /= maximum
        patch = out[y0:y1, x0:x1]
        patch[obj] = d[obj]
    return out
