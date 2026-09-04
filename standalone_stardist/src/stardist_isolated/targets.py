"""
StarDist radial-distance target generation.

Adapted from the StarDist project:
https://github.com/stardist/stardist

BSD 3-Clause License

Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
All rights reserved.
"""

import numpy as np
from numba import njit


@njit
def gen_stardist_maps(inst_map: np.ndarray, n_rays: int) -> np.ndarray:
    """Compute radial distances for each non-zero pixel in an instance mask."""
    n_rays = int(n_rays)
    if n_rays <= 0:
        raise ValueError("n_rays must be > 0")

    dist = np.empty(inst_map.shape + (n_rays,), np.float32)
    st_rays = np.float32((2 * np.pi) / n_rays)

    for i in range(inst_map.shape[0]):
        for j in range(inst_map.shape[1]):
            value = inst_map[i, j]
            if value == 0:
                dist[i, j] = 0
            else:
                for k in range(n_rays):
                    phi = np.float32(k * st_rays)
                    dy = np.cos(phi)
                    dx = np.sin(phi)
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
                            or value != inst_map[ii, jj]
                        ):
                            correction = 1 - 0.5 / max(np.abs(dx), np.abs(dy))
                            x -= correction * dx
                            y -= correction * dy
                            dist[i, j, k] = np.sqrt(x**2 + y**2)
                            break

    return dist.transpose(2, 0, 1)
