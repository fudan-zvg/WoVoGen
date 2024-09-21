import numpy as np
from typing import Dict, List, Tuple, Any
import numpy.typing as npt
import pyquaternion
import torch

def gen_dx_bx(xbound, ybound, zbound):
    dx = np.array([row[2] for row in [xbound, ybound, zbound]])
    bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = np.floor(np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]))
    return dx, bx, nx

def get_range(grid_conf: Dict[str, List]) -> Tuple[npt.NDArray]:
    dx, bx, nx = gen_dx_bx(grid_conf['x_bound'],
                                    grid_conf['y_bound'],
                                    grid_conf['z_bound'],)
    dx = dx
    bx = bx
    nx = nx
    pc_range = np.concatenate((bx - dx / 2., bx - dx / 2. + nx * dx))
    return dx, bx, nx, pc_range
