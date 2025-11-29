import logging

import torch
import torch.nn.functional as F

from utils import log_tensor_stats, log

logger = logging.getLogger(__name__)

GRID_X = 128
GRID_Y = 128

dt = 0.01
T = 2.
h = 1.
nu = 1.
g = -9.81

DIFF_ITER = 20


def diffuse(phi, kappa):

    alpha = kappa*dt/h**2
    phi = F.pad(phi, pad=(0, 0, 1, 1, 1, 1), value=0.)

    for _ in range(DIFF_ITER):
        phi[1:-1, 1:-1] = (phi[1:-1, 1:-1] + alpha*(phi[2:, 1:-1] +
                           phi[:-2, 1:-1] +
                           phi[1:-1, 2:]+phi[1:-1, :-2]))/(1+4*alpha)

    return phi[1:-1, 1:-1]


def init():
    u0 = torch.zeros(size=(GRID_X, GRID_Y, 2))
    p = torch.zeros(size=(GRID_X, GRID_Y))

    cellx, celly = torch.meshgrid(torch.arange(
        GRID_X), torch.arange(GRID_Y), indexing="ij")
    cell = torch.stack([cellx, celly], dim=2)

    f = torch.zeros(size=(GRID_X, GRID_Y, 2))
    f[..., 1] = g

    return u0, p, cell, f


def run():

    u, p, cell, f = init()

    log("init")
    log_tensor_stats(u)

    time_steps = torch.arange(0, T+dt, dt)

    for t in time_steps:
        u = u + f*dt
        u = diffuse(u, kappa=nu)

    log("post diff")
    log_tensor_stats(u)


if __name__ == "__main__":
    run()
