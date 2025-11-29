import logging

import torch
import torch.nn.functional as F
from torch import Tensor

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
DIV_FREE_ITER = 20


def div(u):

    u = F.pad(u, pad=(0, 0, 1, 1, 1, 1), value=0.)
    return (u[2:, 1:-1, 0]-u[:-2, 1:-1, 0])/(2*h) +\
        (u[1:-1, 2:, 1] - u[1:-1, :-2, 1])/(2*h)


def div_free(u, p):

    u_div = div(u)

    p = F.pad(p, pad=(1, 1, 1, 1), value=0.)
    for _ in range(DIV_FREE_ITER):
        p[1:-1, 1:-1] = (p[2:, 1:-1] + p[:-2, 1:-1] +
                         p[1:-1, 2:] + p[1:-1, :-2]-h**2*1/dt*u_div)/4.

    u[..., 0] = u[..., 0] - dt*(p[2:, 1:-1]-p[:-2, 1:-1])/(2*h)
    u[..., 1] = u[..., 1] - dt*(p[1:-1, 2:]-p[1:-1, :-2])/(2*h)

    p = p[1:-1, 1:-1]
    return u, p


def diffuse(phi, kappa):

    alpha = kappa*dt/h**2
    phi = F.pad(phi, pad=(0, 0, 1, 1, 1, 1), value=0.)

    for _ in range(DIFF_ITER):
        phi[1:-1, 1:-1] = (phi[1:-1, 1:-1] + alpha*(phi[2:, 1:-1] +
                           phi[:-2, 1:-1] +
                           phi[1:-1, 2:]+phi[1:-1, :-2]))/(1+4*alpha)

    return phi[1:-1, 1:-1]


def semi_largangian(phi: Tensor, u: Tensor, cell: Tensor):

    cell = cell - u*dt
    cell = cell.round().long()

    inside = (cell[..., 0] >= 0) & (cell[..., 0] < GRID_X) & \
             (cell[..., 1] >= 0) & (cell[..., 1] < GRID_Y)

    x = cell[..., 0]
    y = cell[..., 1]

    advect = torch.zeros_like(phi)

    advect[inside] = phi[x[inside], y[inside]]

    return advect


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
        u, p = div_free(u, p)
        u = semi_largangian(u, u, cell=cell)
        u, p = div_free(u, p)

    log("post diff")
    log_tensor_stats(u)


if __name__ == "__main__":
    run()
