import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from celluloid import Camera
import matplotlib.pyplot as plt

from utils import log

GRID_X = 128
GRID_Y = 128

dt = 0.01
T = 2.
h = 1.
nu = 10.
Fy = 1000.
D = 10.

REACTION_LAMBDA = 0.

DIFF_ITER = 20
DIV_FREE_ITER = 20

DYE_SQUARE_SIDE = 32
DYE_VAL = 1.


def save_gif(dye_sol: Tensor):

    fig = plt.figure()
    ax = plt.subplot()

    camera = Camera(fig)

    for time_step in tqdm(dye_sol):
        time_step = time_step.transpose(1, 0)
        ax.imshow(time_step, cmap="inferno", origin="lower")
        camera.snap()

    anim = camera.animate()

    gif_path = "output/sim.gif"
    anim.save(gif_path, writer="pillow", fps=30)


def pressure_boundary(p):

    p[0:1] = p[1:2]
    p[GRID_X-2:GRID_X-1] = p[GRID_X-3:GRID_X-2]

    p[:, 0:1] = p[:, 1:2]
    p[:, GRID_Y-2:GRID_Y-1] = p[:, GRID_Y-3:GRID_Y-2]


def velocity_boundary(u):

    u[0, :, 0] = 0
    u[GRID_X-1, :, 0] = 0

    u[:, 0, 1] = 0
    u[:, GRID_Y-1, 1] = 0


def div(u):

    u = F.pad(u, pad=(0, 0, 1, 1, 1, 1), value=0.)
    return (u[2:, 1:-1, 0]-u[:-2, 1:-1, 0])/(2*h) +\
        (u[1:-1, 2:, 1] - u[1:-1, :-2, 1])/(2*h)


def div_free(u, p):

    u_div = div(u)

    p = F.pad(p, pad=(1, 1, 1, 1), value=0.)
    for _ in tqdm(range(DIV_FREE_ITER)):
        p[1:-1, 1:-1] = (p[2:, 1:-1] + p[:-2, 1:-1] +
                         p[1:-1, 2:] + p[1:-1, :-2]-h**2*1/dt*u_div)/4.

    u[..., 0] = u[..., 0] - dt*(p[2:, 1:-1]-p[:-2, 1:-1])/(2*h)
    u[..., 1] = u[..., 1] - dt*(p[1:-1, 2:]-p[1:-1, :-2])/(2*h)

    p = p[1:-1, 1:-1]
    return u, p


def diffuse(phi: Tensor, kappa):

    alpha = kappa*dt/h**2

    dim_count = phi.dim()
    if dim_count == 3:
        pad = (0, 0, 1, 1, 1, 1)
    elif dim_count == 2:
        pad = (1, 1, 1, 1)

    phi = F.pad(phi, pad=pad, value=0.)

    for _ in tqdm(range(DIFF_ITER)):
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
    f[..., 1] = Fy

    dye = torch.zeros_like(p)

    dye_s = torch.zeros_like(p)

    centerx = GRID_X // 2
    centery = GRID_Y // 2

    dye_s[centerx-DYE_SQUARE_SIDE//2:centerx+DYE_SQUARE_SIDE//2,
          centery-DYE_SQUARE_SIDE//2:centery+DYE_SQUARE_SIDE//2] = 1.

    return u0, p, cell, f, dye, dye_s


def run():

    u, p, cell, f, dye, dye_s = init()

    dye_sol = [dye]

    time_steps = torch.arange(0, T+dt, dt)

    for _ in tqdm(time_steps):
        u = u + f*dt
        u = diffuse(u, kappa=nu)
        u, p = div_free(u, p)
        u = semi_largangian(u, u, cell=cell)
        u, p = div_free(u, p)
        dye = dye + dye_s*dt
        dye = semi_largangian(dye, u, cell=cell)
        dye = diffuse(dye, kappa=D)
        dye = dye / (1+REACTION_LAMBDA*dt)

        velocity_boundary(u)
        pressure_boundary(p)
        pressure_boundary(dye)

        dye_sol.append(dye)

    dye_sol = torch.stack(dye_sol, dim=0)
    save_gif(dye_sol)

    log("Done!")


if __name__ == "__main__":
    run()
