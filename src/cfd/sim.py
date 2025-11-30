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
nu = 1.
Fy = 500.
D = 1.

REACTION_LAMBDA = 0.

DIFF_ITER = 20
DIV_FREE_ITER = 20

DYE_RADIUS = 4
DYE_VAL = 1.
DYE_S = 0.01


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


def neumann_boundary(p):

    p[0:1] = p[1:2]
    p[GRID_X-2:GRID_X-1] = p[GRID_X-3:GRID_X-2]

    p[:, 0:1] = p[:, 1:2]
    p[:, GRID_Y-2:GRID_Y-1] = p[:, GRID_Y-3:GRID_Y-2]


def impermeable_boundary(u):

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

    if phi.dim() < 3:
        phi = phi[..., None]

    cell = cell - u*dt
    q11 = cell.floor().long()

    q12 = q11.clone()
    q12[:, :, 1] += 1

    q21 = q11.clone()
    q21[:, :, 0] += 1

    q22 = q21.clone()
    q22[:, :, 1] += 1

    def query(phi, q):
        inside = (q[..., 0] >= 0) & (q[..., 0] < GRID_X) & \
            (q[..., 1] >= 0) & (q[..., 1] < GRID_Y)

        x = q[..., 0]
        y = q[..., 1]

        advect = torch.zeros_like(phi)
        advect[inside] = phi[x[inside], y[inside]]
        return advect

    # N, N
    fq11 = query(phi, q11)
    fq12 = query(phi, q12)
    fq21 = query(phi, q21)
    fq22 = query(phi, q22)

    # N, N
    x = cell[..., 0]
    x1 = q11[..., 0]
    x2 = q21[..., 0]

    y = cell[..., 1]
    y1 = q11[..., 1]
    y2 = q12[..., 1]

    x2_x1 = x2-x1
    y2_y1 = y2-y1

    x2_x1_y2_y1 = (x2_x1 * y2_y1)
    factor = 1/(x2_x1_y2_y1+1e-7)
    factor = factor[..., None, None, None]

    x2_x = x2 - x
    x_x1 = x-x1
    y2_y = y2 - y
    y_y1 = y-y1

    X = torch.stack([x2_x, x_x1], dim=2)
    X = X.reshape(X.shape[0], X.shape[1], 1, 1, X.shape[2])
    Y = torch.stack([y2_y, y_y1], dim=2)
    Y = Y.reshape(Y.shape[0], Y.shape[1], 1, Y.shape[2], 1)
    M = torch.stack([fq11, fq12, fq21, fq22], dim=-1)
    M = M.reshape(*M.shape[:-1], 2, 2)

    advect = factor * X @ M @ Y
    return advect.squeeze()


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

    mask = (cell[..., 0]-centerx)**2 + \
        (cell[..., 1] - centery)**2 <= DYE_RADIUS**2
    dye_s[mask] = DYE_S

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

        impermeable_boundary(u)
        neumann_boundary(p)
        neumann_boundary(dye)

        dye_sol.append(dye)

    dye_sol = torch.stack(dye_sol, dim=0)
    save_gif(dye_sol)

    log("Done!")


if __name__ == "__main__":
    run()
