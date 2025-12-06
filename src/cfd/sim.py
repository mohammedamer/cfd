import torch
import torch.nn.functional as F
from torch import Tensor
from rich.progress import track
import hydra

from utils import log, save_gif


h = 1.


def neumann_boundary(p: Tensor):

    gridx, gridy = p.shape[:2]

    p[0:1] = p[1:2]
    p[gridx-2:gridy-1] = p[gridx-3:gridy-2]

    p[:, 0:1] = p[:, 1:2]
    p[:, gridy-2:gridy-1] = p[:, gridy-3:gridy-2]


def impermeable_boundary(u: Tensor):

    gridx, gridy = u.shape[:2]

    u[0, :, 0] = 0
    u[gridx-1, :, 0] = 0

    u[:, 0, 1] = 0
    u[:, gridy-1, 1] = 0


def div(u):

    u = F.pad(u, pad=(0, 0, 1, 1, 1, 1), value=0.)
    return (u[2:, 1:-1, 0]-u[:-2, 1:-1, 0])/(2*h) +\
        (u[1:-1, 2:, 1] - u[1:-1, :-2, 1])/(2*h)


def div_free(u, p, diff_iter, dt):

    u_div = div(u)

    p = F.pad(p, pad=(1, 1, 1, 1), value=0.)
    for _ in track(range(diff_iter), description="div free"):
        p[1:-1, 1:-1] = (p[2:, 1:-1] + p[:-2, 1:-1] +
                         p[1:-1, 2:] + p[1:-1, :-2]-h**2*1/dt*u_div)/4.

    u[..., 0] = u[..., 0] - dt*(p[2:, 1:-1]-p[:-2, 1:-1])/(2*h)
    u[..., 1] = u[..., 1] - dt*(p[1:-1, 2:]-p[1:-1, :-2])/(2*h)

    p = p[1:-1, 1:-1]
    return u, p


def diffuse(phi: Tensor, kappa, dt, diff_iter):

    alpha = kappa*dt/h**2

    dim_count = phi.dim()
    if dim_count == 3:
        pad = (0, 0, 1, 1, 1, 1)
    elif dim_count == 2:
        pad = (1, 1, 1, 1)

    phi = F.pad(phi, pad=pad, value=0.)

    for _ in track(range(diff_iter), description="diffusion"):
        phi[1:-1, 1:-1] = (phi[1:-1, 1:-1] + alpha*(phi[2:, 1:-1] +
                           phi[:-2, 1:-1] +
                           phi[1:-1, 2:]+phi[1:-1, :-2]))/(1+4*alpha)

    return phi[1:-1, 1:-1]


def semi_largangian(phi: Tensor, u: Tensor, cell: Tensor, dt: float):

    gridx, gridy = cell.shape[:2]

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
        inside = (q[..., 0] >= 0) & (q[..., 0] < gridx) & \
            (q[..., 1] >= 0) & (q[..., 1] < gridy)

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


def init(gridx, gridy, Fy, dye_radius, dye_source):
    u0 = torch.zeros(size=(gridx, gridy, 2))
    p = torch.zeros(size=(gridx, gridy))

    cellx, celly = torch.meshgrid(torch.arange(
        gridx), torch.arange(gridy), indexing="ij")
    cell = torch.stack([cellx, celly], dim=2)

    f = torch.zeros(size=(gridx, gridy, 2))
    f[..., 1] = Fy

    dye = torch.zeros_like(p)

    dye_s = torch.zeros_like(p)

    centerx = gridx // 2
    centery = gridy // 2

    mask = (cell[..., 0]-centerx)**2 + \
        (cell[..., 1] - centery)**2 <= dye_radius**2
    dye_s[mask] = dye_source

    return u0, p, cell, f, dye, dye_s


# def run(T: float = 2., dt: float = 0.01, nu: float = 1.,
#         reaction_lambda: float = 0., dye_diff: float = 1.,
#         gridx: int = 128, gridy: int = 128, Fy: float = 500.,
#         dye_radius: float = 4., dye_src: float = 1., diff_iter: int = 20,):

@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg):

    u, p, cell, f, dye, dye_s = init(
        cfg.grid.x, cfg.grid.y, cfg.f.y, cfg.dye.radius, cfg.dye.src)

    dye_sol = [dye]

    dt = cfg.run.dt
    time_steps = torch.arange(0, cfg.run.T+dt, dt)

    diff_iter = cfg.run.diff_iter
    dye_diff = cfg.dye.D
    reaction_lambda = cfg.dye.react_lambda

    for _ in track(time_steps, description="cfd"):
        u = u + f*dye[..., None]*dt

        u = diffuse(u, kappa=cfg.u.nu, dt=dt, diff_iter=diff_iter)
        u, p = div_free(u, p, dt=dt, diff_iter=diff_iter)
        u = semi_largangian(u, u, cell=cell, dt=dt)
        u, p = div_free(u, p, dt=dt, diff_iter=diff_iter)
        dye = dye + dye_s*dt
        dye = semi_largangian(dye, u, cell=cell, dt=dt)
        dye = diffuse(dye, kappa=dye_diff, dt=dt, diff_iter=diff_iter)
        dye = dye / (1+reaction_lambda*dt)

        impermeable_boundary(u)
        neumann_boundary(p)
        neumann_boundary(dye)

        dye_sol.append(dye)

    dye_sol = torch.stack(dye_sol, dim=0)
    save_gif(dye_sol, gif_path=cfg.log.gif_path)

    log("Done!")


if __name__ == "__main__":
    run()
