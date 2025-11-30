from celluloid import Camera
import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm
from rich import print


def log(msg):
    print(msg)


def log_tensor_stats(tensor):
    log(f"{tensor.max()=} {tensor.min()=}")


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
