from celluloid import Camera
import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm
from rich import print


def log(msg):
    print(msg)


def log_tensor_stats(tensor):
    log(f"{tensor.max()=} {tensor.min()=}")


def save_gif(dye_sol: Tensor, gif_path):

    fig = plt.figure()
    ax = plt.subplot()
    ax.set_axis_off()

    camera = Camera(fig)

    for time_step in tqdm(dye_sol):
        time_step = time_step.transpose(1, 0)
        ax.imshow(time_step, cmap="inferno", origin="lower")
        fig.tight_layout()
        camera.snap()

    anim = camera.animate()
    anim.save(gif_path, writer="pillow", fps=30)
