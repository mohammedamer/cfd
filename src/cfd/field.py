from pathlib import Path

import torch
import typer

from utils import log


def set_field(gridx: int = typer.Option(...), gridy: int = typer.Option(...),
              x: float = typer.Option(..., "-x"),
              name: str = typer.Option(...), root: str = None,
              y: float = typer.Option(None, "-y"),
              xrange: tuple[int, int] = None,
              yrange: tuple[int, int] = None,
              constant: float = 0.):

    if root is None:
        root = "."

    if xrange is None:
        xrange = (None, None)

    if yrange is None:
        yrange = (None, None)

    root = Path(root)

    dim = 2
    if y is None:
        dim = 1

    field = torch.full(size=(gridx, gridy, dim), fill_value=constant)

    def set_component(field, component_idx, component):
        field[xrange[0]:xrange[1],
              yrange[0]:yrange[1], component_idx] = component

    set_component(field, 0, component=x)

    if y is not None:
        set_component(field, 1, component=y)

    field = field.squeeze()

    torch.save(field, root / f"{name}.pt")

    log("Done!")


if __name__ == "__main__":
    typer.run(set_field)
