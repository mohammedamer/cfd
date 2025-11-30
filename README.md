# CFD

CFD simulation using PyTorch.

## Install

```bash
git clone https://github.com/mohammedamer/cfd.git

cd cfd
uv sync --frozen
uv run src/cfd/sim.py --help
```

## Example

```bash
uv run src/cfd/sim.py --t 5. --dt 0.05 --gridx 256 --gridy 256
```

![Smoke plume](images/smoke_plume.gif)
