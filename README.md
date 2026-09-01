# geodesic_draping_optimizer

Small Polyscope demo GUI showing how `geodesic_draping` can be used to optimize drape seed position `(x, y)` and fabric angle.

## Install

This demo was tested with Python 3.11.

Install GUI dependencies:

```powershell
python -m pip install numpy scipy pyvista polyscope
```

Use a Python environment with `geodesic_draping` installed. For example installed via a compatible `geodesic_draping` wheel:

```powershell
python -m pip install geodesic-draping numpy scipy pyvista polyscope
```

Optional `uv` setup:

```powershell
uv sync
uv pip install geodesic-draping numpy scipy pyvista polyscope
```

## Run

```powershell
python optimizer_gui.py
```

With mesh:

```powershell
python optimizer_gui.py --mesh path\to\part.stl
```

With `uv`:

```powershell
uv run python optimizer_gui.py
```

## Custom objective

Optional objective file must define:

```python
def objective(result, shear, vertices, faces, seed_xy, angle):
    return float(...)
```

Example: `objective_locking_weighted.py`.
