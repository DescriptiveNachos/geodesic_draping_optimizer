# geodesic_draping_optimizer

Small Polyscope demo GUI showing how `geodesic_draping` can be used to optimize drape seed position `(x, y)` and fabric angle. Includes a demo STL mesh.

## Install

This demo was tested with Python 3.11.

```powershell
python -m pip install geodesic-draping numpy scipy pyvista polyscope
```

Optional `uv` setup:

```powershell
uv venv
uv pip install geodesic-draping numpy scipy pyvista polyscope
```

## Run

```powershell
python optimizer_gui.py
```

By default this loads `meshes/DemoV5_s.stl`.

With another mesh:

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
