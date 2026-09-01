from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import geodesic_draping as gd
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import pyvista as pv
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MESH = APP_DIR / "meshes" / "DemoV5_s.stl"
MODES = ("fast", "hybrid", "complete")
BACKENDS = ("signpost", "integer")
REFINEMENTS = ("none", "flip", "refine")


@dataclass
class Evaluation:
    x: float
    y: float
    angle: float
    value: float


@dataclass
class OptSettings:
    mode: str = "fast"
    backend: str = "signpost"
    refinement: str = "flip"
    percentile: float = 100.0
    maxiter: int = 100
    popsize: int = 6
    powell_refine: bool = False
    custom_objective_path: str = ""
    live_every: int = 1


@dataclass
class OptResult:
    x: np.ndarray
    value: float
    seconds: float
    evaluations: list[Evaluation] = field(default_factory=list)
    drape: gd.DrapeResult | None = None


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = pv.read(path).extract_surface(algorithm="dataset_surface").triangulate()
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape((-1, 4))[:, 1:]
    return np.ascontiguousarray(mesh.points, dtype=np.float64), np.ascontiguousarray(faces)


def mesh_bounds_xy(vertices: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    lo, hi = vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


class SurfaceProjector:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = vertices
        self.tris = vertices[faces]
        self.tree = cKDTree(self.tris[:, :, :2].mean(axis=1))
        self.vertex_tree = cKDTree(vertices[:, :2])

    def project_many(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray([self.project(point) for point in xy], dtype=float)

    def project(self, xy: np.ndarray, k: int = 12) -> np.ndarray:
        _, ids = self.tree.query(xy, k=min(k, len(self.tris)))
        for tri_id in np.atleast_1d(ids):
            bary = barycentric_xy(xy, self.tris[int(tri_id)])
            if bary is not None:
                return bary @ self.tris[int(tri_id)]
        _, vertex_id = self.vertex_tree.query(xy)
        return self.vertices[int(vertex_id)]


def barycentric_xy(p: np.ndarray, tri: np.ndarray) -> np.ndarray | None:
    x, y = p
    x0, y0 = tri[0, :2]
    x1, y1 = tri[1, :2]
    x2, y2 = tri[2, :2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-15:
        return None
    u = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    v = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    bary = np.array([u, v, 1.0 - u - v], dtype=float)
    return bary if np.all(bary >= -1e-10) else None


def load_custom_objective(path: str) -> Callable | None:
    if not path:
        return None
    file = Path(path)
    spec = importlib.util.spec_from_file_location("geodrap_custom_objective", file)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load objective file: {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    objective = getattr(module, "objective", None)
    if not callable(objective):
        raise ValueError(f"{file} must define objective(result, shear, vertices, faces, seed_xy, angle)")
    return objective


class DrapeObjective:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, settings: OptSettings):
        self.vertices = vertices
        self.faces = faces
        self.settings = settings
        self.custom_objective = load_custom_objective(settings.custom_objective_path)
        self.solver = gd.GeoDrapeSolver(vertices, faces, intrinsic_backend=settings.backend, refinement=settings.refinement)
        self.evaluations: list[Evaluation] = []
        self.best_x: np.ndarray | None = None
        self.best_value = float("inf")

    def __call__(self, p: np.ndarray) -> float:
        try:
            result = self.solver.solve(p[:2], float(p[2]), mode=self.settings.mode, retrieval="extrinsic", sample_vertex_shear=True)
            shear = result.vertex_shear if result.vertex_shear is not None else result.face_shear
            value = float(self.custom_objective(result, shear, self.vertices, self.faces, p[:2], float(p[2]))) if self.custom_objective else float(np.nanpercentile(shear, self.settings.percentile))
        except Exception:
            value = 90.0
        self.evaluations.append(Evaluation(float(p[0]), float(p[1]), float(p[2]), value))
        if value < self.best_value:
            self.best_value = value
            self.best_x = np.asarray(p, dtype=float).copy()
        return value

    def best_drape(self) -> gd.DrapeResult | None:
        if self.best_x is None:
            return None
        return self.solver.solve(self.best_x[:2], float(self.best_x[2]), mode=self.settings.mode, retrieval="extrinsic", sample_vertex_shear=True)


def optimize_drape(vertices, faces, bounds, settings: OptSettings, generation=None, live_drape=False, cancel=None) -> OptResult:
    objective = DrapeObjective(vertices, faces, settings)
    generation_count = 0

    def on_generation(xk, convergence=None):
        nonlocal generation_count
        generation_count += 1
        if generation:
            show = live_drape and generation_count % max(1, settings.live_every) == 0
            generation(objective.evaluations.copy(), objective.best_drape() if show else None)
        return bool(cancel and cancel())

    started = perf_counter()
    de = differential_evolution(
        objective,
        bounds=bounds,
        strategy="randtobest1bin",
        popsize=settings.popsize,
        maxiter=settings.maxiter,
        polish=False,
        mutation=(0.6, 1.0),
        recombination=0.8,
        updating="immediate",
        workers=1,
        callback=on_generation,
    )
    x, value = np.asarray(de.x, dtype=float), float(de.fun)
    if settings.powell_refine:
        local = minimize(objective, x0=x, bounds=bounds, method="Powell", options={"maxiter": 20})
        if local.fun < value:
            x, value = np.asarray(local.x, dtype=float), float(local.fun)
    return OptResult(x, value, perf_counter() - started, objective.evaluations, objective.best_drape())


class OptimizerGUI:
    def __init__(self, mesh_path: Path):
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.evaluations: list[Evaluation] = []
        self.best_history: list[float] = []
        self.generator_count = 0
        self.best: OptResult | None = None
        self.error: str | None = None
        self.log: list[str] = []
        self.mesh_path_text = str(mesh_path)
        self.mesh_path = mesh_path
        self.vertices = np.empty((0, 3))
        self.faces = np.empty((0, 3), dtype=np.int64)
        self.mesh = None
        self.params = dict(
            xmin=0.0, xmax=0.0, ymin=0.0, ymax=0.0, amin=0.0, amax=90.0,
            mode="fast", backend="signpost", refinement="flip", percentile=100.0,
            maxiter=100, popsize=6, powell=False, live_drape=True, live_every=1,
            show_evals=True, show_generators=True, show_bounds=True,
            use_custom_objective=False, custom_objective_path="",
        )
        if mesh_path.is_file():
            self.load_mesh(mesh_path)
        else:
            self._log("no default mesh found; enter mesh path and click Load mesh")

    def callback(self) -> None:
        self._drain_queue()
        self._draw_mesh_controls()
        self._draw_optimization_controls()
        self._draw_status()

    def _draw_mesh_controls(self) -> None:
        psim.SeparatorText("Mesh")
        _, self.mesh_path_text = psim.InputText("path", self.mesh_path_text)
        running = self.worker is not None and self.worker.is_alive()
        psim.BeginDisabled(running)
        if psim.Button("Load mesh"):
            self.load_mesh(Path(self.mesh_path_text).expanduser())
        psim.SameLine()
        if psim.Button("Reset bounds to mesh XY"):
            self.reset_bounds()
            self._log("bounds reset to mesh XY")
        psim.EndDisabled()

    def _draw_optimization_controls(self) -> None:
        psim.SeparatorText("Optimization")
        for name, values in (("mode", MODES), ("backend", BACKENDS), ("refinement", REFINEMENTS)):
            changed, idx = psim.Combo(name, values.index(self.params[name]), values)
            if changed:
                self.params[name] = values[idx]

        bounds_changed = False
        if psim.CollapsingHeader("Sampling bounds"):
            for label, lo_key, hi_key in (("x", "xmin", "xmax"), ("y", "ymin", "ymax"), ("angle", "amin", "amax")):
                psim.Text(label); psim.SameLine(); psim.SetNextItemWidth(95.0)
                changed, lo = psim.InputFloat(f"min##{label}", float(self.params[lo_key]), 0.0, 0.0, "%.4f")
                psim.SameLine(); psim.SetNextItemWidth(95.0)
                changed2, hi = psim.InputFloat(f"max##{label}", float(self.params[hi_key]), 0.0, 0.0, "%.4f")
                self.params[lo_key], self.params[hi_key] = lo, hi
                bounds_changed |= changed or changed2
        _, self.params["percentile"] = psim.InputFloat("percentile", float(self.params["percentile"]), 0.0, 0.0, "%.4f")
        _, self.params["use_custom_objective"] = psim.Checkbox("Use custom objective file", bool(self.params["use_custom_objective"]))
        psim.BeginDisabled(not self.params["use_custom_objective"])
        _, self.params["custom_objective_path"] = psim.InputText("objective file", str(self.params["custom_objective_path"]))
        psim.EndDisabled()
        for key in ("maxiter", "popsize", "live_every"):
            _, self.params[key] = psim.InputInt(key, int(self.params[key]))
            self.params[key] = max(1, int(self.params[key]))
        _, self.params["powell"] = psim.Checkbox("Powell refine after DE", bool(self.params["powell"]))
        _, self.params["live_drape"] = psim.Checkbox("Live best shear field", bool(self.params["live_drape"]))
        _, self.params["show_evals"] = psim.Checkbox("Show evaluated seeds", bool(self.params["show_evals"]))
        _, self.params["show_generators"] = psim.Checkbox("Show generators", bool(self.params["show_generators"]))
        changed, self.params["show_bounds"] = psim.Checkbox("Show sampling boundary", bool(self.params["show_bounds"]))
        if bounds_changed or changed:
            self._update_bounds_box()

        running = self.worker is not None and self.worker.is_alive()
        psim.BeginDisabled(running)
        if psim.Button("Start optimization"):
            self.start()
        psim.EndDisabled(); psim.SameLine(); psim.BeginDisabled(not running)
        if psim.Button("Stop after generation"):
            self.stop_event.set(); self._log("stop requested")
        psim.EndDisabled(); psim.SameLine(); psim.BeginDisabled(self.best is None and not self.evaluations)
        if psim.Button("Save result"):
            self.save_result()
        psim.EndDisabled()

    def _draw_status(self) -> None:
        psim.SeparatorText("Status")
        psim.Text(f"Mesh: {self.mesh_path.name} ({len(self.vertices)} V, {len(self.faces)} F)")
        psim.Text(f"Evaluations: {len(self.evaluations)}")
        if self.evaluations:
            psim.Text(f"Current best shear: {min(e.value for e in self.evaluations):.4f} deg")
        if self.best:
            x = self.best.x
            psim.Text(f"Final: {self.best.value:.4f} deg at x={x[0]:.4f}, y={x[1]:.4f}, angle={x[2]:.4f}")
            psim.Text(f"Time: {self.best.seconds:.2f}s")
        if self.error:
            psim.TextColored((1.0, 0.2, 0.2, 1.0), self.error)
        if self.best_history and psim.CollapsingHeader("Best history"):
            values = np.asarray(self.best_history, dtype=np.float32)
            psim.PlotLines("best shear", values, 0, None, float(values.min()), float(values.max()), (0.0, 120.0))
        if psim.CollapsingHeader("Log"):
            psim.BeginChild("log", (0.0, 160.0), True)
            for line in self.log[-200:]:
                psim.TextWrapped(line)
            psim.EndChild()
            if psim.SmallButton("Clear log"):
                self.log.clear()

    def load_mesh(self, mesh_path: Path) -> None:
        try:
            self.vertices, self.faces = load_mesh(str(mesh_path))
        except Exception as exc:
            self.error = f"mesh load failed: {exc}"; self._log(self.error); return
        self.mesh_path = mesh_path
        self.mesh_path_text = str(mesh_path)
        ps.remove_all_structures()
        self.mesh = ps.register_surface_mesh("mesh", self.vertices, self.faces)
        self.projector = SurfaceProjector(self.vertices, self.faces)
        self.evaluations.clear(); self.best_history.clear(); self.best = None; self.generator_count = 0
        self.reset_bounds()
        self._log(f"loaded {mesh_path.name} ({len(self.vertices)} V, {len(self.faces)} F)")

    def reset_bounds(self) -> None:
        (xmin, xmax), (ymin, ymax) = mesh_bounds_xy(self.vertices)
        self.params.update({"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax})
        self._update_bounds_box()

    def validate(self) -> bool:
        checks = (
            (self.params["xmin"] < self.params["xmax"], "xmin must be < xmax"),
            (self.params["ymin"] < self.params["ymax"], "ymin must be < ymax"),
            (self.params["amin"] <= self.params["amax"], "angle min must be <= angle max"),
            (0.0 <= self.params["percentile"] <= 100.0, "percentile must be in [0, 100]"),
            (not self.params["use_custom_objective"] or Path(str(self.params["custom_objective_path"])).is_file(), "custom objective file not found"),
        )
        for ok, message in checks:
            if not ok:
                self.error = message; self._log(f"validation: {message}"); return False
        return True

    def start(self) -> None:
        if not self.validate():
            return
        self.error = None; self.best = None; self.evaluations = []; self.best_history = []
        self.stop_event.clear(); self._clear_dynamic(); self._update_bounds_box()
        settings = OptSettings(
            mode=self.params["mode"], backend=self.params["backend"], refinement=self.params["refinement"],
            percentile=float(self.params["percentile"]), maxiter=int(self.params["maxiter"]), popsize=int(self.params["popsize"]),
            powell_refine=bool(self.params["powell"]), live_every=int(self.params["live_every"]),
            custom_objective_path=str(self.params["custom_objective_path"]) if self.params["use_custom_objective"] else "",
        )
        bounds = [(float(self.params["xmin"]), float(self.params["xmax"])), (float(self.params["ymin"]), float(self.params["ymax"])), (float(self.params["amin"]), float(self.params["amax"]))]
        self._log(f"start: mode={settings.mode}, objective={'custom' if settings.custom_objective_path else 'percentile'}, popsize={settings.popsize}, maxiter={settings.maxiter}")
        self.worker = threading.Thread(target=self._run, args=(bounds, settings), daemon=True)
        self.worker.start()

    def _run(self, bounds, settings: OptSettings) -> None:
        try:
            result = optimize_drape(self.vertices, self.faces, bounds, settings, generation=lambda e, d: self.queue.put(("generation", (e, d))), live_drape=bool(self.params["live_drape"]), cancel=self.stop_event.is_set)
            self.queue.put(("done", result))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _drain_queue(self) -> None:
        dirty = False
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "generation":
                self.evaluations, drape = payload
                self._refresh_best_history(); self._update_evaluations()
                if drape is not None: self._update_drape(drape)
                if self.evaluations: self._log(f"generation: {len(self.evaluations)} evals, best {self.best_history[-1]:.4f} deg")
                dirty = True
            elif kind == "done":
                self.best = payload; self.evaluations = self.best.evaluations
                self._refresh_best_history(); self._update_evaluations()
                if self.best.drape is not None: self._update_drape(self.best.drape)
                self._log(f"done: {self.best.value:.4f} deg; {len(self.best.evaluations)} evals in {self.best.seconds:.2f}s")
                dirty = True
            elif kind == "error":
                self.error = str(payload); self._log(f"error: {self.error}")
        if dirty:
            ps.request_redraw()

    def _refresh_best_history(self) -> None:
        best = float("inf"); self.best_history = []
        for ev in self.evaluations:
            best = min(best, ev.value); self.best_history.append(best)

    def _update_evaluations(self) -> None:
        ps.remove_point_cloud("evaluated seeds", error_if_absent=False); ps.remove_point_cloud("best seed", error_if_absent=False)
        if not self.evaluations: return
        if self.params["show_evals"]:
            points = self.projector.project_many(np.array([[e.x, e.y] for e in self.evaluations], dtype=float))
            values = np.array([e.value for e in self.evaluations], dtype=float)
            cloud = ps.register_point_cloud("evaluated seeds", points, radius=0.004)
            cloud.add_scalar_quantity("shear", values, enabled=True, cmap="viridis")
        best = min(self.evaluations, key=lambda e: e.value)
        ps.register_point_cloud("best seed", self.projector.project(np.array([best.x, best.y])).reshape(1, 3), radius=0.01, color=(1, 0, 0))

    def _update_drape(self, result: gd.DrapeResult) -> None:
        if result.vertex_shear is not None:
            self.mesh.add_scalar_quantity("live best vertex shear", result.vertex_shear, defined_on="vertices", enabled=True, cmap="jet")
        if self.params["show_generators"]:
            self._update_generators(result)

    def _update_generators(self, result: gd.DrapeResult, radius: float = 0.001) -> None:
        lines = [np.asarray(line, dtype=float) for family in result.generators for line in family]
        for i in range(max(self.generator_count, len(lines))):
            ps.remove_curve_network(f"generator {i}", error_if_absent=False); ps.remove_point_cloud(f"generator {i} points", error_if_absent=False)
        for i, points in enumerate(lines):
            if len(points): ps.register_point_cloud(f"generator {i} points", points, radius=radius, color=(1, 0, 0))
            if len(points) >= 2:
                edges = np.column_stack((np.arange(len(points) - 1), np.arange(1, len(points))))
                ps.register_curve_network(f"generator {i}", points, edges, radius=radius, color=(1, 1, 1))
        self.generator_count = len(lines)

    def _update_bounds_box(self) -> None:
        ps.remove_curve_network("sampling boundary", error_if_absent=False)
        if not self.params["show_bounds"] or len(self.vertices) == 0: return
        xmin, xmax, ymin, ymax = map(float, (self.params["xmin"], self.params["xmax"], self.params["ymin"], self.params["ymax"]))
        z = float(self.vertices[:, 2].max()) + 0.01 * float(np.ptp(self.vertices[:, 2]) or 1.0)
        points = np.array([[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z]], dtype=float)
        ps.register_curve_network("sampling boundary", points, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), radius=0.0015, color=(0.1, 0.7, 1.0))

    def _clear_dynamic(self) -> None:
        self.mesh.remove_quantity("live best vertex shear", error_if_absent=False)
        for name in ("evaluated seeds", "best seed"):
            ps.remove_point_cloud(name, error_if_absent=False)
        for i in range(self.generator_count):
            ps.remove_curve_network(f"generator {i}", error_if_absent=False); ps.remove_point_cloud(f"generator {i} points", error_if_absent=False)
        self.generator_count = 0

    def save_result(self) -> None:
        out = Path("results") / datetime.now().strftime("opt_%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        with (out / "evaluations.csv").open("w", newline="") as f:
            writer = csv.writer(f); writer.writerow(["x", "y", "angle", "value"])
            writer.writerows((e.x, e.y, e.angle, e.value) for e in self.evaluations)
        meta = {"mesh": str(self.mesh_path), "params": self.params, "best": None if self.best is None else {"x": self.best.x.tolist(), "value": self.best.value, "seconds": self.best.seconds}}
        (out / "summary.json").write_text(json.dumps(meta, indent=2))
        self._log(f"saved result to {out}")

    def _log(self, msg: str) -> None:
        self.log.append(msg); print(msg)


def run(mesh_path: Path) -> None:
    ps.init()
    ps.set_ground_plane_mode("none")
    gui = OptimizerGUI(mesh_path)
    ps.set_user_callback(gui.callback)
    ps.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Polyscope GUI geodesic draping optimizer")
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    args = parser.parse_args()
    run(args.mesh.resolve())


if __name__ == "__main__":
    main()
