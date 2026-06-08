from __future__ import annotations

from collections import defaultdict
import heapq
import json
from pathlib import Path

import numpy as np


VTK_FILE = Path(__file__).with_name("lc_3.vtk")
OUT_HTML = Path(__file__).with_name("lc_3_interactive_velocity_viewer.html")


def read_legacy_vtk_polydata(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    i = 0
    points = None
    cells = []
    point_scalars = {}
    cell_scalars = {}
    mode = None
    n_point_data = n_cell_data = None

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        toks = line.split()
        key = toks[0].upper()

        if key == "POINTS":
            n = int(toks[1])
            vals = []
            i += 1
            while len(vals) < 3 * n:
                vals.extend(float(x) for x in lines[i].split())
                i += 1
            points = np.array(vals, dtype=float).reshape(n, 3)
            continue

        if key == "LINES":
            total = int(toks[2])
            vals = []
            i += 1
            while len(vals) < total:
                vals.extend(int(x) for x in lines[i].split())
                i += 1
            k = 0
            while k < len(vals):
                m = vals[k]
                cells.append(vals[k + 1 : k + 1 + m])
                k += m + 1
            continue

        if key == "POINT_DATA":
            mode = "point"
            n_point_data = int(toks[1])
            i += 1
            continue

        if key == "CELL_DATA":
            mode = "cell"
            n_cell_data = int(toks[1])
            i += 1
            continue

        if key == "SCALARS":
            name = toks[1]
            i += 1
            if i < len(lines) and lines[i].strip().upper().startswith("LOOKUP_TABLE"):
                i += 1

            n = n_point_data if mode == "point" else n_cell_data
            vals = []
            while len(vals) < n:
                vals.extend(float(x) for x in lines[i].split())
                i += 1

            arr = np.array(vals[:n], dtype=float)
            if mode == "point":
                point_scalars[name] = arr
            elif mode == "cell":
                cell_scalars[name] = arr
            continue

        i += 1

    if points is None:
        raise RuntimeError("VTK file does not contain POINTS.")

    return points, cells, point_scalars, cell_scalars


def normalize_points(points: np.ndarray):
    center = points.mean(axis=0)
    centered = points - center
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale == 0:
        raise RuntimeError("Point cloud has zero extent.")
    return centered / scale, center, scale


def build_segments(
    cells,
    points_raw,
    velocity,
    radius_um,
    flow_nl_min=None,
    hematocrit=None,
    diam_um=None,
    shear_stress=None,
    point_pressure=None,
):
    segments = []
    for cell_index, ids in enumerate(cells):
        if len(ids) < 2:
            continue
        signed_velocity = float(velocity[cell_index])
        speed = abs(signed_velocity)
        cell_radius = float(radius_um[cell_index])
        cell_diam = float(diam_um[cell_index]) if diam_um is not None else cell_radius * 2.0
        signed_flow = signed_velocity
        if flow_nl_min is not None:
            signed_flow = float(np.sign(signed_velocity) * abs(float(flow_nl_min[cell_index])))
        cell_hematocrit = float(hematocrit[cell_index]) if hematocrit is not None else 0.0
        cell_shear = float(shear_stress[cell_index]) if shear_stress is not None else 0.0
        for a, b in zip(ids[:-1], ids[1:]):
            pa = points_raw[a]
            pb = points_raw[b]
            length_raw = float(np.linalg.norm(pb - pa))
            pressure_a = float(point_pressure[a]) if point_pressure is not None else 0.0
            pressure_b = float(point_pressure[b]) if point_pressure is not None else 0.0
            segments.append(
                {
                    "a": int(a),
                    "b": int(b),
                    "velocity": signed_velocity,
                    "speed": speed,
                    "flow": signed_flow,
                    "hematocrit": cell_hematocrit,
                    "radius_um": cell_radius,
                    "diam_um": cell_diam,
                    "shear_stress": cell_shear,
                    "pressure_a": pressure_a,
                    "pressure_b": pressure_b,
                    "pressure_drop": pressure_a - pressure_b,
                    "length_raw": length_raw,
                }
            )
    return segments


def build_adjacency(segments):
    adjacency = defaultdict(list)
    for idx, seg in enumerate(segments):
        a = seg["a"]
        b = seg["b"]
        w = seg["length_raw"]
        adjacency[a].append((b, w, idx))
        adjacency[b].append((a, w, idx))
    return adjacency


def connected_components(adjacency):
    visited = set()
    components = []

    for node in adjacency:
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        nodes = []
        while stack:
            current = stack.pop()
            nodes.append(current)
            for nxt, _, _ in adjacency[current]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(nodes)

    components.sort(key=len, reverse=True)
    return components


def dijkstra(adjacency, start, allowed_nodes):
    allowed = set(allowed_nodes)
    distances = {start: 0.0}
    previous = {}
    queue = [(0.0, start)]

    while queue:
        dist, node = heapq.heappop(queue)
        if dist != distances[node]:
            continue
        for nxt, weight, _ in adjacency[node]:
            if nxt not in allowed:
                continue
            new_dist = dist + weight
            if nxt not in distances or new_dist < distances[nxt]:
                distances[nxt] = new_dist
                previous[nxt] = node
                heapq.heappush(queue, (new_dist, nxt))

    return distances, previous


def recover_path(previous, start, end):
    path = [end]
    while path[-1] != start:
        parent = previous.get(path[-1])
        if parent is None:
            raise RuntimeError("Could not recover camera path through the graph.")
        path.append(parent)
    path.reverse()
    return path


def resample_polyline(points: np.ndarray, sample_count: int):
    if len(points) < 2:
        return points.copy()

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = cumulative[-1]
    if total == 0:
        return np.repeat(points[:1], sample_count, axis=0)

    targets = np.linspace(0.0, total, sample_count)
    result = []
    seg_idx = 0

    for target in targets:
        while seg_idx < len(segment_lengths) - 1 and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        seg_len = segment_lengths[seg_idx]
        if seg_len == 0:
            result.append(points[seg_idx].copy())
            continue
        local = (target - cumulative[seg_idx]) / seg_len
        result.append(points[seg_idx] * (1.0 - local) + points[seg_idx + 1] * local)

    return np.vstack(result)


def normalize_vector(vec: np.ndarray):
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def smooth_polyline(points: np.ndarray, passes: int = 3):
    if len(points) <= 2:
        return points.copy()

    result = points.copy()
    for _ in range(max(passes, 0)):
        previous = result.copy()
        result[1:-1] = previous[1:-1] * 0.5 + (previous[:-2] + previous[2:]) * 0.25
    return result


def build_exterior_camera_path(points_norm: np.ndarray, guide_path: np.ndarray, min_clearance: float):
    if len(guide_path) < 2:
        return guide_path.copy()

    camera_points = []
    world_up = np.array([0.0, 1.0, 0.0], dtype=float)
    world_right = np.array([1.0, 0.0, 0.0], dtype=float)
    sample_count = len(guide_path)

    for index, base in enumerate(guide_path):
        prev_point = guide_path[index - 1] if index > 0 else guide_path[index]
        next_point = guide_path[index + 1] if index < sample_count - 1 else guide_path[index]
        tangent = normalize_vector(next_point - prev_point)
        if not np.any(tangent):
            tangent = world_right.copy()

        radial = normalize_vector(base)
        if not np.any(radial):
            radial = world_up.copy()

        side = normalize_vector(np.cross(tangent, radial))
        if not np.any(side):
            side = normalize_vector(np.cross(tangent, world_up))
        if not np.any(side):
            side = normalize_vector(np.cross(tangent, world_right))

        lift = normalize_vector(np.cross(side, tangent))
        if not np.any(lift):
            lift = world_up.copy()

        progress = index / max(sample_count - 1, 1)
        weave = np.sin(progress * np.pi * 3.2 + 0.45)
        lean = np.cos(progress * np.pi * 2.1 - 0.3)
        outward = normalize_vector(
            radial * (0.88 + 0.12 * (1.0 - progress))
            + side * (0.48 * weave)
            + lift * (0.24 * lean)
        )
        if not np.any(outward):
            outward = radial.copy()

        core_pull = np.sin(progress * np.pi) ** 1.35
        offset = min_clearance * (1.82 - 0.96 * core_pull + 0.08 * np.sin(progress * np.pi * 2.0))
        candidate = base + outward * offset
        nearest = float(np.min(np.linalg.norm(points_norm - candidate, axis=1)))

        attempt = 0
        while nearest < min_clearance and attempt < 8:
            attempt += 1
            offset += min_clearance * 0.34
            lateral = side * ((-1.0) ** attempt) * min_clearance * (0.18 + attempt * 0.04)
            candidate = base + outward * offset + lateral
            nearest = float(np.min(np.linalg.norm(points_norm - candidate, axis=1)))

        camera_points.append(candidate)

    camera_points = np.vstack(camera_points)
    smoothed = smooth_polyline(camera_points, passes=4)

    for index, base in enumerate(guide_path):
        candidate = smoothed[index]
        nearest = float(np.min(np.linalg.norm(points_norm - candidate, axis=1)))
        if nearest < min_clearance * 0.92:
            direction = normalize_vector(candidate - base)
            if not np.any(direction):
                direction = normalize_vector(base)
            if not np.any(direction):
                direction = world_up.copy()
            smoothed[index] = base + direction * min_clearance

    return resample_polyline(smoothed, len(guide_path))


def choose_tour(points_raw: np.ndarray, points_norm: np.ndarray, adjacency, segments, scale: float):
    components = connected_components(adjacency)
    if not components:
        raise RuntimeError("No connected vessel component found.")

    component_nodes = components[0]
    component_set = set(component_nodes)
    component_points = points_raw[component_nodes]
    center = component_points.mean(axis=0)

    degrees = {node: len([1 for nxt, _, _ in adjacency[node] if nxt in component_set]) for node in component_nodes}
    endpoints = [node for node in component_nodes if degrees[node] == 1]
    branches = [node for node in component_nodes if degrees[node] >= 3]

    if len(endpoints) < 2:
        raise RuntimeError("Need at least two vessel endpoints to build a fly-through.")

    if not branches:
        dist0, _ = dijkstra(adjacency, endpoints[0], component_nodes)
        far_entry = max(endpoints, key=lambda node: dist0.get(node, -1.0))
        dist1, prev1 = dijkstra(adjacency, far_entry, component_nodes)
        far_exit = max(endpoints, key=lambda node: dist1.get(node, -1.0))
        path_nodes = recover_path(prev1, far_entry, far_exit)
        branch_strengths = []
        branch_points = np.empty((0, 3), dtype=float)
        hub_node = path_nodes[len(path_nodes) // 2]
    else:
        edge_lengths = np.array([seg["length_raw"] for seg in segments], dtype=float)
        span = np.ptp(component_points, axis=0)
        neighborhood_radius = max(float(np.median(edge_lengths)) * 18.0, float(np.max(span)) * 0.08)
        branch_coords = points_raw[branches]
        branch_density = {}
        branch_strengths = []
        for node in branches:
            distances = np.linalg.norm(branch_coords - points_raw[node], axis=1)
            strength = float(np.sum(distances < neighborhood_radius))
            branch_density[node] = strength
            branch_strengths.append(strength)

        entry_node = max(endpoints, key=lambda node: np.linalg.norm(points_raw[node] - center))
        dist_from_entry, prev_from_entry = dijkstra(adjacency, entry_node, component_nodes)
        hub_node = max(
            branches,
            key=lambda node: (
                branch_density[node],
                dist_from_entry.get(node, -1.0),
                -np.linalg.norm(points_raw[node] - center),
            ),
        )
        dist_from_hub, prev_from_hub = dijkstra(adjacency, hub_node, component_nodes)
        exit_candidates = [node for node in endpoints if node != entry_node]
        exit_node = max(
            exit_candidates,
            key=lambda node: (
                dist_from_hub.get(node, -1.0),
                np.linalg.norm(points_raw[node] - center),
            ),
        )
        path_nodes = recover_path(prev_from_entry, entry_node, hub_node) + recover_path(prev_from_hub, hub_node, exit_node)[1:]
        branch_points = points_norm[branches]

    path_points_norm = points_norm[path_nodes]
    path_points_raw = points_raw[path_nodes]
    path_length_norm = float(np.linalg.norm(np.diff(path_points_norm, axis=0), axis=1).sum())
    path_length_raw = float(np.linalg.norm(np.diff(path_points_raw, axis=0), axis=1).sum())

    segment_lengths_norm = np.linalg.norm(np.diff(path_points_norm, axis=0), axis=1)
    median_step = float(np.median(segment_lengths_norm)) if len(segment_lengths_norm) else 0.01
    sample_count = int(np.clip(path_length_norm / max(median_step, 1e-6) * 1.05, 360, 760))
    sampled_path = resample_polyline(path_points_norm, sample_count)
    segment_radii_norm = np.array([seg["radius_um"] / scale for seg in segments], dtype=float)
    segment_lengths_norm = np.array([seg["length_raw"] / scale for seg in segments], dtype=float)
    camera_clearance = float(
        np.clip(
            max(
                float(np.percentile(segment_radii_norm, 75)) * 7.5 if len(segment_radii_norm) else 0.0,
                float(np.median(segment_lengths_norm)) * 4.8 if len(segment_lengths_norm) else 0.0,
                0.045,
            ),
            0.045,
            0.18,
        )
    )
    camera_path = build_exterior_camera_path(points_norm, sampled_path, camera_clearance)

    branch_strengths = np.array(branch_strengths, dtype=float)
    if branch_strengths.size:
        branch_strengths = branch_strengths / max(float(branch_strengths.max()), 1.0)

    return {
        "path_nodes": path_nodes,
        "focus_path_norm": sampled_path,
        "camera_path_norm": camera_path,
        "path_length_raw": path_length_raw,
        "branch_points": branch_points,
        "branch_nodes": [int(node) for node in branches],
        "branch_strengths": branch_strengths,
        "branch_count": len(branches),
        "endpoint_count": len(endpoints),
        "hub_point": points_norm[hub_node].tolist(),
        "camera_clearance": camera_clearance,
    }


def colorized_data(points_norm, segments, robust_max, scale):
    radius_values_norm = np.array([seg["radius_um"] / scale for seg in segments], dtype=float)
    edge_lengths_norm = np.array([seg["length_raw"] / scale for seg in segments], dtype=float)
    median_edge_length_norm = float(np.median(edge_lengths_norm)) if len(edge_lengths_norm) else 0.01
    median_radius_norm = float(np.median(radius_values_norm)) if len(radius_values_norm) else 0.003
    min_display_radius = max(median_edge_length_norm * 0.52, median_radius_norm * 1.6)

    encoded_edges = []
    for seg, radius_norm, length_norm in zip(segments, radius_values_norm, edge_lengths_norm):
        speed_norm = 0.0 if robust_max <= 0 else min(seg["speed"] / robust_max, 1.0)
        encoded_edges.append(
            [
                seg["a"],
                seg["b"],
                round(float(speed_norm), 6),
                round(float(seg["velocity"]), 6),
                round(float(radius_norm), 6),
                round(float(length_norm), 6),
                round(float(seg["flow"]), 6),
                round(float(seg["hematocrit"]), 6),
                round(float(seg["diam_um"]), 6),
                round(float(seg["shear_stress"]), 6),
                round(float(seg["pressure_a"]), 6),
                round(float(seg["pressure_b"]), 6),
                round(float(seg["pressure_drop"]), 6),
            ]
        )

    return {
        "points": [[round(float(v), 6) for v in row] for row in points_norm],
        "edges": encoded_edges,
        "median_edge_length_norm": median_edge_length_norm,
        "median_radius_norm": median_radius_norm,
        "min_display_radius": min_display_radius,
    }


def build_render_strands(points_norm: np.ndarray, segments, robust_max: float, scale: float):
    incident_edges = defaultdict(list)
    for edge_index, seg in enumerate(segments):
        incident_edges[seg["a"]].append(edge_index)
        incident_edges[seg["b"]].append(edge_index)

    degrees = {node: len(edges) for node, edges in incident_edges.items()}
    visited_edges = set()
    strands = []

    def other_node(edge_index: int, node: int):
        seg = segments[edge_index]
        return seg["b"] if seg["a"] == node else seg["a"]

    def encode_strand(node_indices, edge_indices):
        if len(node_indices) < 2 or not edge_indices:
            return None

        strand_points = []
        strand_radii = []
        strand_speeds = []
        for idx, node in enumerate(node_indices):
            left_edge = edge_indices[idx - 1] if idx > 0 else edge_indices[0]
            right_edge = edge_indices[idx] if idx < len(edge_indices) else edge_indices[-1]
            related = [left_edge] if left_edge == right_edge else [left_edge, right_edge]
            radius_norm = float(np.mean([segments[edge]["radius_um"] / scale for edge in related]))
            speed_norm = float(np.mean([min(segments[edge]["speed"] / robust_max, 1.0) for edge in related]))
            strand_points.append([round(float(v), 6) for v in points_norm[node]])
            strand_radii.append(round(radius_norm, 6))
            strand_speeds.append(round(speed_norm, 6))

        return {
            "points": strand_points,
            "radii": strand_radii,
            "speeds": strand_speeds,
            "startNode": int(node_indices[0]),
            "endNode": int(node_indices[-1]),
            "degreeStart": int(degrees.get(node_indices[0], 1)),
            "degreeEnd": int(degrees.get(node_indices[-1], 1)),
        }

    def walk_from_anchor(start_node: int, first_edge: int):
        node_indices = [start_node]
        edge_indices = []
        current_node = start_node
        edge_index = first_edge

        while True:
            if edge_index in visited_edges:
                break
            visited_edges.add(edge_index)
            edge_indices.append(edge_index)
            next_node = other_node(edge_index, current_node)
            node_indices.append(next_node)
            if degrees.get(next_node, 0) != 2:
                break
            next_edges = [candidate for candidate in incident_edges[next_node] if candidate != edge_index]
            if not next_edges:
                break
            current_node = next_node
            edge_index = next_edges[0]

        return node_indices, edge_indices

    def walk_loop(first_edge: int):
        start_node = segments[first_edge]["a"]
        node_indices = [start_node]
        edge_indices = []
        current_node = start_node
        edge_index = first_edge

        while True:
            if edge_index in visited_edges:
                break
            visited_edges.add(edge_index)
            edge_indices.append(edge_index)
            next_node = other_node(edge_index, current_node)
            node_indices.append(next_node)
            next_edges = [candidate for candidate in incident_edges[next_node] if candidate != edge_index]
            if not next_edges:
                break
            current_node = next_node
            edge_index = next_edges[0]
            if current_node == start_node:
                break

        return node_indices, edge_indices

    anchor_nodes = sorted(node for node, degree in degrees.items() if degree != 2)
    for node in anchor_nodes:
        for edge_index in incident_edges[node]:
            if edge_index in visited_edges:
                continue
            strand = encode_strand(*walk_from_anchor(node, edge_index))
            if strand is not None:
                strands.append(strand)

    for edge_index in range(len(segments)):
        if edge_index in visited_edges:
            continue
        strand = encode_strand(*walk_loop(edge_index))
        if strand is not None:
            strands.append(strand)

    return strands


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lamina Cribrosa Vessel Immersive Tour</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  color-scheme: dark;
  --ink: #f5efe7;
  --muted: rgba(245, 239, 231, 0.72);
  --edge: rgba(255, 255, 255, 0.08);
  --glass: rgba(11, 15, 22, 0.58);
  --glass-strong: rgba(12, 16, 24, 0.78);
  --accent: #ff734d;
  --accent-soft: #ffd3a1;
  --cool: #2f8cff;
}

* {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  height: 100%;
  overflow: hidden;
  font-family: "Space Grotesk", "Segoe UI", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at 20% 15%, rgba(180, 58, 38, 0.18), transparent 28%),
    radial-gradient(circle at 82% 18%, rgba(34, 110, 255, 0.16), transparent 26%),
    radial-gradient(circle at 50% 45%, rgba(255, 122, 64, 0.08), transparent 34%),
    linear-gradient(180deg, #080c12 0%, #05070b 42%, #020306 100%);
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(255, 255, 255, 0.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.018) 1px, transparent 1px);
  background-size: 32px 32px;
  mask-image: radial-gradient(circle at center, black 0%, rgba(0, 0, 0, 0.72) 55%, transparent 85%);
}

#stage {
  position: fixed;
  inset: 0;
}

canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.vignette {
  position: fixed;
  inset: 0;
  pointer-events: none;
  background: radial-gradient(circle at 50% 46%, transparent 0%, rgba(4, 5, 9, 0.06) 36%, rgba(4, 5, 9, 0.52) 72%, rgba(4, 5, 9, 0.92) 100%);
}

.compat-notice {
  position: fixed;
  inset: 16px;
  z-index: 20;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.compat-notice.visible {
  display: flex;
}

.compat-card {
  width: min(700px, 100%);
  padding: 24px;
  border-radius: 24px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: linear-gradient(180deg, rgba(12, 15, 22, 0.9), rgba(8, 10, 15, 0.96));
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
}

.compat-card h2 {
  margin: 0 0 10px;
  font-size: clamp(28px, 4vw, 38px);
  line-height: 1;
}

.compat-card p {
  margin: 0 0 12px;
  color: var(--muted);
  line-height: 1.55;
}

.compat-card code {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.96em;
}

.controls {
  position: fixed;
  right: 16px;
  top: 18px;
  width: min(280px, calc(100vw - 32px));
  padding: 12px 12px;
  border-radius: 18px;
  backdrop-filter: blur(18px);
  background: linear-gradient(180deg, rgba(18, 22, 31, 0.62), rgba(10, 13, 20, 0.8));
  border: 1px solid var(--edge);
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
  z-index: 2;
}

.button-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

button {
  appearance: none;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.05);
  color: var(--ink);
  padding: 9px 10px;
  font: inherit;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
}

button:hover {
  transform: translateY(-1px);
  background: rgba(255, 255, 255, 0.09);
  border-color: rgba(255, 255, 255, 0.2);
}

button.active {
  background: linear-gradient(135deg, rgba(255, 115, 77, 0.22), rgba(255, 209, 102, 0.12));
  border-color: rgba(255, 167, 110, 0.45);
}

.speed-row {
  margin-top: 10px;
}

.phase-row {
  margin-top: 10px;
}

.phase-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.phase-btn {
  padding: 8px 8px;
  font-size: 11px;
  font-weight: 500;
}

.speed-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(245, 239, 231, 0.58);
}

input[type="range"] {
  width: 100%;
  accent-color: var(--accent);
}

.info-panel {
  position: fixed;
  right: 16px;
  top: 50%;
  transform: translateY(-50%);
  width: min(360px, calc(100vw - 32px));
  max-height: 85vh;
  overflow-y: auto;
  padding: 18px;
  border-radius: 20px;
  backdrop-filter: blur(20px);
  background: linear-gradient(180deg, rgba(18, 22, 31, 0.78), rgba(10, 13, 20, 0.92));
  border: 1px solid var(--edge);
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
  z-index: 10;
  display: none;
}

.info-panel.visible {
  display: block;
}

.info-panel::-webkit-scrollbar {
  width: 6px;
}

.info-panel::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.03);
  border-radius: 3px;
}

.info-panel::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.15);
  border-radius: 3px;
}

.info-panel::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.25);
}

.info-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--edge);
}

.info-icon {
  width: 36px;
  height: 36px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
}

.info-icon.node {
  background: linear-gradient(135deg, rgba(79, 172, 254, 0.25), rgba(72, 202, 254, 0.12));
}

.info-icon.vessel {
  background: linear-gradient(135deg, rgba(255, 115, 77, 0.25), rgba(255, 209, 102, 0.12));
}

.info-title {
  flex: 1;
}

.info-title h3 {
  margin: 0 0 2px;
  font-size: 15px;
  font-weight: 600;
}

.info-title p {
  margin: 0;
  font-size: 12px;
  color: var(--muted);
}

.info-close {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  border: none;
  background: rgba(255, 255, 255, 0.08);
  color: var(--ink);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  transition: background 120ms ease;
}

.info-close:hover {
  background: rgba(255, 255, 255, 0.15);
}

.info-grid {
  display: grid;
  gap: 10px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 10px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.02);
}

.info-row:hover {
  background: rgba(255, 255, 255, 0.04);
}

.info-label {
  font-size: 12px;
  color: rgba(245, 239, 231, 0.6);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.info-value {
  font-size: 13px;
  font-weight: 600;
  font-family: "IBM Plex Mono", monospace;
}

.info-value.highlight {
  color: var(--accent);
}

.info-value.warning {
  color: #ffaa00;
}

.info-value.success {
  color: #4ade80;
}

.info-value.positive {
  color: #4ade80;
}

.info-value.negative {
  color: #fda4af;
}

.info-value.error {
  color: #f87171;
}

.info-section {
  margin-top: 16px;
}

.info-section h4 {
  margin: 0 0 10px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: rgba(245, 239, 231, 0.45);
}

.info-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.info-tag {
  padding: 4px 10px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.06);
  font-size: 11px;
  font-family: "IBM Plex Mono", monospace;
}

.info-tag.positive {
  background: rgba(74, 222, 128, 0.15);
  color: #4ade80;
}

.info-tag.negative {
  background: rgba(248, 113, 113, 0.15);
  color: #f87171;
}

@media (max-width: 560px) {
  .controls {
    right: 10px;
    top: 10px;
    width: min(260px, calc(100vw - 20px));
    padding: 10px 10px;
  }

  .phase-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
</head>
<body>
<div id="stage"></div>
<div class="vignette"></div>
<div class="compat-notice" id="compatNotice">
  <div class="compat-card">
    <h2>Local Resource Load Failed</h2>
    <p>This viewer now supports direct <code>file:///</code> opening in most browsers. If you are seeing this card, the local <code>Three.js</code> bundle did not load successfully.</p>
    <p>Try opening the same file again, or start a local server in <code>E:\KSU\mi3d</code> and use <code>http://127.0.0.1:8000/lc_3_interactive_velocity_viewer.html</code>.</p>
    <p>The generator script can be rerun any time from <code>vtk_to_interactive_viewer.py</code>.</p>
  </div>
</div>

<section class="info-panel" id="infoPanel">
  <div class="info-header">
    <div class="info-icon" id="infoIcon"></div>
    <div class="info-title">
      <h3 id="infoTitle">Info</h3>
      <p id="infoSubtitle">Click to inspect</p>
    </div>
    <button class="info-close" id="infoCloseBtn">&times;</button>
  </div>
  <div class="info-grid" id="infoGrid"></div>
</section>

<section class="controls">
  <div class="button-row">
      <button id="playPauseBtn" class="active">Pause</button>
      <button id="restartBtn">Restart</button>
  </div>

  <div class="phase-row">
    <div class="speed-head">
      <span>Phases</span>
      <span>Jump</span>
    </div>
    <div class="phase-grid">
      <button class="phase-btn" data-phase-key="top-overview" data-phase-progress="0.07">Overview</button>
      <button class="phase-btn" data-phase-key="perimeter-orbit" data-phase-progress="0.22">Perimeter</button>
      <button class="phase-btn" data-phase-key="core-approach" data-phase-progress="0.37">Approach</button>
      <button class="phase-btn" data-phase-key="interior-grove" data-phase-progress="0.66">Interior</button>
      <button class="phase-btn" data-phase-key="exit-reveal" data-phase-progress="0.94">Exit</button>
    </div>
  </div>

  <div class="speed-row">
    <div class="speed-head">
      <span>Playback speed</span>
      <span id="speedReadout">1.0x</span>
    </div>
    <input id="speedSlider" type="range" min="0.5" max="1.8" step="0.1" value="1.0">
  </div>
</section>

<script id="viewer-data" type="application/json">__VIEWER_DATA__</script>
<script src="./vendor/three.global.js"></script>
<script>
const compatNotice = document.getElementById("compatNotice");
if (typeof window.THREE === "undefined") {
  if (compatNotice) {
    compatNotice.classList.add("visible");
  }
  throw new Error("Three.js bundle failed to load.");
}

const data = JSON.parse(document.getElementById("viewer-data").textContent);
const points = data.points.map((row) => new THREE.Vector3(row[0], row[1], row[2]));
const pointPressure = Array.isArray(data.pointPressure) ? data.pointPressure : null;
const vesselStrands = data.vesselStrands;
const branchPoints = data.branchPoints.map((row) => new THREE.Vector3(row[0], row[1], row[2]));
const branchNodes = data.branchNodes;
const branchStrengths = data.branchStrengths;
const hubPoint = new THREE.Vector3(data.hubPoint[0], data.hubPoint[1], data.hubPoint[2]);
const cameraPoints = data.cameraPath.map((row) => new THREE.Vector3(row[0], row[1], row[2]));
const focusPoints = data.focusPath.map((row) => new THREE.Vector3(row[0], row[1], row[2]));

const stage = document.getElementById("stage");
const playPauseBtn = document.getElementById("playPauseBtn");
const restartBtn = document.getElementById("restartBtn");
const phaseButtons = Array.from(document.querySelectorAll(".phase-btn"));
const speedSlider = document.getElementById("speedSlider");
const speedReadout = document.getElementById("speedReadout");

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.28;
renderer.domElement.style.cursor = "grab";
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x06080d, 0.19);

const camera = new THREE.PerspectiveCamera(54, window.innerWidth / window.innerHeight, 0.001, 100);
camera.position.set(2.0, 0.72, 1.58);
scene.add(camera);

const ambient = new THREE.AmbientLight(0xfff0e0, 0.42);
scene.add(ambient);

const hemi = new THREE.HemisphereLight(0x86b8ff, 0x1e0806, 1.05);
scene.add(hemi);

const keyLight = new THREE.DirectionalLight(0xffd4bd, 1.2);
keyLight.position.set(1.7, 2.3, 1.4);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x5d88ff, 0.35);
fillLight.position.set(-1.2, 0.6, -1.6);
scene.add(fillLight);

const cameraLight = new THREE.PointLight(0xfff3dd, 1.35, 1.7, 2.0);
camera.add(cameraLight);

const vesselWallTint = new THREE.Color("#dbe8f6");
const scaffoldTint = new THREE.Color("#a6bed7");
const balancedTint = new THREE.Color("#7bd1ff");
const sourceTint = new THREE.Color("#ffc76d");
const alertTint = new THREE.Color("#ff5b6e");
const flowPaletteStops = [
  { t: 0.0, color: new THREE.Color("#4a0d08") },
  { t: 0.22, color: new THREE.Color("#74140c") },
  { t: 0.5, color: new THREE.Color("#a61f12") },
  { t: 0.78, color: new THREE.Color("#dd4523") },
  { t: 1.0, color: new THREE.Color("#ffd0b7") },
];

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function samplePalette(t) {
  const value = clamp(t, 0, 1);
  for (let i = 0; i < flowPaletteStops.length - 1; i += 1) {
    const left = flowPaletteStops[i];
    const right = flowPaletteStops[i + 1];
    if (value <= right.t) {
      const local = (value - left.t) / (right.t - left.t || 1);
      return left.color.clone().lerp(right.color, local);
    }
  }
  return flowPaletteStops[flowPaletteStops.length - 1].color.clone();
}

function vecFromPointIndex(index) {
  return points[index];
}

const vesselMaterial = new THREE.MeshPhysicalMaterial({
  color: vesselWallTint,
  roughness: 0.2,
  metalness: 0.0,
  clearcoat: 0.28,
  transparent: true,
  opacity: 0.24,
  transmission: 0.34,
  thickness: 0.12,
  ior: 1.08,
  attenuationDistance: 0.72,
  attenuationColor: new THREE.Color("#8fb5da"),
  side: THREE.DoubleSide,
  depthWrite: false,
});
const edgeRecords = [];
const pointEdges = Array.from({ length: points.length }, () => []);
const sceneScale = data.stats?.scale || 1;
const velocityAbsMax = Math.max(
  Math.abs(data.stats?.velocityMin ?? 0),
  Math.abs(data.stats?.velocityMax ?? 0),
  1e-6,
);
const pressureRange = Math.max(
  Math.abs((data.stats?.pressureMax ?? 0) - (data.stats?.pressureMin ?? 0)),
  1e-6,
);
data.edges.forEach((edge, index) => {
  const speedNorm = edge[2];
  const velocity = edge[3];
  const radiusNormRaw = edge[4];
  const displayRadius = Math.max(data.visual.minDisplayRadius, radiusNormRaw * data.visual.radiusBoost);
  const length = edge[5];
  const flow = edge[6];
  const hematocrit = edge[7];
  const diamUm = edge[8] ?? radiusNormRaw * sceneScale * 2;
  const shearStress = edge[9] ?? 0;
  const pressureStart = edge[10] ?? null;
  const pressureEnd = edge[11] ?? null;
  const pressureDrop = edge[12] ?? (pressureStart != null && pressureEnd != null ? pressureStart - pressureEnd : 0);
  const radiusUm = diamUm * 0.5;
  const directionForward = flow < 0 ? false : velocity >= 0 || flow >= 0;
  const pressureConflict = pressureStart != null
    && pressureEnd != null
    && Math.abs(pressureDrop) > pressureRange * 0.005
    && ((flow >= 0 && pressureDrop < 0) || (flow < 0 && pressureDrop > 0));
  const edgeRecord = {
    index,
    a: edge[0],
    b: edge[1],
    speedNorm,
    velocity,
    speed: Math.abs(velocity),
    flow,
    flowAbs: Math.abs(flow),
    hematocrit,
    diamUm,
    shearStress,
    radius: displayRadius,
    displayRadius,
    radiusNormRaw,
    radiusUm,
    velocityNorm: clamp(Math.abs(velocity) / velocityAbsMax, 0, 1),
    pressureStart,
    pressureEnd,
    pressureDrop,
    pressureConflict,
    length,
    preferredStart: directionForward ? edge[0] : edge[1],
    preferredEnd: directionForward ? edge[1] : edge[0],
  };
  edgeRecords.push(edgeRecord);
  pointEdges[edge[0]].push(edgeRecord);
  pointEdges[edge[1]].push(edgeRecord);
});

function initialNormalFromTangent(tangent) {
  let normal = new THREE.Vector3(0, 1, 0);
  if (Math.abs(tangent.dot(normal)) > 0.92) {
    normal.set(1, 0, 0);
  }
  normal.addScaledVector(tangent, -normal.dot(tangent)).normalize();
  return normal;
}

function sampleScalarAlongPath(values, cumulativeLengths, targetDistance) {
  if (values.length === 1 || cumulativeLengths.length === 1) {
    return values[0];
  }

  for (let i = 0; i < cumulativeLengths.length - 1; i += 1) {
    const left = cumulativeLengths[i];
    const right = cumulativeLengths[i + 1];
    if (targetDistance <= right || i === cumulativeLengths.length - 2) {
      const span = Math.max(right - left, 1e-6);
      const local = clamp((targetDistance - left) / span, 0, 1);
      return values[i] + (values[i + 1] - values[i]) * local;
    }
  }

  return values[values.length - 1];
}

function buildSmoothVesselGeometry(strand) {
  if (!strand || !strand.points || strand.points.length < 2) {
    return null;
  }

  const centers = strand.points.map((row) => new THREE.Vector3(row[0], row[1], row[2]));
  const exactSegmentTangents = [];
  const cumulativeLengths = [0];
  for (let i = 1; i < centers.length; i += 1) {
    const segment = centers[i].clone().sub(centers[i - 1]);
    const segmentLength = segment.length();
    exactSegmentTangents.push(segmentLength > 1e-8 ? segment.normalize() : new THREE.Vector3(1, 0, 0));
    cumulativeLengths.push(cumulativeLengths[i - 1] + segmentLength);
  }
  const totalLength = cumulativeLengths[cumulativeLengths.length - 1];
  if (totalLength <= 1e-6) {
    return null;
  }

  const radialSegments = 18;
  const sampleCount = Math.max(
    12,
    Math.min(220, Math.round(totalLength / Math.max(data.visual.minDisplayRadius * 0.44, 1e-4))),
  );

  function samplePolylineFrame(targetDistance) {
    if (targetDistance <= 0) {
      return { point: centers[0].clone(), tangent: exactSegmentTangents[0].clone() };
    }
    if (targetDistance >= totalLength) {
      return { point: centers[centers.length - 1].clone(), tangent: exactSegmentTangents[exactSegmentTangents.length - 1].clone() };
    }

    for (let i = 0; i < cumulativeLengths.length - 1; i += 1) {
      const left = cumulativeLengths[i];
      const right = cumulativeLengths[i + 1];
      if (targetDistance <= right || i === cumulativeLengths.length - 2) {
        const span = Math.max(right - left, 1e-6);
        const local = clamp((targetDistance - left) / span, 0, 1);
        return {
          point: centers[i].clone().lerp(centers[i + 1], local),
          tangent: exactSegmentTangents[i].clone(),
        };
      }
    }

    return { point: centers[centers.length - 1].clone(), tangent: exactSegmentTangents[exactSegmentTangents.length - 1].clone() };
  }

  const samplePoints = [];
  const tangents = [];
  const sampleRadii = [];
  for (let i = 0; i < sampleCount; i += 1) {
    const u = sampleCount === 1 ? 0 : i / (sampleCount - 1);
    const distanceAlong = totalLength * u;
    const radiusNorm = sampleScalarAlongPath(strand.radii, cumulativeLengths, distanceAlong);
    const frame = samplePolylineFrame(distanceAlong);
    samplePoints.push(frame.point);
    tangents.push(frame.tangent.normalize());
    sampleRadii.push(Math.max(data.visual.minDisplayRadius * 0.74, radiusNorm * data.visual.radiusBoost));
  }

  const startBlendRadius = strand.degreeStart >= 3
    ? sampleRadii[0] * (1.08 + Math.min(strand.degreeStart - 3, 3) * 0.03)
    : sampleRadii[0];
  const endBlendRadius = strand.degreeEnd >= 3
    ? sampleRadii[sampleRadii.length - 1] * (1.08 + Math.min(strand.degreeEnd - 3, 3) * 0.03)
    : sampleRadii[sampleRadii.length - 1];
  const blendCount = Math.max(2, Math.min(6, Math.floor(sampleCount * 0.12)));
  for (let i = 0; i < blendCount; i += 1) {
    const t = 1.0 - i / Math.max(blendCount - 1, 1);
    sampleRadii[i] = Math.max(sampleRadii[i], THREE.MathUtils.lerp(sampleRadii[i], startBlendRadius, t * 0.72));
    const endIndex = sampleRadii.length - 1 - i;
    sampleRadii[endIndex] = Math.max(sampleRadii[endIndex], THREE.MathUtils.lerp(sampleRadii[endIndex], endBlendRadius, t * 0.72));
  }

  const normals = new Array(sampleCount);
  const binormals = new Array(sampleCount);
  normals[0] = initialNormalFromTangent(tangents[0]);
  binormals[0] = new THREE.Vector3().crossVectors(tangents[0], normals[0]).normalize();
  normals[0] = new THREE.Vector3().crossVectors(binormals[0], tangents[0]).normalize();

  for (let i = 1; i < sampleCount; i += 1) {
    const prevNormal = normals[i - 1].clone();
    const axis = new THREE.Vector3().crossVectors(tangents[i - 1], tangents[i]);
    if (axis.lengthSq() > 1e-10) {
      axis.normalize();
      const angle = Math.acos(clamp(tangents[i - 1].dot(tangents[i]), -1, 1));
      prevNormal.applyQuaternion(new THREE.Quaternion().setFromAxisAngle(axis, angle)).normalize();
    }
    normals[i] = prevNormal;
    binormals[i] = new THREE.Vector3().crossVectors(tangents[i], normals[i]).normalize();
    if (binormals[i].lengthSq() < 1e-10) {
      binormals[i] = binormals[i - 1].clone();
    } else {
      normals[i] = new THREE.Vector3().crossVectors(binormals[i], tangents[i]).normalize();
    }
  }

  const vertexCount = sampleCount * radialSegments;
  const positions = new Float32Array(vertexCount * 3);
  const normalsArray = new Float32Array(vertexCount * 3);
  const indexCount = (sampleCount - 1) * radialSegments * 6;
  const indexArray = vertexCount > 65535 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);

  for (let ring = 0; ring < sampleCount; ring += 1) {
    const center = samplePoints[ring];
    const normal = normals[ring];
    const binormal = binormals[ring];
    const radius = sampleRadii[ring];

    for (let side = 0; side < radialSegments; side += 1) {
      const angle = (side / radialSegments) * Math.PI * 2;
      const cosAngle = Math.cos(angle);
      const sinAngle = Math.sin(angle);
      const radial = new THREE.Vector3()
        .copy(normal)
        .multiplyScalar(cosAngle)
        .addScaledVector(binormal, sinAngle)
        .normalize();
      const position = center.clone().addScaledVector(radial, radius);
      const vertexIndex = ring * radialSegments + side;
      const offset = vertexIndex * 3;
      positions[offset + 0] = position.x;
      positions[offset + 1] = position.y;
      positions[offset + 2] = position.z;
      normalsArray[offset + 0] = radial.x;
      normalsArray[offset + 1] = radial.y;
      normalsArray[offset + 2] = radial.z;
    }
  }

  let cursor = 0;
  for (let ring = 0; ring < sampleCount - 1; ring += 1) {
    for (let side = 0; side < radialSegments; side += 1) {
      const nextSide = (side + 1) % radialSegments;
      const a = ring * radialSegments + side;
      const b = ring * radialSegments + nextSide;
      const c = (ring + 1) * radialSegments + side;
      const d = (ring + 1) * radialSegments + nextSide;
      indexArray[cursor + 0] = a;
      indexArray[cursor + 1] = c;
      indexArray[cursor + 2] = b;
      indexArray[cursor + 3] = b;
      indexArray[cursor + 4] = c;
      indexArray[cursor + 5] = d;
      cursor += 6;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normalsArray, 3));
  geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));
  geometry.computeBoundingSphere();
  return geometry;
}

const vesselGroup = new THREE.Group();
vesselGroup.frustumCulled = false;
vesselStrands.forEach((strand) => {
  const geometry = buildSmoothVesselGeometry(strand);
  if (!geometry) {
    return;
  }
  const strandMesh = new THREE.Mesh(geometry, vesselMaterial);
  strandMesh.frustumCulled = false;
  vesselGroup.add(strandMesh);
});
scene.add(vesselGroup);

const jointCandidates = [];
pointEdges.forEach((incidentEdges, pointIndex) => {
  if (incidentEdges.length < 3) {
    return;
  }

  const center = vecFromPointIndex(pointIndex);
  let radiusMax = 0;
  let radiusTotal = 0;
  let speedTotal = 0;
  incidentEdges.forEach((edge) => {
    radiusMax = Math.max(radiusMax, edge.radius);
    radiusTotal += edge.radius;
    speedTotal += edge.speedNorm;
  });
  const radiusMean = radiusTotal / incidentEdges.length;
  const incidentLengths = incidentEdges.map((edge) => {
    const other = edge.a === pointIndex ? edge.b : edge.a;
    return center.distanceTo(vecFromPointIndex(other));
  });
  const shortestLength = Math.max(Math.min(...incidentLengths), 1e-6);

  const bendStrength = clamp(0.32 + (incidentEdges.length - 3) * 0.12, 0.32, 1.0);

  jointCandidates.push({
    point: center,
    degree: incidentEdges.length,
    bendStrength,
    radiusMean,
    radius: radiusMax,
    shortestLength,
    speedNorm: speedTotal / incidentEdges.length,
  });
});

if (jointCandidates.length) {
  const jointGeometry = new THREE.IcosahedronGeometry(1, 3);
  const jointMaterial = new THREE.MeshPhysicalMaterial({
    color: vesselWallTint,
    roughness: 0.2,
    metalness: 0.0,
    clearcoat: 0.28,
    transparent: true,
    opacity: 0.18,
    transmission: 0.24,
    depthWrite: false,
  });
  const jointMesh = new THREE.InstancedMesh(jointGeometry, jointMaterial, jointCandidates.length);
  jointMesh.frustumCulled = false;
  jointMesh.renderOrder = 2;
  jointMesh.visible = false;
  const jointDummy = new THREE.Object3D();

  jointCandidates.forEach((joint, index) => {
    const baseRadius = Math.min(joint.radius, joint.radiusMean * 1.08);
    const desiredScale = baseRadius * (1.02 + Math.min(joint.degree - 3, 3) * 0.04 + joint.bendStrength * 0.05);
    const lowerBound = baseRadius * 0.98;
    const upperBound = Math.max(baseRadius * 1.12, joint.shortestLength * 0.48);
    const scale = clamp(desiredScale, lowerBound, upperBound);
    jointDummy.position.copy(joint.point);
    jointDummy.scale.setScalar(scale);
    jointDummy.updateMatrix();
    jointMesh.setMatrixAt(index, jointDummy.matrix);
  });

  jointMesh.instanceMatrix.needsUpdate = true;
  jointMesh.computeBoundingSphere();
  scene.add(jointMesh);
}

const linePositions = [];
const lineColors = [];
edgeRecords.forEach((edge) => {
  const start = vecFromPointIndex(edge.a);
  const end = vecFromPointIndex(edge.b);
  const edgeColor = samplePalette(edge.speedNorm).lerp(new THREE.Color("#ffe0d1"), edge.speedNorm * 0.12);
  linePositions.push(start.x, start.y, start.z, end.x, end.y, end.z);
  lineColors.push(
    edgeColor.r,
    edgeColor.g,
    edgeColor.b,
    edgeColor.r,
    edgeColor.g,
    edgeColor.b,
  );
});

const coreLineGeometry = new THREE.BufferGeometry();
coreLineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
coreLineGeometry.setAttribute("color", new THREE.Float32BufferAttribute(lineColors, 3));
const coreLineMaterial = new THREE.LineBasicMaterial({
  vertexColors: true,
  transparent: true,
  opacity: 0.045,
});
const coreLines = new THREE.LineSegments(coreLineGeometry, coreLineMaterial);
coreLines.frustumCulled = false;
coreLines.renderOrder = 0;
coreLines.visible = false;
scene.add(coreLines);

const pathGeometry = new THREE.BufferGeometry().setFromPoints(cameraPoints);
const pathMaterial = new THREE.LineBasicMaterial({
  color: 0xf5dfc6,
  transparent: true,
  opacity: 0.08,
});
const pathLine = new THREE.Line(pathGeometry, pathMaterial);
pathLine.frustumCulled = false;
pathLine.renderOrder = 0;
pathLine.visible = false;
scene.add(pathLine);

function edgeFlux(edge) {
  return Math.max(Math.abs(edge.flow), 1e-4);
}

function edgeTargetWeight(edge) {
  const crossSectionProxy = Math.PI * Math.pow(Math.max(edge.radiusUm, 1e-4), 2);
  const volumeProxy = Math.max(edge.length, 1e-4) * crossSectionProxy;
  const hematocritWeight = 0.08 + Math.max(edge.hematocrit, 0) * 1.92;
  const flowSupport = 0.35 + edge.velocityNorm * 0.65;
  return Math.max(0.02, volumeProxy * hematocritWeight * flowSupport);
}

function edgeFlowMatchesPressure(edge) {
  if (edge.pressureStart == null || edge.pressureEnd == null) {
    return true;
  }
  if (Math.abs(edge.pressureDrop) <= pressureRange * 0.005) {
    return true;
  }
  return edge.flow >= 0 ? edge.pressureDrop >= 0 : edge.pressureDrop <= 0;
}

function branchContinuationScore(nodeIndex, incomingEdge, outgoingEdge) {
  if (!incomingEdge || !outgoingEdge || incomingEdge.index === outgoingEdge.index) {
    return 0;
  }

  const prevNode = incomingEdge.a === nodeIndex ? incomingEdge.b : incomingEdge.a;
  const nextNode = outgoingEdge.a === nodeIndex ? outgoingEdge.b : outgoingEdge.a;
  const incomingDir = points[nodeIndex].clone().sub(points[prevNode]).normalize();
  const outgoingDir = points[nextNode].clone().sub(points[nodeIndex]).normalize();
  const alignment = clamp((incomingDir.dot(outgoingDir) + 1) * 0.5, 0, 1);
  const caliberMatch = Math.min(incomingEdge.displayRadius, outgoingEdge.displayRadius) / Math.max(incomingEdge.displayRadius, outgoingEdge.displayRadius, 1e-6);
  const speedMatch = 1.0 - Math.abs(incomingEdge.speedNorm - outgoingEdge.speedNorm);
  return clamp(alignment * 0.56 + caliberMatch * 0.3 + speedMatch * 0.14, 0, 1);
}

function connectivityScoreForNode(nodeIndex, incomingEdges, outgoingEdges) {
  if (!incomingEdges.length || !outgoingEdges.length) {
    return 0;
  }

  const totalIncoming = incomingEdges.reduce((sum, edge) => sum + edgeFlux(edge), 0);
  let weightedScore = 0;
  incomingEdges.forEach((incomingEdge) => {
    let bestScore = 0;
    outgoingEdges.forEach((outgoingEdge) => {
      bestScore = Math.max(bestScore, branchContinuationScore(nodeIndex, incomingEdge, outgoingEdge));
    });
    weightedScore += edgeFlux(incomingEdge) * bestScore;
  });
  return clamp(weightedScore / Math.max(totalIncoming, 1e-6), 0, 1);
}

const nodeEdges = pointEdges;
const nodeFlowProfiles = nodeEdges.map((incidentEdges, nodeIndex) => {
  if (!incidentEdges.length) {
    return null;
  }

  const incomingEdges = incidentEdges.filter((edge) => edge.preferredEnd === nodeIndex);
  const outgoingEdges = incidentEdges.filter((edge) => edge.preferredStart === nodeIndex);
  const incomingFlux = incomingEdges.reduce((sum, edge) => sum + edgeFlux(edge), 0);
  const outgoingFlux = outgoingEdges.reduce((sum, edge) => sum + edgeFlux(edge), 0);
  const dominantDirectionRatio = incidentEdges.length
    ? Math.max(incomingEdges.length, outgoingEdges.length) / incidentEdges.length
    : 0;
  const fluxTotal = Math.max(incomingFlux + outgoingFlux, 1e-6);
  const massConservationError = Math.abs(outgoingFlux - incomingFlux) / Math.max(Math.max(incomingFlux, outgoingFlux), 1e-6);
  const connectivityScore = connectivityScoreForNode(nodeIndex, incomingEdges, outgoingEdges);
  const pressureConflictEdges = incidentEdges.filter((edge) => edge.pressureConflict);
  const pressureConsistency = incidentEdges.length
    ? 1.0 - pressureConflictEdges.length / incidentEdges.length
    : 1.0;
  const localPressure = pointPressure ? pointPressure[nodeIndex] : null;
  const imbalance = Math.abs(outgoingFlux - incomingFlux) / fluxTotal;
  const branchLike = incidentEdges.length >= 3;
  const hardAnomaly = branchLike && (!incomingEdges.length || !outgoingEdges.length);
  const birthFactor = clamp((outgoingFlux - incomingFlux) / Math.max(outgoingFlux, 1e-6), 0, 1);
  const deathFactor = clamp((incomingFlux - outgoingFlux) / Math.max(incomingFlux, 1e-6), 0, 1);

  let type = "balanced";
  if (!incomingEdges.length && outgoingEdges.length) {
    type = "source";
  } else if (!outgoingEdges.length && incomingEdges.length) {
    type = "sink";
  } else if (imbalance > 0.34) {
    type = "imbalanced";
  }

  let diagnosticLabel = "Conserved";
  if (hardAnomaly && incomingEdges.length) {
    diagnosticLabel = "All inflow / no outflow";
  } else if (hardAnomaly && outgoingEdges.length) {
    diagnosticLabel = "All outflow / no inflow";
  } else if (pressureConflictEdges.length >= Math.max(1, Math.floor(incidentEdges.length * 0.5))) {
    diagnosticLabel = "Pressure-flow mismatch";
  } else if (massConservationError > 0.22) {
    diagnosticLabel = outgoingFlux > incomingFlux ? "Outflow exceeds inflow" : "Inflow exceeds outflow";
  } else if (branchLike && connectivityScore < 0.42) {
    diagnosticLabel = "Weak branch continuity";
  }

  const isPhysicallyValid = !hardAnomaly && (!branchLike || massConservationError < 0.36) && pressureConsistency > 0.3;

  return {
    nodeIndex,
    point: points[nodeIndex],
    pressure: localPressure,
    degree: incidentEdges.length,
    incidentEdges,
    incomingEdges,
    outgoingEdges,
    incomingFlux,
    outgoingFlux,
    imbalance,
    massConservationError,
    connectivityScore,
    pressureConflictEdges,
    pressureConsistency,
    birthFactor,
    deathFactor,
    hardAnomaly,
    branchLike,
    isPhysicallyValid,
    diagnosticLabel,
    anomaly: clamp(
      (hardAnomaly ? 0.82 : 0.0)
        + (branchLike ? massConservationError * 0.56 + (1.0 - connectivityScore) * 0.24 + dominantDirectionRatio * 0.14 : 0.0)
        + (1.0 - pressureConsistency) * 0.18
        + (type === "source" || type === "sink" ? Math.max(0, incidentEdges.length - 1) * 0.04 : 0.0),
      0,
      1,
    ),
    speedNorm: incidentEdges.reduce((sum, edge) => sum + edge.speedNorm, 0) / incidentEdges.length,
    type,
  };
});

const spawnProfiles = nodeFlowProfiles.filter(
  (profile) => profile && profile.outgoingEdges.length && (profile.type === "source" || profile.outgoingFlux >= profile.incomingFlux * 0.9),
);

const branchGeometry = new THREE.IcosahedronGeometry(1, 1);
const branchMaterial = new THREE.MeshBasicMaterial({
  transparent: true,
  opacity: 0.22,
  depthWrite: false,
});
const branchMesh = new THREE.InstancedMesh(branchGeometry, branchMaterial, branchPoints.length);
branchMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
branchMesh.frustumCulled = false;
branchMesh.renderOrder = 6;
const branchDummy = new THREE.Object3D();

function updateBranchMarkers(focusPoint) {
  branchPoints.forEach((point, index) => {
    const strength = branchStrengths[index] || 0;
    const profile = nodeFlowProfiles[branchNodes[index]];
    const anomaly = profile ? profile.anomaly : 0;
    const continuity = profile ? profile.connectivityScore : 0;
    const distance = point.distanceTo(focusPoint);
    const glow = clamp(1.0 - distance / 0.22, 0, 1);
    const scale = data.visual.minDisplayRadius * (0.36 + strength * 0.28 + glow * 0.46 + anomaly * 0.52 + (1 - continuity) * 0.16);
    branchDummy.position.copy(point);
    branchDummy.scale.setScalar(scale);
    branchDummy.updateMatrix();
    branchMesh.setMatrixAt(index, branchDummy.matrix);

    const neutralColor = samplePalette(profile ? profile.speedNorm : strength).lerp(
      sourceTint,
      profile && profile.type === "source" ? 0.36 : 0.0,
    );
    const color = neutralColor
      .lerp(balancedTint, continuity * 0.22)
      .lerp(alertTint, anomaly);
    branchMesh.setColorAt(index, color);
  });
  branchMesh.instanceMatrix.needsUpdate = true;
  if (branchMesh.instanceColor) {
    branchMesh.instanceColor.needsUpdate = true;
  }
}

updateBranchMarkers(hubPoint);
scene.add(branchMesh);

const hubHalo = new THREE.Mesh(
  new THREE.SphereGeometry(data.visual.minDisplayRadius * 2.4, 18, 18),
  new THREE.MeshBasicMaterial({
    color: 0xb8d8ff,
    transparent: true,
    opacity: 0.055,
    depthWrite: false,
  }),
);
hubHalo.position.copy(hubPoint);
hubHalo.renderOrder = 5;
scene.add(hubHalo);

const dustCount = 520;
const dustPositions = new Float32Array(dustCount * 3);
const dustColors = new Float32Array(dustCount * 3);
for (let i = 0; i < dustCount; i += 1) {
  const radius = Math.pow(Math.random(), 0.7) * 2.3;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  const sinPhi = Math.sin(phi);
  dustPositions[i * 3 + 0] = radius * sinPhi * Math.cos(theta);
  dustPositions[i * 3 + 1] = radius * Math.cos(phi) * 0.8;
  dustPositions[i * 3 + 2] = radius * sinPhi * Math.sin(theta);
  const dustColor = samplePalette(Math.random() * 0.85);
  dustColors[i * 3 + 0] = dustColor.r;
  dustColors[i * 3 + 1] = dustColor.g;
  dustColors[i * 3 + 2] = dustColor.b;
}

const dustGeometry = new THREE.BufferGeometry();
dustGeometry.setAttribute("position", new THREE.BufferAttribute(dustPositions, 3));
dustGeometry.setAttribute("color", new THREE.BufferAttribute(dustColors, 3));
const dustMaterial = new THREE.PointsMaterial({
  size: 0.01,
  transparent: true,
  opacity: 0.16,
  depthWrite: false,
  vertexColors: true,
});
const dust = new THREE.Points(dustGeometry, dustMaterial);
dust.frustumCulled = false;
dust.visible = false;
scene.add(dust);

const cameraCurve = new THREE.CatmullRomCurve3(cameraPoints, false, "centripetal", 0.18);
const focusCurve = new THREE.CatmullRomCurve3(focusPoints, false, "centripetal", 0.16);
const worldUp = new THREE.Vector3(0, 1, 0);
const center = new THREE.Vector3(0, 0, 0);
const focusMid = focusCurve.getPointAt(0.5);

function getFrame(u) {
  const clamped = clamp(u, 0, 1);
  const position = cameraCurve.getPointAt(clamped);
  const tangent = cameraCurve.getTangentAt(clamped).normalize();
  const focus = focusCurve.getPointAt(clamped);
  const focusLead = focusCurve.getPointAt(clamp(clamped + 0.03, 0, 0.999));

  let normal = focus.clone().sub(position);
  if (normal.lengthSq() < 1e-6) {
    normal.copy(worldUp);
  } else {
    normal.normalize();
  }

  let side = new THREE.Vector3().crossVectors(tangent, normal);
  if (side.lengthSq() < 1e-6) {
    side = new THREE.Vector3().crossVectors(worldUp, tangent);
  }
  if (side.lengthSq() < 1e-6) {
    side = new THREE.Vector3(1, 0, 0).cross(tangent);
  }
  side.normalize();
  normal = new THREE.Vector3().crossVectors(side, tangent).normalize();

  return { position, tangent, side, normal, focus, focusLead };
}

let hubFocusIndex = 0;
let hubFocusDistance = Infinity;
focusPoints.forEach((point, index) => {
  const distance = point.distanceToSquared(hubPoint);
  if (distance < hubFocusDistance) {
    hubFocusDistance = distance;
    hubFocusIndex = index;
  }
});

const hubU = focusPoints.length > 1 ? hubFocusIndex / (focusPoints.length - 1) : 0.5;
let interiorStartU = clamp(hubU - 0.24, 0.08, 0.72);
let interiorEndU = clamp(hubU + 0.24, 0.28, 0.94);
if (interiorEndU - interiorStartU < 0.42) {
  const middle = (interiorStartU + interiorEndU) * 0.5;
  interiorStartU = clamp(middle - 0.21, 0.08, 0.72);
  interiorEndU = clamp(middle + 0.21, 0.28, 0.94);
}

function topDownSurveyState(localT) {
  const eased = easeInOutCubic(localT);
  const angle = -0.24 + eased * Math.PI * 1.06;
  const radiusX = 0.56;
  const radiusZ = 0.44;
  return {
    position: new THREE.Vector3(
      Math.cos(angle) * radiusX,
      1.62 + Math.sin(localT * Math.PI) * 0.1,
      Math.sin(angle) * radiusZ,
    ),
    lookAt: center.clone().lerp(hubPoint, 0.12),
    up: worldUp.clone(),
    phase: "Top overview",
    copy: "A bird's-eye pass restores the full model silhouette first, so the audience can register the vascular canopy before we descend into it.",
    vesselOpacity: 0.26,
    fogDensity: 0.12,
    pathOpacity: 0.1,
    hubOpacity: 0.08,
  };
}

function outerSurveyState(localT) {
  const eased = easeInOutCubic(localT);
  const angle = 0.54 + eased * Math.PI * 1.52;
  const radiusX = 2.2 - eased * 0.08;
  const radiusZ = 1.9 - eased * 0.08;
  return {
    position: new THREE.Vector3(
      Math.cos(angle) * radiusX,
      0.66 + Math.sin(eased * Math.PI * 1.15) * 0.16,
      Math.sin(angle) * radiusZ,
    ),
    lookAt: focusMid.clone().lerp(hubPoint, 0.08 + Math.sin(localT * Math.PI) * 0.03),
    up: worldUp.clone(),
    phase: "Perimeter orbit",
    copy: "The camera keeps the original outer ring around the whole model, preserving a clean architectural read before the interior roam begins.",
    vesselOpacity: 0.24,
    fogDensity: 0.15,
    pathOpacity: 0.09,
    hubOpacity: 0.08,
  };
}

const interiorEntryFrame = getFrame(interiorStartU);
const interiorEntryPosition = interiorEntryFrame.position
  .clone()
  .add(interiorEntryFrame.tangent.clone().multiplyScalar(-0.08))
  .add(interiorEntryFrame.normal.clone().multiplyScalar(0.04))
  .add(interiorEntryFrame.side.clone().multiplyScalar(-0.03));
const interiorEntryLookAt = interiorEntryFrame.focusLead.clone();

const interiorExitFrame = getFrame(interiorEndU);
const exitFrame = getFrame(0.985);
const exitRevealPosition = exitFrame.position
  .clone()
  .add(exitFrame.tangent.clone().multiplyScalar(0.22))
  .add(exitFrame.normal.clone().multiplyScalar(0.14))
  .add(exitFrame.side.clone().multiplyScalar(0.06));
const exitRevealLookAt = focusCurve.getPointAt(0.96).clone().lerp(center, 0.08);
const interiorStandBase = center.clone().lerp(hubPoint, 0.18).add(new THREE.Vector3(0, -0.028, 0));
const topOverviewEnd = 0.14;
const perimeterEnd = 0.3;
const approachEnd = 0.44;
const interiorEnd = 0.88;
const phaseDefinitions = [
  { key: "top-overview", end: topOverviewEnd },
  { key: "perimeter-orbit", end: perimeterEnd },
  { key: "core-approach", end: approachEnd },
  { key: "interior-grove", end: interiorEnd },
  { key: "exit-reveal", end: 1.0 },
];

function interiorAtriumState(localT) {
  const eased = easeInOutCubic(localT);
  const lookAngle = -0.36 + eased * Math.PI * 1.84;
  const stanceAngle = -0.18 + eased * Math.PI * 0.9;
  const stanceRadius = 0.012 + Math.sin(eased * Math.PI * 2.0) * 0.002;
  const breathe = Math.sin(eased * Math.PI * 4.0) * 0.004;
  const headLift = 0.46 + Math.sin(eased * Math.PI * 1.15) * 0.04;
  return {
    position: interiorStandBase
      .clone()
      .add(new THREE.Vector3(
        Math.cos(stanceAngle) * stanceRadius,
        breathe,
        Math.sin(stanceAngle) * stanceRadius,
      )),
    lookAt: center.clone().add(new THREE.Vector3(
      Math.cos(lookAngle) * 0.24,
      headLift,
      Math.sin(lookAngle) * 0.24,
    )),
    up: worldUp.clone(),
    phase: "Interior atrium",
    copy: "The camera now stands near the network center and looks upward while slowly sweeping around, reading the vessel canopy overhead instead of traveling along a single branch.",
    vesselOpacity: 0.17,
    fogDensity: 0.19,
    pathOpacity: 0.03,
    hubOpacity: 0.06,
  };
}

const interiorStartPose = interiorAtriumState(0.0);
const interiorEndPose = interiorAtriumState(1.0);

function getTourTargets(progress) {
  if (progress <= topOverviewEnd) {
    return topDownSurveyState(progress / topOverviewEnd);
  }

  if (progress <= perimeterEnd) {
    return outerSurveyState((progress - topOverviewEnd) / (perimeterEnd - topOverviewEnd));
  }

  if (progress <= approachEnd) {
    const local = easeInOutCubic((progress - perimeterEnd) / (approachEnd - perimeterEnd));
    const start = outerSurveyState(1.0);
    return {
      position: start.position.clone().lerp(interiorStartPose.position, local),
      lookAt: start.lookAt.clone().lerp(interiorStartPose.lookAt, local),
      up: worldUp.clone().lerp(interiorStartPose.up, local * 0.22).normalize(),
      phase: "Core approach",
      copy: "The descent now leaves the perimeter and settles toward the center of the vascular network, preparing for an upward-looking interior survey rather than a vessel-following flythrough.",
      vesselOpacity: 0.21,
      fogDensity: 0.16,
      pathOpacity: 0.06,
      hubOpacity: 0.09,
    };
  }

  if (progress <= interiorEnd) {
    const local = (progress - approachEnd) / (interiorEnd - approachEnd);
    return interiorAtriumState(local);
  }

  const local = easeInOutCubic((progress - interiorEnd) / (1.0 - interiorEnd));
  return {
    position: interiorEndPose.position.clone().lerp(exitRevealPosition, local),
    lookAt: interiorEndPose.lookAt.clone().lerp(exitRevealLookAt, local),
    up: interiorEndPose.up.clone().lerp(worldUp, 0.68 * local + 0.18).normalize(),
    phase: "Exit reveal",
    copy: "The final pullback returns to an outside read on the whole structure after the centered upward-looking survey within the network core.",
    vesselOpacity: 0.22,
    fogDensity: 0.15,
    pathOpacity: 0.06,
    hubOpacity: 0.09,
  };
}

function getPhaseKeyFromProgress(progress) {
  const wrapped = ((progress % 1) + 1) % 1;
  for (const phase of phaseDefinitions) {
    if (wrapped <= phase.end) {
      return phase.key;
    }
  }
  return phaseDefinitions[phaseDefinitions.length - 1].key;
}

function weightedChoice(items, weightFn) {
  if (!items.length) {
    return null;
  }

  const weights = items.map((item) => Math.max(0, weightFn(item)));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (total <= 1e-6) {
    return items[items.length - 1];
  }

  let target = Math.random() * total;
  for (let i = 0; i < items.length; i += 1) {
    target -= weights[i];
    if (target <= 0) {
      return items[i];
    }
  }
  return items[items.length - 1];
}

const RBC_ROUTE_ALPHA = 1.0;
const RBC_ROUTE_BETA = 0.45;
const RBC_PHASE_GAIN = 0.55;
const RBC_WAIT_MIN = 0.03;
const RBC_WAIT_MAX = 0.11;

function transitionFromEdge(edge) {
  return {
    edge,
    from: edge.preferredStart,
    to: edge.preferredEnd,
  };
}

const edgeTargetWeights = edgeRecords.map((edge) => edgeTargetWeight(edge));
const edgeTargetWeightTotal = edgeTargetWeights.reduce((sum, value) => sum + value, 0);
const edgeOccupancy = new Array(edgeRecords.length).fill(0);

function desiredParticlesForEdge(edgeIndex) {
  return particleCount * edgeTargetWeights[edgeIndex] / Math.max(edgeTargetWeightTotal, 1e-6);
}

function occupancyBiasForEdge(edge) {
  const target = desiredParticlesForEdge(edge.index);
  const deficit = target - edgeOccupancy[edge.index];
  return clamp(1.0 + deficit / Math.max(target, 1.0) * 1.15, 0.22, 3.2);
}

function selectSpawnProfile() {
  const pool = spawnProfiles.length ? spawnProfiles : nodeFlowProfiles.filter(Boolean);
  return weightedChoice(
    pool,
    (profile) => 0.12 + profile.outgoingFlux * (0.72 + profile.connectivityScore * 0.28) + profile.speedNorm * 0.4,
  );
}

function selectEdgeForRespawn(profile = null) {
  const candidates = profile && profile.outgoingEdges.length ? profile.outgoingEdges : edgeRecords;
  return weightedChoice(
    candidates,
    (edge) => edgeTargetWeights[edge.index] * occupancyBiasForEdge(edge) * (0.72 + edge.velocityNorm * 0.28),
  );
}

function nextEdgeFromNode(nodeIndex, previousEdge, options = {}) {
  const profile = nodeFlowProfiles[nodeIndex];
  let candidates = profile ? profile.outgoingEdges.slice() : nodeEdges[nodeIndex].slice();
  if (!candidates.length) {
    return null;
  }

  if (options.preferDifferent) {
    const filtered = candidates.filter((edge) => edge.index !== previousEdge.index);
    if (filtered.length) {
      candidates = filtered;
    }
  }

  const previousNode = previousEdge.a === nodeIndex ? previousEdge.b : previousEdge.a;
  const incoming = points[nodeIndex].clone().sub(points[previousNode]).normalize();
  const totalOutflow = candidates.reduce((sum, edge) => sum + Math.abs(edge.flow), 0);
  const meanDiameter = candidates.reduce((sum, edge) => sum + edge.diamUm, 0) / Math.max(candidates.length, 1);

  const chosen = weightedChoice(candidates, (edge) => {
    const other = edge.a === nodeIndex ? edge.b : edge.a;
    const outgoing = points[other].clone().sub(points[nodeIndex]).normalize();
    const alignment = clamp((incoming.dot(outgoing) + 1) * 0.5, 0, 1);
    const caliberMatch = Math.min(previousEdge.displayRadius, edge.displayRadius) / Math.max(previousEdge.displayRadius, edge.displayRadius, 1e-6);
    const flowProportion = totalOutflow > 0 ? Math.abs(edge.flow) / totalOutflow : edgeFlux(edge);
    const continuity = profile ? profile.connectivityScore : 0.5;
    const diameterBias = Math.pow(Math.max(edge.diamUm, 1e-4) / Math.max(meanDiameter, 1e-4), RBC_ROUTE_BETA);
    const phaseBias = Math.pow(clamp(edge.diamUm / Math.max(meanDiameter, 1e-4), 0.58, 1.8), RBC_PHASE_GAIN);
    const pressureBias = edgeFlowMatchesPressure(edge) ? 1.0 : 0.65;
    const occupancyBias = occupancyBiasForEdge(edge);
    return Math.max(
      0.01,
      Math.pow(Math.max(flowProportion, 1e-5), RBC_ROUTE_ALPHA)
        * diameterBias
        * phaseBias
        * (0.35 + alignment * 0.95)
        * (0.4 + caliberMatch * 0.6)
        * (0.52 + edge.speedNorm * 0.48)
        * (0.6 + continuity * 0.4)
        * pressureBias
        * occupancyBias,
    );
  });

  if (!chosen) {
    return null;
  }

  const targetNode = chosen.a === nodeIndex ? chosen.b : chosen.a;
  return { edge: chosen, from: nodeIndex, to: targetNode };
}

const particleCount = Math.min(720, Math.max(320, Math.floor(edgeRecords.length * 0.18)));
const particleStates = [];
const offscreenPosition = new THREE.Vector3(999, 999, 999);
const particleAxis = new THREE.Vector3(1, 0, 0);
const particleDummy = new THREE.Object3D();
const particlePosition = new THREE.Vector3();
const particleDirection = new THREE.Vector3();
const particleNormal = new THREE.Vector3();
const particleBinormal = new THREE.Vector3();
const particleQuaternion = new THREE.Quaternion();
const particleColor = new THREE.Color();

function createRedBloodCellGeometry() {
  const geometry = new THREE.SphereGeometry(1, 18, 14);
  const positions = geometry.attributes.position;
  const vertex = new THREE.Vector3();
  for (let i = 0; i < positions.count; i += 1) {
    vertex.fromBufferAttribute(positions, i);
    const radial = Math.min(1, Math.sqrt(vertex.x * vertex.x + vertex.y * vertex.y));
    const concavity = 0.16 + 0.56 * Math.pow(radial, 1.25);
    vertex.x *= 1.08;
    vertex.y *= 0.74;
    vertex.z *= concavity * 0.62;
    positions.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }
  geometry.computeVertexNormals();
  return geometry;
}

const redCellGeometry = createRedBloodCellGeometry();
const redCellMaterial = new THREE.MeshPhysicalMaterial({
  color: 0xffffff,
  roughness: 0.36,
  metalness: 0.0,
  clearcoat: 0.12,
  sheen: 0.08,
  transparent: true,
  opacity: 0.94,
  depthWrite: false,
});
const particleCloud = new THREE.InstancedMesh(redCellGeometry, redCellMaterial, particleCount);
particleCloud.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
particleCloud.frustumCulled = false;
particleCloud.renderOrder = 4;
scene.add(particleCloud);

function hideParticle(index) {
  particleDummy.position.copy(offscreenPosition);
  particleDummy.scale.setScalar(1e-4);
  particleDummy.rotation.set(0, 0, 0);
  particleDummy.updateMatrix();
  particleCloud.setMatrixAt(index, particleDummy.matrix);
  particleCloud.setColorAt(index, particleColor.setRGB(0, 0, 0));
}

function activateParticle(index, transition, spawnT = Math.random() * 0.22, tempo = 0.82 + Math.random() * 0.42) {
  if (!particleStates[index]) {
    particleStates[index] = {};
  }
  particleStates[index].active = true;
  particleStates[index].cooldown = 0;
  particleStates[index].edge = transition.edge;
  particleStates[index].from = transition.from;
  particleStates[index].to = transition.to;
  particleStates[index].t = spawnT;
  particleStates[index].tempo = tempo;
  particleStates[index].age = 0;
  particleStates[index].roll = Math.random() * Math.PI * 2;
  particleStates[index].rollRate = -1.4 + Math.random() * 2.8;
  particleStates[index].laneAngle = Math.random() * Math.PI * 2;
  particleStates[index].laneDrift = 0.45 + Math.random() * 0.95;
  particleStates[index].laneRadius = Math.random() * Math.min(transition.edge.displayRadius * 0.28, data.visual.minDisplayRadius * 1.8);
  particleStates[index].wobble = 0.8 + Math.random() * 0.9;
  particleStates[index].waitTimer = 0;
  particleStates[index].waitDuration = 0;
  particleStates[index].lingerT = 0;
}

function deactivateParticle(index, cooldown = 0.12 + Math.random() * 0.32) {
  if (!particleStates[index]) {
    particleStates[index] = {};
  }
  particleStates[index].active = false;
  particleStates[index].cooldown = cooldown;
  hideParticle(index);
}

function respawnParticle(index) {
  const profile = selectSpawnProfile();
  if (profile) {
    const edge = selectEdgeForRespawn(profile);
    activateParticle(index, transitionFromEdge(edge));
    return;
  }

  const edge = selectEdgeForRespawn();
  activateParticle(index, transitionFromEdge(edge), Math.random());
}

function seedParticle(index) {
  respawnParticle(index);
}

function resetParticleSystem() {
  edgeOccupancy.fill(0);
  for (let i = 0; i < particleCount; i += 1) {
    particleStates[i] = {
      active: false,
      cooldown: 0,
      edge: null,
      from: 0,
      to: 0,
      t: 0,
      tempo: 1,
      age: 0,
      roll: 0,
      rollRate: 0,
      laneAngle: 0,
      laneDrift: 0,
      laneRadius: 0,
      wobble: 0,
      waitTimer: 0,
      waitDuration: 0,
      lingerT: 0,
    };
    hideParticle(i);
  }

  for (let i = 0; i < particleCount; i += 1) {
    seedParticle(i);
  }

  particleCloud.instanceMatrix.needsUpdate = true;
  if (particleCloud.instanceColor) {
    particleCloud.instanceColor.needsUpdate = true;
  }
}

resetParticleSystem();

function findDormantParticle(excludeIndex) {
  for (let i = 0; i < particleCount; i += 1) {
    if (i === excludeIndex) {
      continue;
    }
    const state = particleStates[i];
    if (state && !state.active && state.cooldown <= 0) {
      return i;
    }
  }
  return -1;
}

function spawnDormantParticleFromNode(nodeIndex, previousEdge, excludeIndex) {
  const dormantIndex = findDormantParticle(excludeIndex);
  if (dormantIndex === -1) {
    return;
  }

  const next = nextEdgeFromNode(nodeIndex, previousEdge, { preferDifferent: true });
  if (!next) {
    return;
  }
  activateParticle(dormantIndex, next, Math.random() * 0.08, 0.9 + Math.random() * 0.22);
}

function updateParticleInstance(index, state) {
  const start = points[state.from];
  const end = points[state.to];
  particlePosition.copy(start).lerp(end, state.t);
  particleDirection.copy(end).sub(start);
  if (particleDirection.lengthSq() < 1e-10) {
    hideParticle(index);
    return;
  }
  particleDirection.normalize();

  particleNormal.set(0, 1, 0);
  if (Math.abs(particleDirection.dot(particleNormal)) > 0.92) {
    particleNormal.set(0, 0, 1);
  }
  particleNormal.addScaledVector(particleDirection, -particleNormal.dot(particleDirection)).normalize();
  particleBinormal.crossVectors(particleDirection, particleNormal).normalize();

  const laneRadius = Math.min(
    state.edge.displayRadius * 0.34,
    Math.max(state.edge.displayRadius * 0.06, state.laneRadius),
  );
  const phase = state.laneAngle + state.age * state.laneDrift * (0.9 + state.edge.speedNorm * 1.4);
  particlePosition
    .addScaledVector(particleNormal, Math.cos(phase) * laneRadius)
    .addScaledVector(particleBinormal, Math.sin(phase) * laneRadius * 0.72);

  particleQuaternion.setFromUnitVectors(particleAxis, particleDirection);
  const baseSize = clamp(
    state.edge.displayRadius * (0.34 + state.edge.hematocrit * 0.46),
    data.particles.size * 0.82,
    state.edge.displayRadius * 0.78,
  );

  particleDummy.position.copy(particlePosition);
  particleDummy.quaternion.copy(particleQuaternion);
  particleDummy.rotateX(state.roll + state.age * state.rollRate);
  particleDummy.rotateZ(Math.sin(state.age * state.wobble * 3.2 + state.laneAngle) * 0.18);
  particleDummy.scale.set(baseSize * 1.08, baseSize * 0.74, baseSize * 0.56);
  particleDummy.updateMatrix();
  particleCloud.setMatrixAt(index, particleDummy.matrix);

  particleColor
    .set("#a61e16")
    .lerp(samplePalette(state.edge.speedNorm), 0.58 + state.edge.speedNorm * 0.26)
    .lerp(new THREE.Color("#ffe0d1"), 0.05 + state.edge.speedNorm * 0.08);
  particleCloud.setColorAt(index, particleColor);
}

function updateParticles(dt) {
  edgeOccupancy.fill(0);
  for (let i = 0; i < particleCount; i += 1) {
    const state = particleStates[i];
    if (state && state.active && state.edge) {
      edgeOccupancy[state.edge.index] += 1;
    }
  }

  for (let i = 0; i < particleCount; i += 1) {
    const state = particleStates[i];
    if (!state || !state.active) {
      if (state) {
        state.cooldown -= dt;
        if (state.cooldown <= 0) {
          respawnParticle(i);
        }
      }
      continue;
    }

    state.age += dt;
    if (state.waitTimer > 0) {
      state.waitTimer = Math.max(0, state.waitTimer - dt);
      const lingerProgress = state.waitDuration > 0
        ? 1.0 - state.waitTimer / state.waitDuration
        : 1.0;
      state.t = Math.min(0.045, state.lingerT + lingerProgress * 0.03);
      updateParticleInstance(i, state);
      continue;
    }

    let advance = dt * data.particles.sceneSpeedScale * (0.34 + state.edge.speedNorm * 2.24) * state.tempo;
    let localT = state.t + advance / Math.max(state.edge.length, 1e-6);

    while (localT >= 1.0) {
      localT -= 1.0;
      const arrivalNode = state.to;
      const profile = nodeFlowProfiles[arrivalNode];
      if (!profile || !profile.outgoingEdges.length) {
        deactivateParticle(i, 0.08 + Math.random() * 0.22);
        localT = 0;
        break;
      }

      if (profile.degree >= 3) {
        const deathProb = clamp(
          profile.deathFactor * (0.74 + profile.anomaly * 0.28) + (profile.hardAnomaly ? 0.22 : 0.0),
          0,
          profile.hardAnomaly ? 0.98 : 0.86,
        );
        if (Math.random() < deathProb) {
          deactivateParticle(i, 0.08 + Math.random() * 0.22);
          localT = 0;
          break;
        }
      }

      const next = nextEdgeFromNode(arrivalNode, state.edge);
      if (!next) {
        deactivateParticle(i, 0.08 + Math.random() * 0.22);
        localT = 0;
        break;
      }

      state.edge = next.edge;
      state.from = next.from;
      state.to = next.to;
      state.laneRadius = Math.min(state.laneRadius, state.edge.displayRadius * 0.34);
      state.laneAngle += (Math.random() - 0.5) * 0.72;
      localT = 0;

      if (profile.degree >= 3) {
        const waitScale = 0.5 + (1.0 - (profile.connectivityScore || 0)) * 0.7 + profile.anomaly * 0.45;
        state.waitDuration = RBC_WAIT_MIN
          + (RBC_WAIT_MAX - RBC_WAIT_MIN) * Math.min(waitScale, 1.0) * (0.75 + Math.random() * 0.25);
        state.waitTimer = state.waitDuration;
        state.lingerT = Math.random() * 0.012;
      }

      if (profile.degree >= 3 && profile.birthFactor > 0.03) {
        const spawnProb = clamp(
          profile.birthFactor * (0.34 + profile.connectivityScore * 0.66) * (profile.outgoingEdges.length > 1 ? 1.12 : 0.72),
          0,
          profile.hardAnomaly ? 0.2 : 0.74,
        );
        if (Math.random() < spawnProb) {
          spawnDormantParticleFromNode(arrivalNode, state.edge, i);
        }
      }
    }

    if (!state.active) {
      continue;
    }

    state.t = localT;
    updateParticleInstance(i, state);
  }

  particleCloud.instanceMatrix.needsUpdate = true;
  if (particleCloud.instanceColor) {
    particleCloud.instanceColor.needsUpdate = true;
  }
}

const state = {
  playing: true,
  autoCamera: true,
  progress: 0,
  speedMultiplier: 1.0,
  durationSeconds: data.visual.tourDurationSeconds,
  activePhaseKey: "top-overview",
  smoothPosition: camera.position.clone(),
  smoothLookAt: new THREE.Vector3(0, 0, 0),
  smoothUp: new THREE.Vector3(0, 1, 0),
};
const manualOrbit = {
  active: false,
  pointerId: null,
  theta: 0,
  phi: Math.PI * 0.5,
  radius: 1.8,
  target: new THREE.Vector3(0, 0, 0),
  spherical: new THREE.Spherical(),
  offset: new THREE.Vector3(),
  lastX: 0,
  lastY: 0,
};

function syncButtons() {
  playPauseBtn.textContent = state.playing ? "Pause tour" : "Play tour";
  playPauseBtn.classList.toggle("active", state.playing);
  const currentPhaseKey = getPhaseKeyFromProgress(state.progress);
  if (currentPhaseKey !== state.activePhaseKey) {
    state.activePhaseKey = currentPhaseKey;
  }
  phaseButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.phaseKey === state.activePhaseKey);
  });
}

function applyCameraTarget(progress) {
  state.progress = ((progress % 1) + 1) % 1;
  const target = getTourTargets(state.progress);
  state.smoothPosition.copy(target.position);
  state.smoothLookAt.copy(target.lookAt);
  state.smoothUp.copy(target.up);
  camera.position.copy(state.smoothPosition);
  camera.up.copy(state.smoothUp);
  camera.lookAt(state.smoothLookAt);
}

function syncManualOrbitFromCamera() {
  manualOrbit.target.copy(state.smoothLookAt);
  manualOrbit.offset.copy(camera.position).sub(manualOrbit.target);
  if (manualOrbit.offset.lengthSq() < 1e-8) {
    manualOrbit.offset.set(0.001, 0.001, 1.0);
  }
  manualOrbit.spherical.setFromVector3(manualOrbit.offset);
  manualOrbit.radius = clamp(manualOrbit.spherical.radius, 0.12, 8.0);
  manualOrbit.phi = clamp(manualOrbit.spherical.phi, 0.08, Math.PI - 0.08);
  manualOrbit.theta = manualOrbit.spherical.theta;
}

function applyManualOrbitCamera() {
  manualOrbit.spherical.set(
    manualOrbit.radius,
    manualOrbit.phi,
    manualOrbit.theta,
  );
  manualOrbit.offset.setFromSpherical(manualOrbit.spherical);
  state.smoothPosition.copy(manualOrbit.target).add(manualOrbit.offset);
  state.smoothLookAt.copy(manualOrbit.target);
  state.smoothUp.copy(worldUp);
  camera.position.copy(state.smoothPosition);
  camera.up.copy(state.smoothUp);
  camera.lookAt(state.smoothLookAt);
}

function enterManualCamera() {
  state.playing = false;
  state.autoCamera = false;
  syncManualOrbitFromCamera();
  renderer.domElement.style.cursor = manualOrbit.active ? "grabbing" : "grab";
  syncButtons();
}

function restoreAutoCamera() {
  state.autoCamera = true;
  manualOrbit.active = false;
  manualOrbit.pointerId = null;
  renderer.domElement.style.cursor = "grab";
}

syncButtons();

playPauseBtn.addEventListener("click", () => {
  state.playing = !state.playing;
  if (state.playing) {
    restoreAutoCamera();
  }
  syncButtons();
});

restartBtn.addEventListener("click", () => {
  restoreAutoCamera();
  applyCameraTarget(0);
  syncButtons();
});

phaseButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.playing = false;
    restoreAutoCamera();
    applyCameraTarget(Number(button.dataset.phaseProgress));
    syncButtons();
  });
});

const infoPanel = document.getElementById("infoPanel");
const infoIcon = document.getElementById("infoIcon");
const infoTitle = document.getElementById("infoTitle");
const infoSubtitle = document.getElementById("infoSubtitle");
const infoGrid = document.getElementById("infoGrid");
const infoCloseBtn = document.getElementById("infoCloseBtn");

function showNodeInfo(nodeIndex) {
  const profile = nodeFlowProfiles[nodeIndex];
  if (!profile) return;
  const point = profile.point;
  const coordText = `${point.x.toFixed(3)}, ${point.y.toFixed(3)}, ${point.z.toFixed(3)}`;
  const incomingTags = profile.incomingEdges.length
    ? profile.incomingEdges.map((edge) => `<span class="info-tag negative">Vessel #${edge.index}</span>`).join("")
    : `<span class="info-tag">None</span>`;
  const outgoingTags = profile.outgoingEdges.length
    ? profile.outgoingEdges.map((edge) => `<span class="info-tag positive">Vessel #${edge.index}</span>`).join("")
    : `<span class="info-tag">None</span>`;

  infoIcon.className = "info-icon node";
  infoIcon.textContent = "○";
  infoTitle.textContent = `Node #${nodeIndex}`;
  infoSubtitle.textContent = `(${coordText})`;

  infoGrid.innerHTML = `
    <div class="info-row">
      <span class="info-label">Coordinates</span>
      <span class="info-value">${coordText}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Incoming Vessel</span>
      <span class="info-value">${profile.incomingEdges.length}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Outgoing Vessel</span>
      <span class="info-value">${profile.outgoingEdges.length}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Total Inflow</span>
      <span class="info-value">${Math.abs(profile.incomingFlux).toFixed(2)} nL/min</span>
    </div>
    <div class="info-row">
      <span class="info-label">Total Outflow</span>
      <span class="info-value">${Math.abs(profile.outgoingFlux).toFixed(2)} nL/min</span>
    </div>
    <div class="info-row">
      <span class="info-label">Conservation Error</span>
      <span class="info-value ${profile.massConservationError > 0.25 ? 'error' : profile.massConservationError > 0.1 ? 'warning' : 'success'}">
        ${(profile.massConservationError * 100).toFixed(1)}%
      </span>
    </div>
    <div class="info-row">
      <span class="info-label">Connectivity</span>
      <span class="info-value ${profile.connectivityScore < 0.4 ? 'error' : profile.connectivityScore < 0.62 ? 'warning' : 'success'}">
        ${(profile.connectivityScore * 100).toFixed(0)}%
      </span>
    </div>
    <div class="info-row">
      <span class="info-label">Pressure</span>
      <span class="info-value">${profile.pressure == null ? 'N/A' : profile.pressure.toFixed(4)}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Pressure Consistency</span>
      <span class="info-value ${profile.pressureConsistency < 0.45 ? 'error' : profile.pressureConsistency < 0.75 ? 'warning' : 'success'}">
        ${(profile.pressureConsistency * 100).toFixed(0)}%
      </span>
    </div>
    <div class="info-row">
      <span class="info-label">Diagnostic</span>
      <span class="info-value ${profile.isPhysicallyValid ? 'success' : 'error'}">
        ${profile.diagnosticLabel}
      </span>
    </div>
    <div class="info-row">
      <span class="info-label">Status</span>
      <span class="info-value ${profile.isPhysicallyValid ? 'success' : 'error'}">
        ${profile.isPhysicallyValid ? 'Pass' : 'Check'}
      </span>
    </div>
    <div class="info-row">
      <span class="info-label">Pressure Conflict Edges</span>
      <span class="info-value ${profile.pressureConflictEdges.length ? 'warning' : 'success'}">
        ${profile.pressureConflictEdges.length}
      </span>
    </div>
    <div class="info-section">
      <h4>Incoming Vessel</h4>
      <div class="info-list">
        ${incomingTags}
      </div>
    </div>
    <div class="info-section">
      <h4>Outgoing Vessel</h4>
      <div class="info-list">
        ${outgoingTags}
      </div>
    </div>
  `;

  infoPanel.classList.add("visible");
}

function showVesselInfo(edgeIndex) {
  const edge = edgeRecords[edgeIndex];
  if (!edge) return;

  const flowDirection = `${edge.preferredStart} -> ${edge.preferredEnd}`;
  const averagePressure = edge.pressureStart == null || edge.pressureEnd == null
    ? null
    : (edge.pressureStart + edge.pressureEnd) * 0.5;

  infoIcon.className = "info-icon vessel";
  infoIcon.textContent = "↔";
  infoTitle.textContent = `Vessel #${edgeIndex}`;
  const startPoint = points[edge.a];
  const endPoint = points[edge.b];
  infoSubtitle.textContent = `Node #${edge.a} -> Node #${edge.b}`;

  infoGrid.innerHTML = `
    <div class="info-row">
      <span class="info-label">Start Node</span>
      <span class="info-value">Node #${edge.a}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Start Coordinates</span>
      <span class="info-value">${startPoint.x.toFixed(3)}, ${startPoint.y.toFixed(3)}, ${startPoint.z.toFixed(3)}</span>
    </div>
    <div class="info-row">
      <span class="info-label">End Node</span>
      <span class="info-value">Node #${edge.b}</span>
    </div>
    <div class="info-row">
      <span class="info-label">End Coordinates</span>
      <span class="info-value">${endPoint.x.toFixed(3)}, ${endPoint.y.toFixed(3)}, ${endPoint.z.toFixed(3)}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Diameter</span>
      <span class="info-value">${edge.diamUm.toFixed(2)} μm</span>
    </div>
    <div class="info-row">
      <span class="info-label">Radius</span>
      <span class="info-value">${edge.radiusUm.toFixed(2)} μm</span>
    </div>
    <div class="info-row">
      <span class="info-label">Flow Rate</span>
      <span class="info-value">
        ${edge.flowAbs.toFixed(2)} nL/min
      </span>
    </div>
    <div class="info-row">
      <span class="info-label">Flow Direction</span>
      <span class="info-value">${flowDirection}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Velocity</span>
      <span class="info-value">${edge.speed.toFixed(3)} mm/s</span>
    </div>
    <div class="info-row">
      <span class="info-label">Velocity Tint</span>
      <span class="info-value ${edge.speedNorm > 0.66 ? 'highlight' : edge.speedNorm > 0.33 ? 'warning' : 'success'}">
        ${(edge.speedNorm * 100).toFixed(0)}%
      </span>
    </div>
    <div class="info-row">
      <span class="info-label">Hematocrit</span>
      <span class="info-value">${edge.hematocrit.toFixed(3)}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Shear Stress</span>
      <span class="info-value">${edge.shearStress.toFixed(4)}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Pressure Start</span>
      <span class="info-value">${edge.pressureStart == null ? 'N/A' : edge.pressureStart.toFixed(4)}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Pressure End</span>
      <span class="info-value">${edge.pressureEnd == null ? 'N/A' : edge.pressureEnd.toFixed(4)}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Average Pressure</span>
      <span class="info-value">${averagePressure == null ? 'N/A' : averagePressure.toFixed(4)}</span>
    </div>
  `;

  infoPanel.classList.add("visible");
}

function closeInfoPanel() {
  infoPanel.classList.remove("visible");
}

infoCloseBtn.addEventListener("click", closeInfoPanel);

function handleClick(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1
  );

  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(mouse, camera);

  const branchIntersects = raycaster.intersectObject(branchMesh);
  if (branchIntersects.length > 0) {
    const index = branchIntersects[0].instanceId;
    if (index !== undefined && branchNodes[index] !== undefined) {
      showNodeInfo(branchNodes[index]);
      return;
    }
  }

  const intersects = raycaster.intersectObjects(vesselGroup.children, true);
  
  if (intersects.length > 0) {
    const point = intersects[0].point;
    let closestEdge = null;
    let closestDistance = Infinity;

    edgeRecords.forEach((edge, index) => {
      const p1 = points[edge.a];
      const p2 = points[edge.b];
      const dist = pointToSegmentDistance(point, p1, p2);
      if (dist < closestDistance) {
        closestDistance = dist;
        closestEdge = index;
      }
    });

    if (closestEdge !== null && closestDistance < 0.05) {
      showVesselInfo(closestEdge);
      return;
    }
  }

  closeInfoPanel();
}

function pointToSegmentDistance(point, p1, p2) {
  const v = p2.clone().sub(p1);
  const w = point.clone().sub(p1);
  const c1 = w.dot(v);
  if (c1 <= 0) return point.distanceTo(p1);
  const c2 = v.dot(v);
  if (c2 <= c1) return point.distanceTo(p2);
  const t = c1 / c2;
  const projection = p1.clone().add(v.multiplyScalar(t));
  return point.distanceTo(projection);
}

renderer.domElement.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) {
    return;
  }
  
  handleClick(event);
  enterManualCamera();
  manualOrbit.active = true;
  manualOrbit.pointerId = event.pointerId;
  manualOrbit.lastX = event.clientX;
  manualOrbit.lastY = event.clientY;
  renderer.domElement.setPointerCapture(event.pointerId);
  renderer.domElement.style.cursor = "grabbing";
});

renderer.domElement.addEventListener("pointermove", (event) => {
  if (!manualOrbit.active || event.pointerId !== manualOrbit.pointerId) {
    return;
  }

  const dx = event.clientX - manualOrbit.lastX;
  const dy = event.clientY - manualOrbit.lastY;
  manualOrbit.lastX = event.clientX;
  manualOrbit.lastY = event.clientY;

  manualOrbit.theta -= (dx / Math.max(window.innerWidth, 1)) * Math.PI * 1.8;
  manualOrbit.phi = clamp(
    manualOrbit.phi + (dy / Math.max(window.innerHeight, 1)) * Math.PI * 1.25,
    0.08,
    Math.PI - 0.08,
  );
  applyManualOrbitCamera();
});

function endManualPointer(event) {
  if (event.pointerId !== manualOrbit.pointerId) {
    return;
  }
  if (renderer.domElement.hasPointerCapture(event.pointerId)) {
    renderer.domElement.releasePointerCapture(event.pointerId);
  }
  manualOrbit.active = false;
  manualOrbit.pointerId = null;
  renderer.domElement.style.cursor = "grab";
}

renderer.domElement.addEventListener("pointerup", endManualPointer);
renderer.domElement.addEventListener("pointercancel", endManualPointer);
renderer.domElement.addEventListener("contextmenu", (event) => {
  event.preventDefault();
});
renderer.domElement.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    if (state.autoCamera) {
      enterManualCamera();
    }
    manualOrbit.radius = clamp(manualOrbit.radius * Math.exp(event.deltaY * 0.0012), 0.12, 8.0);
    applyManualOrbitCamera();
  },
  { passive: false },
);

speedSlider.addEventListener("input", (event) => {
  state.speedMultiplier = Number(event.target.value);
  speedReadout.textContent = `${state.speedMultiplier.toFixed(1)}x`;
});

const clock = new THREE.Clock();

function renderViewerFrame(dt, progressOverride = null) {
  if (typeof progressOverride === "number") {
    state.progress = ((progressOverride % 1) + 1) % 1;
  } else if (state.playing) {
    state.progress = (state.progress + (dt * state.speedMultiplier) / state.durationSeconds) % 1;
  }
  const targets = getTourTargets(state.progress);

  if (state.autoCamera) {
    const follow = 1 - Math.exp(-dt * 6.0);
    const orient = 1 - Math.exp(-dt * 5.0);
    state.smoothPosition.lerp(targets.position, follow);
    state.smoothLookAt.lerp(targets.lookAt, orient);
    state.smoothUp.lerp(targets.up, orient).normalize();

    camera.position.copy(state.smoothPosition);
    camera.up.copy(state.smoothUp);
    camera.lookAt(state.smoothLookAt);
  }

  vesselMaterial.opacity += (targets.vesselOpacity - vesselMaterial.opacity) * (1 - Math.exp(-dt * 4.0));
  coreLineMaterial.opacity += (Math.max(0.05, targets.vesselOpacity * 0.28) - coreLineMaterial.opacity) * (1 - Math.exp(-dt * 3.2));
  pathMaterial.opacity += (targets.pathOpacity - pathMaterial.opacity) * (1 - Math.exp(-dt * 4.5));
  scene.fog.density += (targets.fogDensity - scene.fog.density) * (1 - Math.exp(-dt * 3.8));
  hubHalo.material.opacity += (targets.hubOpacity - hubHalo.material.opacity) * (1 - Math.exp(-dt * 3.8));
  hubHalo.scale.setScalar(1.0 + Math.sin(state.progress * Math.PI * 12.0) * 0.04);
  dust.rotation.y += dt * 0.02;

  updateBranchMarkers(state.autoCamera ? state.smoothLookAt : camera.position);
  updateParticles(dt);
  syncButtons();
  renderer.render(scene, camera);
  return targets;
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.04);
  renderViewerFrame(dt);
}

function createSeededRandom(seed) {
  let value = (seed >>> 0) || 1;
  return function seededRandom() {
    value += 0x6D2B79F5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let exportRandomGenerator = null;

function withExportRandom(callback) {
  if (!exportRandomGenerator) {
    return callback();
  }
  const originalRandom = Math.random;
  Math.random = () => exportRandomGenerator();
  try {
    return callback();
  } finally {
    Math.random = originalRandom;
  }
}

window.viewerExport = {
  ready: true,
  reset(options = {}) {
    const seed = Number.isFinite(options.seed) ? options.seed : 12345;
    exportRandomGenerator = createSeededRandom(seed);
    state.playing = false;
    restoreAutoCamera();
    closeInfoPanel();
    dust.rotation.set(0, 0, 0);
    hubHalo.scale.setScalar(1.0);
    withExportRandom(() => {
      resetParticleSystem();
    });
    applyCameraTarget(Number.isFinite(options.startProgress) ? options.startProgress : 0);
    renderViewerFrame(0, state.progress);
    return true;
  },
  renderFrame(frameIndex, totalFrames, options = {}) {
    if (!exportRandomGenerator) {
      this.reset(options);
    }
    const safeTotal = Math.max(1, totalFrames);
    const computedProgress = safeTotal <= 1 ? 0 : (frameIndex % safeTotal) / safeTotal;
    const progress = Number.isFinite(options.progress) ? options.progress : computedProgress;
    const durationSeconds = Number.isFinite(options.durationSeconds) ? options.durationSeconds : state.durationSeconds;
    const dt = Number.isFinite(options.dt) ? options.dt : durationSeconds / safeTotal;
    return withExportRandom(() => {
      state.playing = false;
      restoreAutoCamera();
      renderViewerFrame(dt, progress);
      return renderer.domElement.toDataURL("image/png");
    });
  },
  snapshot(options = {}) {
    if (!exportRandomGenerator) {
      this.reset(options);
    }
    return renderer.domElement.toDataURL("image/png");
  },
};

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  if (!state.autoCamera) {
    applyManualOrbitCamera();
  }
});

applyCameraTarget(0);
syncButtons();
animate();
</script>
</body>
</html>"""


def main():
    points_raw, cells, point_scalars, cell_scalars = read_legacy_vtk_polydata(VTK_FILE)

    velocity = cell_scalars.get("velocity_mm_s")
    if velocity is None:
        raise RuntimeError("velocity_mm_s not found in CELL_DATA.")
    flow_nl_min = cell_scalars.get("flow_nl_min")
    hematocrit = cell_scalars.get("hematocrit")
    shear_stress = cell_scalars.get("shear_stress")
    point_pressure = point_scalars.get("pressure")

    radius_um = cell_scalars.get("radius_um")
    diam_um = cell_scalars.get("diam_um")
    if radius_um is None:
        if diam_um is not None:
            radius_um = diam_um / 2.0
        else:
            radius_um = np.full(len(cells), 4.0, dtype=float)
    if diam_um is None:
        diam_um = radius_um * 2.0

    points_norm, center, scale = normalize_points(points_raw)
    segments = build_segments(
        cells,
        points_raw,
        velocity,
        radius_um,
        flow_nl_min=flow_nl_min,
        hematocrit=hematocrit,
        diam_um=diam_um,
        shear_stress=shear_stress,
        point_pressure=point_pressure,
    )
    adjacency = build_adjacency(segments)

    speed = np.abs(velocity)
    robust_max = float(np.percentile(speed, 98))
    if robust_max <= 0:
        robust_max = float(np.max(speed))
    if robust_max <= 0:
        robust_max = 1.0

    scene_data = colorized_data(points_norm, segments, robust_max, scale)
    render_strands = build_render_strands(points_norm, segments, robust_max, scale)
    tour = choose_tour(points_raw, points_norm, adjacency, segments, scale)

    median_edge_length_norm = scene_data["median_edge_length_norm"]
    radius_boost = 1.85 if scene_data["median_radius_norm"] <= 0 else max(1.4, min(2.3, median_edge_length_norm / scene_data["median_radius_norm"] * 0.55))
    tour_duration = float(np.clip(tour["path_length_raw"] / 185.0, 30.0, 44.0))

    viewer_data = {
        "points": scene_data["points"],
        "edges": scene_data["edges"],
        "pointPressure": None if point_pressure is None else [round(float(v), 6) for v in point_pressure],
        "vesselStrands": render_strands,
        "cameraPath": [[round(float(v), 6) for v in row] for row in tour["camera_path_norm"]],
        "focusPath": [[round(float(v), 6) for v in row] for row in tour["focus_path_norm"]],
        "branchPoints": [[round(float(v), 6) for v in row] for row in tour["branch_points"]],
        "branchNodes": [int(v) for v in tour["branch_nodes"]],
        "branchStrengths": [round(float(v), 6) for v in tour["branch_strengths"]],
        "hubPoint": [round(float(v), 6) for v in tour["hub_point"]],
        "stats": {
            "pointCount": int(len(points_raw)),
            "cellCount": int(len(cells)),
            "segmentCount": int(len(segments)),
            "branchCount": int(tour["branch_count"]),
            "endpointCount": int(tour["endpoint_count"]),
            "velocityMin": float(np.min(velocity)),
            "velocityMax": float(np.max(velocity)),
            "robustMax": robust_max,
            "pressureMin": None if point_pressure is None else float(np.min(point_pressure)),
            "pressureMax": None if point_pressure is None else float(np.max(point_pressure)),
            "tourLengthRaw": float(tour["path_length_raw"]),
            "scale": scale,
        },
        "visual": {
            "radiusBoost": round(float(radius_boost), 6),
            "minDisplayRadius": round(float(scene_data["min_display_radius"]), 6),
            "tourDurationSeconds": round(tour_duration, 3),
            "cameraClearance": round(float(tour["camera_clearance"]), 6),
        },
        "particles": {
            "sceneSpeedScale": round(float(1000.0 / scale * 0.17), 6),
            "size": round(float(scene_data["min_display_radius"] * 2.1), 6),
        },
        "fields": {
            "pointScalars": sorted(point_scalars.keys()),
            "cellScalars": sorted(cell_scalars.keys()),
        },
    }

    html = HTML_TEMPLATE.replace("__VIEWER_DATA__", json.dumps(viewer_data, separators=(",", ":")))
    OUT_HTML.write_text(html, encoding="utf-8")

    summary = [
        f"Wrote {OUT_HTML}",
        f"Points: {len(points_raw)} | Cells: {len(cells)} | Segments: {len(segments)}",
        f"Renderable strands: {len(render_strands)}",
        f"Branch points: {tour['branch_count']} | Endpoints: {tour['endpoint_count']}",
        f"Velocity range: {float(np.min(velocity)):.4g} to {float(np.max(velocity)):.4g} mm/s",
        f"Robust |velocity| max (98th percentile): {robust_max:.4g} mm/s",
        f"Camera path length: {tour['path_length_raw']:.1f} raw units",
        f"Point fields: {', '.join(sorted(point_scalars.keys())) or 'none'}",
        f"Cell fields: {', '.join(sorted(cell_scalars.keys())) or 'none'}",
    ]
    print("\n".join(summary))


if __name__ == "__main__":
    main()
