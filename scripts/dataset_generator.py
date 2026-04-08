#!/usr/bin/env python3
"""
建筑立面点云数据集生成器 v4 (自适应密度采样)

Copied from semantic_guided_completion_proposal for convenience.
See original for full documentation.

Usage:
    python scripts/dataset_generator.py "path/to/LOD3_DATA" --recursive --mode both
    python scripts/dataset_generator.py "path/to/LOD3_DATA" --recursive --mode synthetic --density 1500
"""

# This script is a direct copy of the original dataset_generator.py.
# Please refer to the afterglowY/semantic_guided_completion_proposal repo
# for the canonical version.
#
# To use: pip install numpy trimesh laspy rtree scipy
# Output format:
#   {scene_id}_synthetic.npz: points(N,3), semantic_id(N,), colors(N,3), mesh_area
#   {scene_id}_annotated.npz: points(N,3), semantic_id(N,), confidence_dist(N,), colors(N,3)

import json, argparse, os, sys, time, traceback
import numpy as np
import trimesh
from trimesh.proximity import ProximityQuery
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SEMANTIC_LABELS = {
    0: 'wall', 1: 'window', 2: 'door', 3: 'roof',
    4: 'banister', 5: 'equipment', 6: 'sign', 7: 'awning',
    8: 'stairs', 9: 'balcony', 10: 'eave', 11: 'other'
}
SEMANTIC_COLORS_RGB = {
    0:(30,136,229), 1:(67,160,71), 2:(229,57,53), 3:(142,36,170),
    4:(251,140,0), 5:(0,172,193), 6:(253,216,53), 7:(216,27,96),
    8:(109,76,65), 9:(0,137,123), 10:(94,53,177), 11:(117,117,117),
}
SMALL_COMPONENT_LABELS = {1, 2, 5, 6, 7}

def discover_scene(folder):
    result = {'folder': folder, 'scene_id': folder.name, 'obj': None, 'json': None, 'las': None}
    for name in ['trimesh.obj', 'trimesh']:
        if (folder / name).exists(): result['obj'] = str(folder / name); break
    for name in ['trimesh.JSON', 'trimesh.json']:
        if (folder / name).exists(): result['json'] = str(folder / name); break
    for ext in ['*.las', '*.laz']:
        found = list(folder.glob(ext))
        if found: result['las'] = str(found[0]); break
    return result

def discover_scenes_recursive(root):
    scenes = []
    for dirpath, _, filenames in os.walk(root):
        if 'trimesh.json' in [f.lower() for f in filenames]:
            scene = discover_scene(Path(dirpath))
            if scene['obj'] and scene['json']: scenes.append(scene)
    return sorted(scenes, key=lambda s: s['scene_id'])

def load_mesh(obj_path):
    ext = Path(obj_path).suffix.lower()
    kwargs = dict(process=False, force='mesh')
    if not ext or ext not in ('.obj', '.stl', '.ply', '.off'): kwargs['file_type'] = 'obj'
    return trimesh.load(obj_path, **kwargs)

def build_face_labels(json_path, num_faces):
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    face_labels = np.full(num_faces, 11, dtype=np.int32)
    for poly in data['polygon']:
        obj_id = int(poly['OBJ_ID']); label = int(poly.get('label', 11))
        if 0 <= obj_id < num_faces: face_labels[obj_id] = label
    return face_labels

def compute_num_points(mesh_area, density, min_points, max_points, fixed=None):
    if fixed is not None: return fixed
    return min(max(int(mesh_area * density), min_points), max_points)

def refine_small_components(scan_points, semantic_id, distances, mesh, face_labels, k_refine=6):
    from scipy.spatial import cKDTree
    wall_mask = (semantic_id == 0) & (distances < 0.5)
    wall_indices = np.where(wall_mask)[0]
    if len(wall_indices) == 0: return semantic_id
    small_face_ids = np.where(np.isin(face_labels, list(SMALL_COMPONENT_LABELS)))[0]
    if len(small_face_ids) == 0: return semantic_id
    small_centroids = mesh.triangles[small_face_ids].mean(axis=1)
    tree = cKDTree(small_centroids)
    wall_pts = scan_points[wall_indices].astype(np.float64)
    dists_to_small, idx_in_small = tree.query(wall_pts, k=1)
    margin = np.maximum(distances[wall_indices] * 0.5, 0.15)
    close_enough = dists_to_small < (distances[wall_indices] + margin)
    refined = semantic_id.copy()
    for i, wi in enumerate(wall_indices):
        if close_enough[i]: refined[wi] = face_labels[small_face_ids[idx_in_small[i]]]
    n_changed = (refined != semantic_id).sum()
    if n_changed > 0: print(f"  边界修正: {n_changed:,} 点 wall→小构件")
    return refined

def run_synthetic(mesh, face_labels, num_points, output_path):
    print(f"  [synthetic] 采样 {num_points:,} 点...")
    points, face_indices = trimesh.sample.sample_surface(mesh, count=num_points)
    points = np.array(points, dtype=np.float32)
    semantic_id = face_labels[face_indices]
    colors = np.array([SEMANTIC_COLORS_RGB.get(int(s), (117,117,117)) for s in semantic_id], dtype=np.uint8)
    np.savez_compressed(output_path, points=points, semantic_id=semantic_id.astype(np.int32),
                        colors=colors, mesh_area=np.float32(mesh.area))
    _print_stats(semantic_id, output_path)

def run_annotate(mesh, face_labels, las_path, distance_threshold, other_label, output_path):
    print(f"  [annotate] 加载 LAS: {Path(las_path).name}")
    import laspy
    las = laspy.read(las_path)
    scan_points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    N = len(scan_points); print(f"  点数: {N:,}")
    pq = ProximityQuery(mesh)
    batch_size = 500_000
    all_distances = np.empty(N, dtype=np.float32)
    all_face_ids = np.empty(N, dtype=np.int64)
    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        _, dists, fids = pq.on_surface(scan_points[start:end])
        all_distances[start:end] = dists; all_face_ids[start:end] = fids
    semantic_id = face_labels[all_face_ids]
    semantic_id[all_distances > distance_threshold] = other_label
    semantic_id = refine_small_components(scan_points, semantic_id, all_distances, mesh, face_labels)
    colors = np.array([SEMANTIC_COLORS_RGB.get(int(s), (117,117,117)) for s in semantic_id], dtype=np.uint8)
    np.savez_compressed(output_path, points=scan_points.astype(np.float32),
                        semantic_id=semantic_id.astype(np.int32),
                        confidence_dist=all_distances, colors=colors)
    _print_stats(semantic_id, output_path)

def _print_stats(semantic_id, output_path):
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    unique, counts = np.unique(semantic_id, return_counts=True)
    total = len(semantic_id)
    parts = [f"{SEMANTIC_LABELS.get(int(u), f'?{u}')}: {c/total*100:.1f}%" for u, c in zip(unique, counts)]
    print(f"  → {Path(output_path).name} ({size_mb:.1f}MB) [{', '.join(parts)}]")

def process_scene(scene, mode, density, min_points, max_points, fixed_num_points,
                  distance_threshold, other_label, export_ply, output_dir=None,
                  skip_existing=False, protect_annotated=True):
    sid = scene['scene_id']; t0 = time.time()
    out_dir = Path(output_dir) if output_dir else scene['folder']
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    synth_path = out_dir / f"{sid}_synthetic.npz"
    ann_path = out_dir / f"{sid}_annotated.npz"
    run_synth = mode in ('synthetic', 'both')
    run_ann = mode in ('annotate', 'both') and scene.get('las')
    if skip_existing:
        if (not run_synth or synth_path.exists()) and (not run_ann or ann_path.exists()):
            return {'scene_id': sid, 'status': 'skipped'}
    if protect_annotated and run_ann and ann_path.exists(): run_ann = False
    if not run_synth and not run_ann: return {'scene_id': sid, 'status': 'skipped'}
    if not scene['obj'] or not scene['json']: return {'scene_id': sid, 'status': 'missing_files'}
    mesh = load_mesh(scene['obj'])
    face_labels = build_face_labels(scene['json'], len(mesh.faces))
    num_points = compute_num_points(mesh.area, density, min_points, max_points, fixed_num_points)
    if run_synth: run_synthetic(mesh, face_labels, num_points, str(synth_path))
    if run_ann: run_annotate(mesh, face_labels, scene['las'], distance_threshold, other_label, str(ann_path))
    return {'scene_id': sid, 'status': 'success', 'time': time.time()-t0,
            'num_points': num_points, 'area': mesh.area}

def main():
    parser = argparse.ArgumentParser(description='建筑立面点云数据集生成器 v4')
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--recursive', '-r', action='store_true')
    parser.add_argument('--mode', choices=['synthetic', 'annotate', 'both'], default='both')
    parser.add_argument('--density', type=int, default=1500)
    parser.add_argument('--min_points', type=int, default=100_000)
    parser.add_argument('--max_points', type=int, default=5_000_000)
    parser.add_argument('--num_points', type=int, default=None)
    parser.add_argument('--distance_threshold', type=float, default=1.0)
    parser.add_argument('--other_label', type=int, default=11)
    parser.add_argument('--ply', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--no_protect_annotated', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    scenes = []
    if args.recursive:
        for root in args.folders:
            scenes.extend(discover_scenes_recursive(Path(root)))
    else:
        for folder in args.folders:
            scene = discover_scene(Path(folder))
            if scene['obj'] and scene['json']: scenes.append(scene)
    if not scenes: print("未找到场景"); return
    print(f"共 {len(scenes)} 个场景")
    for scene in scenes:
        try:
            process_scene(scene, args.mode, args.density, args.min_points, args.max_points,
                          args.num_points, args.distance_threshold, args.other_label,
                          args.ply, args.output_dir, args.skip_existing,
                          not args.no_protect_annotated)
        except Exception as e:
            print(f"Error {scene['scene_id']}: {e}"); traceback.print_exc()

if __name__ == '__main__':
    main()
