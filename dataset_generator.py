# -*- coding: utf-8 -*-
"""
建筑立面点云数据集生成器 v4 (自适应密度采样)

v3→v4 变更:
  - synthetic 采样改为基于 mesh 表面积的自适应密度，不再固定50万点
  - 新增 --density (点/m²)、--min_points、--max_points 参数
  - 新增 --protect_annotated: 默认开启，防止覆盖已有的 annotated NPZ
  - 保留旧的 --num_points 参数作为手动覆盖（优先级高于 --density）

=== 只重新生成 synthetic（不碰 annotated）===
  python facade_dataset_generator_v4.py "C:/LOD3_DATA" --recursive --mode synthetic

=== 全量生成（首次使用）===
  python facade_dataset_generator_v4.py "C:/LOD3_DATA" --recursive --mode both

=== 其他选项 ===
  --density 1500              每平方米采样点数 (默认1500)
  --min_points 100000         最少采样点数 (默认100K, 防止极小建筑太稀疏)
  --max_points 5000000        最多采样点数 (默认5M, 防止巨型建筑吃内存)
  --num_points N              手动指定固定点数 (设置后忽略 --density)
  --distance_threshold 1.0    annotate 最大可信距离(m)
  --ply                       同时导出 PLY
  --output_dir DIR            指定输出目录
  --skip_existing             跳过已有输出的场景
  --no_protect_annotated      允许覆盖已有 annotated 文件
  --workers N                 并行进程数 (默认1)

依赖: pip install numpy trimesh laspy rtree scipy
"""

import json
import argparse
import os
import sys
import time
import traceback
import numpy as np
import trimesh
from trimesh.proximity import ProximityQuery
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


# ==================== 语义定义 ====================

SEMANTIC_LABELS = {
    0: 'wall',      1: 'window',    2: 'door',      3: 'roof',
    4: 'banister',  5: 'equipment', 6: 'sign',      7: 'awning',
    8: 'stairs',    9: 'balcony',   10: 'eave',     11: 'other'
}

SEMANTIC_COLORS_RGB = {
    0:  (30, 136, 229),   1:  (67, 160, 71),    2:  (229, 57, 53),
    3:  (142, 36, 170),   4:  (251, 140, 0),    5:  (0, 172, 193),
    6:  (253, 216, 53),   7:  (216, 27, 96),    8:  (109, 76, 65),
    9:  (0, 137, 123),    10: (94, 53, 177),    11: (117, 117, 117),
}

# 小构件类别（窗/门/设备/标牌/雨篷），annotate 时做边界修正
SMALL_COMPONENT_LABELS = {1, 2, 5, 6, 7}


# ==================== 文件发现 ====================

def discover_scene(folder: Path) -> dict:
    """在单个文件夹中查找 OBJ / JSON / LAS"""
    result = {'folder': folder, 'scene_id': folder.name,
              'obj': None, 'json': None, 'las': None}

    for name in ['trimesh.obj', 'trimesh']:
        if (folder / name).exists():
            result['obj'] = str(folder / name)
            break

    for name in ['trimesh.JSON', 'trimesh.json']:
        if (folder / name).exists():
            result['json'] = str(folder / name)
            break

    for ext in ['*.las', '*.laz']:
        found = list(folder.glob(ext))
        if found:
            result['las'] = str(found[0])
            break

    return result


def discover_scenes_recursive(root: Path) -> list:
    """递归扫描目录树，找到所有包含 trimesh.JSON 的场景文件夹。"""
    scenes = []
    for dirpath, dirnames, filenames in os.walk(root):
        lower_files = [f.lower() for f in filenames]
        if 'trimesh.json' in lower_files:
            scene = discover_scene(Path(dirpath))
            if scene['obj'] and scene['json']:
                scenes.append(scene)
    return sorted(scenes, key=lambda s: s['scene_id'])


# ==================== 模型加载 ====================

def load_mesh(obj_path: str) -> trimesh.Trimesh:
    """加载 OBJ，process=False 保持面片顺序与 OBJ_ID 一致。"""
    ext = Path(obj_path).suffix.lower()
    kwargs = dict(process=False, force='mesh')
    if not ext or ext not in ('.obj', '.stl', '.ply', '.off'):
        kwargs['file_type'] = 'obj'
    return trimesh.load(obj_path, **kwargs)


def build_face_labels(json_path: str, num_faces: int) -> np.ndarray:
    """从 JSON 构建 face_index → semantic_label 映射。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    face_labels = np.full(num_faces, 11, dtype=np.int32)
    for poly in data['polygon']:
        obj_id = int(poly['OBJ_ID'])
        label = int(poly.get('label', 11))
        if 0 <= obj_id < num_faces:
            face_labels[obj_id] = label

    return face_labels


# ==================== 自适应采样点数计算 ====================

def compute_num_points(mesh_area: float, density: int,
                       min_points: int, max_points: int,
                       fixed_num_points: int = None) -> int:
    """
    根据 mesh 表面积计算采样点数。

    Args:
        mesh_area: mesh 表面积 (m²)
        density: 每平方米采样点数
        min_points: 最少采样点数
        max_points: 最多采样点数
        fixed_num_points: 手动指定固定点数 (设置后忽略自适应)

    Returns:
        实际采样点数
    """
    if fixed_num_points is not None:
        return fixed_num_points

    n = int(mesh_area * density)
    n = max(n, min_points)
    n = min(n, max_points)
    return n


# ==================== 小构件边界修正 ====================

def refine_small_components(scan_points: np.ndarray,
                            semantic_id: np.ndarray,
                            distances: np.ndarray,
                            mesh: trimesh.Trimesh,
                            face_labels: np.ndarray,
                            k_refine: int = 6) -> np.ndarray:
    """
    修正窗户/门等小构件的边界标注。

    问题: 几何最近邻匹配时，位于窗框边缘的扫描点可能被分配到相邻的
    大面积墙面三角形上，导致本应是 window 的点被标为 wall。

    修正策略:
      对每个被标为 wall 的点，检查其在 mesh 上的 K 个最近面片。
      如果其中存在小构件面片（window/door 等），且该面片到该点的
      距离与最近面片距离之差小于容差(margin)，则改为小构件标签。
    """
    from scipy.spatial import cKDTree

    wall_mask = (semantic_id == 0) & (distances < 0.5)
    wall_indices = np.where(wall_mask)[0]

    if len(wall_indices) == 0:
        return semantic_id

    small_face_mask = np.isin(face_labels, list(SMALL_COMPONENT_LABELS))
    small_face_ids = np.where(small_face_mask)[0]

    if len(small_face_ids) == 0:
        return semantic_id

    small_centroids = mesh.triangles[small_face_ids].mean(axis=1)
    tree = cKDTree(small_centroids)

    wall_pts = scan_points[wall_indices].astype(np.float64)
    dists_to_small, idx_in_small = tree.query(wall_pts, k=1)

    margin = np.maximum(distances[wall_indices] * 0.5, 0.15)
    close_enough = dists_to_small < (distances[wall_indices] + margin)

    refined = semantic_id.copy()
    for i, wi in enumerate(wall_indices):
        if close_enough[i]:
            fid = small_face_ids[idx_in_small[i]]
            refined[wi] = face_labels[fid]

    n_changed = (refined != semantic_id).sum()
    if n_changed > 0:
        print(f"  边界修正: {n_changed:,} 点 wall→小构件")

    return refined


# ==================== synthetic ====================

def run_synthetic(mesh, face_labels, num_points, output_path):
    """从模型表面均匀采样，生成理想合成点云。"""
    print(f"  [synthetic] 采样 {num_points:,} 点 "
          f"(面积={mesh.area:.0f}m², 密度={num_points/mesh.area:.0f}点/m²)...")
    points, face_indices = trimesh.sample.sample_surface(mesh, count=num_points)
    points = np.array(points, dtype=np.float32)
    semantic_id = face_labels[face_indices]

    colors = np.array(
        [SEMANTIC_COLORS_RGB.get(int(s), (117, 117, 117)) for s in semantic_id],
        dtype=np.uint8)

    np.savez_compressed(output_path,
                        points=points,
                        semantic_id=semantic_id.astype(np.int32),
                        colors=colors,
                        mesh_area=np.float32(mesh.area))
    _print_stats(semantic_id, output_path)


# ==================== annotate ====================

def run_annotate(mesh, face_labels, las_path, distance_threshold,
                 other_label, output_path):
    """将真实扫描 LAS 投影到模型表面，赋予语义标签。"""
    print(f"  [annotate] 加载 LAS: {Path(las_path).name}")
    import laspy
    las = laspy.read(las_path)
    scan_points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    N = len(scan_points)
    print(f"  点数: {N:,}")

    print(f"  投影 (阈值={distance_threshold}m)...")
    pq = ProximityQuery(mesh)

    batch_size = 500_000
    all_distances = np.empty(N, dtype=np.float32)
    all_face_ids = np.empty(N, dtype=np.int64)

    t0 = time.time()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        _, dists, fids = pq.on_surface(scan_points[start:end])
        all_distances[start:end] = dists
        all_face_ids[start:end] = fids
        elapsed = time.time() - t0
        speed = end / elapsed if elapsed > 0 else 0
        eta = (N - end) / speed if speed > 0 else 0
        print(f"  {end:,}/{N:,} ({end/N*100:.0f}%) "
              f"{speed/1000:.0f}K/s ETA={eta:.0f}s", end='\r')
    print()

    # 基础标签分配
    semantic_id = face_labels[all_face_ids]
    far_mask = all_distances > distance_threshold
    semantic_id[far_mask] = other_label

    # 小构件边界修正
    semantic_id = refine_small_components(
        scan_points, semantic_id, all_distances, mesh, face_labels)

    # 生成颜色
    colors = np.array(
        [SEMANTIC_COLORS_RGB.get(int(s), (117, 117, 117)) for s in semantic_id],
        dtype=np.uint8)

    n_ok = (~far_mask).sum()
    n_far = far_mask.sum()
    print(f"  可信: {n_ok:,} ({n_ok/N*100:.1f}%)  "
          f"不可信: {n_far:,} ({n_far/N*100:.1f}%)")
    print(f"  距离: mean={all_distances.mean():.3f} "
          f"p50={np.median(all_distances):.3f} "
          f"p95={np.percentile(all_distances, 95):.3f} "
          f"max={all_distances.max():.3f}")

    np.savez_compressed(output_path,
                        points=scan_points.astype(np.float32),
                        semantic_id=semantic_id.astype(np.int32),
                        confidence_dist=all_distances,
                        colors=colors)
    _print_stats(semantic_id, output_path)


# ==================== PLY 导出 ====================

def save_ply(npz_path: str):
    """NPZ → 带语义颜色的 PLY"""
    data = np.load(npz_path)
    pts = data['points']
    N = len(pts)

    if 'semantic_id' in data:
        sem = data['semantic_id'].astype(np.int32)
    elif 'semantics' in data:
        sem = data['semantics'].astype(np.int32)
    else:
        sem = np.full(N, 11, dtype=np.int32)

    if 'colors' in data:
        colors = data['colors'].astype(np.float32)
    else:
        colors = np.array(
            [SEMANTIC_COLORS_RGB.get(int(s), (117, 117, 117)) for s in sem],
            dtype=np.float32)

    ply_path = str(Path(npz_path).with_suffix('.ply'))

    with open(ply_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int semantic\nend_header\n")

    combined = np.column_stack([pts, colors, sem.astype(np.float32)])
    with open(ply_path, 'a') as f:
        np.savetxt(f, combined, fmt='%.4f %.4f %.4f %d %d %d %d')

    size_mb = os.path.getsize(ply_path) / 1024 / 1024
    print(f"  → {Path(ply_path).name} ({size_mb:.1f}MB)")


def _print_stats(semantic_id, output_path):
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    unique, counts = np.unique(semantic_id, return_counts=True)
    total = len(semantic_id)
    parts = [f"{SEMANTIC_LABELS.get(int(u), f'?{u}')}: {c/total*100:.1f}%"
             for u, c in zip(unique, counts)]
    print(f"  → {Path(output_path).name} ({size_mb:.1f}MB) [{', '.join(parts)}]")


# ==================== 场景处理 ====================

def process_scene(scene: dict, mode: str,
                  density: int, min_points: int, max_points: int,
                  fixed_num_points: int,
                  distance_threshold: float, other_label: int,
                  export_ply: bool, output_dir: str = None,
                  skip_existing: bool = False,
                  protect_annotated: bool = True) -> dict:
    """处理单个场景，返回结果字典"""
    sid = scene['scene_id']
    t0 = time.time()

    out_dir = Path(output_dir) if output_dir else scene['folder']
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_path = out_dir / f"{sid}_synthetic.npz"
    ann_path = out_dir / f"{sid}_annotated.npz"

    # 判断是否需要跑 synthetic
    run_synth = mode in ('synthetic', 'both')
    # 判断是否需要跑 annotate
    run_ann = mode in ('annotate', 'both') and scene.get('las')

    # skip_existing 检查
    if skip_existing:
        synth_ok = synth_path.exists() if run_synth else True
        ann_ok = ann_path.exists() if run_ann else True
        if synth_ok and ann_ok:
            return {'scene_id': sid, 'status': 'skipped'}

    # protect_annotated: 即使 mode=both, 也跳过已有的 annotated
    if protect_annotated and run_ann and ann_path.exists():
        print(f"  ⓘ {sid}: annotated 已存在, 跳过 (--no_protect_annotated 可覆盖)")
        run_ann = False

    # 如果两个都不需要跑, 直接返回
    if not run_synth and not run_ann:
        return {'scene_id': sid, 'status': 'skipped'}

    print(f"\n{'='*60}")
    print(f"场景: {sid}")
    print(f"{'='*60}")

    if not scene['obj'] or not scene['json']:
        print(f"  ✗ 缺少文件，跳过")
        return {'scene_id': sid, 'status': 'missing_files'}

    mesh = load_mesh(scene['obj'])
    face_labels = build_face_labels(scene['json'], len(mesh.faces))

    # 计算自适应采样点数
    num_points = compute_num_points(
        mesh.area, density, min_points, max_points, fixed_num_points)

    print(f"  模型: {len(mesh.faces)} 面, {mesh.area:.0f}m²")
    print(f"  采样: {num_points:,} 点 "
          f"(密度={num_points/mesh.area:.0f}点/m², "
          f"预估间距={1/np.sqrt(num_points/mesh.area)*100:.2f}cm)")

    unique, counts = np.unique(face_labels, return_counts=True)
    parts = [f"{SEMANTIC_LABELS.get(int(u),'?')}:{c}" for u, c in zip(unique, counts)]
    print(f"  标签: {', '.join(parts)}")

    if run_synth:
        run_synthetic(mesh, face_labels, num_points, str(synth_path))
        if export_ply:
            save_ply(str(synth_path))

    if run_ann:
        run_annotate(mesh, face_labels, scene['las'],
                     distance_threshold, other_label, str(ann_path))
        if export_ply:
            save_ply(str(ann_path))

    elapsed = time.time() - t0
    print(f"  完成 ({elapsed:.1f}s)")
    return {'scene_id': sid, 'status': 'success', 'time': elapsed,
            'num_points': num_points, 'area': mesh.area}


def _process_scene_wrapper(args_tuple):
    """进程池包装函数"""
    try:
        return process_scene(*args_tuple)
    except Exception as e:
        return {'scene_id': args_tuple[0]['scene_id'],
                'status': 'error', 'error': str(e)}


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description='建筑立面点云数据集生成器 v4 (自适应密度采样)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
=== 只重新生成 synthetic (保护已有 annotated) ===
  %(prog)s "C:/LOD3_DATA" --recursive --mode synthetic

=== 首次全量生成 ===
  %(prog)s "C:/LOD3_DATA" --recursive --mode both

=== 自定义密度 ===
  %(prog)s "C:/LOD3_DATA" --recursive --mode synthetic --density 2000

=== 手动固定点数 (回退到v3行为) ===
  %(prog)s "C:/LOD3_DATA" --recursive --mode synthetic --num_points 500000
        """,
    )
    parser.add_argument('folders', nargs='+',
                        help='场景文件夹 或 --recursive 时的根目录')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='递归扫描目录树，自动发现所有场景')
    parser.add_argument('--mode', choices=['synthetic', 'annotate', 'both'],
                        default='both', help='运行模式 (默认: both)')

    # 自适应采样参数
    parser.add_argument('--density', type=int, default=1500,
                        help='每平方米采样点数 (默认1500)')
    parser.add_argument('--min_points', type=int, default=100_000,
                        help='最少采样点数 (默认100K)')
    parser.add_argument('--max_points', type=int, default=5_000_000,
                        help='最多采样点数 (默认5M)')
    parser.add_argument('--num_points', type=int, default=None,
                        help='手动固定采样点数 (设置后忽略 --density)')

    # annotate 参数
    parser.add_argument('--distance_threshold', type=float, default=1.0,
                        help='annotate 最大可信距离 (默认1.0m)')
    parser.add_argument('--other_label', type=int, default=11,
                        help='不可信点标签 (默认11=other)')

    # 输出控制
    parser.add_argument('--ply', action='store_true',
                        help='同时导出 PLY')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认输出到各场景文件夹内）')
    parser.add_argument('--skip_existing', action='store_true',
                        help='跳过已有输出的场景')
    parser.add_argument('--no_protect_annotated', action='store_true',
                        help='允许覆盖已有 annotated 文件 (默认保护)')
    parser.add_argument('--workers', type=int, default=1,
                        help='并行进程数 (默认1)')

    args = parser.parse_args()

    protect_annotated = not args.no_protect_annotated

    # 提示采样策略
    if args.num_points is not None:
        print(f"采样策略: 固定 {args.num_points:,} 点/场景")
    else:
        print(f"采样策略: 自适应 {args.density} 点/m² "
              f"(范围 {args.min_points:,}~{args.max_points:,})")

    if protect_annotated:
        print(f"Annotated 保护: 开启 (已有的 annotated 文件不会被覆盖)")

    # ---- 收集场景 ----
    scenes = []
    if args.recursive:
        for root in args.folders:
            print(f"递归扫描: {root}")
            found = discover_scenes_recursive(Path(root))
            scenes.extend(found)
            print(f"  找到 {len(found)} 个场景")
    else:
        for folder in args.folders:
            scene = discover_scene(Path(folder))
            if scene['obj'] and scene['json']:
                scenes.append(scene)
            else:
                print(f"⚠ {folder}: 缺少 OBJ 或 JSON，跳过")

    if not scenes:
        print("未找到任何有效场景。")
        return

    total = len(scenes)
    print(f"\n共 {total} 个场景，模式={args.mode}，"
          f"workers={args.workers}")

    # ---- 处理 ----
    t_start = time.time()
    results = []

    if args.workers <= 1:
        for i, scene in enumerate(scenes, 1):
            try:
                r = process_scene(
                    scene, args.mode,
                    args.density, args.min_points, args.max_points,
                    args.num_points,
                    args.distance_threshold, args.other_label,
                    args.ply, args.output_dir, args.skip_existing,
                    protect_annotated)
                results.append(r)
            except Exception as e:
                print(f"\n  ✗ {scene['scene_id']}: {e}")
                traceback.print_exc()
                results.append({'scene_id': scene['scene_id'],
                                'status': 'error', 'error': str(e)})
    else:
        tasks = [
            (scene, args.mode,
             args.density, args.min_points, args.max_points,
             args.num_points,
             args.distance_threshold, args.other_label,
             args.ply, args.output_dir, args.skip_existing,
             protect_annotated)
            for scene in scenes
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_scene_wrapper, t): t[0]['scene_id']
                       for t in tasks}
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                sid = r['scene_id']
                status = r['status']
                if status == 'error':
                    print(f"\n  ✗ {sid}: {r.get('error')}")

    # ---- 汇总 ----
    elapsed = time.time() - t_start
    n_success = sum(1 for r in results if r['status'] == 'success')
    n_skip = sum(1 for r in results if r['status'] == 'skipped')
    n_error = sum(1 for r in results if r['status'] == 'error')

    print(f"\n{'='*60}")
    print(f"完成: {n_success} 成功, {n_skip} 跳过, {n_error} 失败 "
          f"(共 {total}, {elapsed:.0f}s)")

    # 打印采样统计
    success_results = [r for r in results if r.get('num_points')]
    if success_results:
        pts_list = [r['num_points'] for r in success_results]
        area_list = [r['area'] for r in success_results]
        print(f"\n采样统计:")
        print(f"  面积: min={min(area_list):.0f}  median={np.median(area_list):.0f}  "
              f"max={max(area_list):.0f} m²")
        print(f"  点数: min={min(pts_list):,}  median={int(np.median(pts_list)):,}  "
              f"max={max(pts_list):,}")
        print(f"  间距: {1/np.sqrt(np.median(pts_list)/np.median(area_list))*100:.2f}cm (中位)")
        avg_time = elapsed / n_success
        print(f"  速度: {avg_time:.1f}s/场景")

    print(f"{'='*60}")


if __name__ == '__main__':
    main()