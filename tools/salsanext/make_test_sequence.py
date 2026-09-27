"""
Writes a small synthetic SemanticKITTI-format sequence and, as the reference for the C++ side
(AI/Kitti.h + AI/RangeProjection.h, checked by test_kitti_projection.cpp), the network input that
SalsaNext's own projection code builds for it.

  /opt/miniconda3/bin/python make_test_sequence.py <out_dir> <path to SalsaNext's train/common/laserscan.py> <pretrained dir>

The scene, in HDL-64 geometry (64 beams from +2 to -24.8 deg, 2048 columns, sensor 1.73 m up):
a road plane, a building wall 15 m ahead, a pole, and a "moving car" box, with labels
40 road / 50 building / 80 pole / 252 moving-car. Scan 1 is scan 0 seen from 1 m further along x.
"""
import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import yaml

out = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("laserscan", sys.argv[2])
laserscan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(laserscan)
arch = yaml.safe_load((Path(sys.argv[3]) / "arch_cfg.yaml").read_text())
sensor = arch["dataset"]["sensor"]
H, W = sensor["img_prop"]["height"], sensor["img_prop"]["width"]
means = np.array(sensor["img_means"], dtype=np.float32)
stds = np.array(sensor["img_stds"], dtype=np.float32)

SENSOR_H = 1.73
rng = np.random.default_rng(3)


def trace(origin_x):
    """Casts every beam at the scene; returns points (x,y,z,remission) and SemanticKITTI labels."""
    elev = np.deg2rad(np.linspace(2.0, -24.8, 64))
    azim = np.deg2rad(np.linspace(-180, 180, 2048, endpoint=False) + 0.05)
    pts, labels = [], []
    for e in elev:
        for a in azim:
            d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            best, lab = 80.0, 0
            if d[2] < -1e-6:                                         # road
                t = SENSOR_H / -d[2]
                if t < best: best, lab = t, 40
            if d[0] > 1e-6:                                          # building wall at x = 15
                t = (15.0 - origin_x) / d[0]
                p = t * d
                if 0 < t < best and abs(p[1]) < 20 and p[2] < 6: best, lab = t, 50
            # pole: vertical cylinder r = 0.15 at (8, -3)
            ox, oy = -8.0 + origin_x, 3.0
            a2, b2 = d[0] ** 2 + d[1] ** 2, 2 * (ox * d[0] + oy * d[1])
            c2 = ox ** 2 + oy ** 2 - 0.15 ** 2
            disc = b2 * b2 - 4 * a2 * c2
            if a2 > 1e-9 and disc >= 0:
                t = (-b2 - np.sqrt(disc)) / (2 * a2)
                if 0 < t < best and (t * d)[2] < 4 - SENSOR_H: best, lab = t, 80
            # moving car: box x 5..9, y 2..4, z up to 1.5 m above the road
            lo = np.array([5.0 - origin_x, 2.0, -SENSOR_H])
            hi = np.array([9.0 - origin_x, 4.0, 1.5 - SENSOR_H])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo / d), (hi / d)
            tmin = np.nanmax(np.minimum(t1, t2)); tmax = np.nanmin(np.maximum(t1, t2))
            if tmax >= max(tmin, 0) and 0 < tmin < best: best, lab = tmin, 252
            if lab:
                p = best * d
                pts.append([p[0], p[1], p[2], rng.uniform(0.05, 0.6)])
                labels.append(lab | (7 << 16))   # instance id 7 in the upper bits
    return np.array(pts, dtype=np.float32), np.array(labels, dtype=np.uint32)


(out / "velodyne").mkdir(parents=True, exist_ok=True)
(out / "labels").mkdir(exist_ok=True)
for i, ox in enumerate([0.0, 1.0]):
    pts, labels = trace(ox)
    pts.tofile(out / "velodyne" / f"{i:06d}.bin")
    labels.tofile(out / "labels" / f"{i:06d}.label")
    print(f"scan {i}: {len(pts)} points")

# calib Tr (LiDAR -> camera) and camera-frame poses, like KITTI's: the camera looks along +z.
Tr = np.array([[0, -1, 0, 0.1], [0, 0, -1, -0.05], [1, 0, 0, -0.27], [0, 0, 0, 1]], dtype=np.float64)
(out / "calib.txt").write_text("Tr: " + " ".join(f"{v:.9e}" for v in Tr[:3].ravel()) + "\n")
lidar_poses = [np.eye(4), np.eye(4)]
lidar_poses[1][0, 3] = 1.0                                   # moved 1 m forward (LiDAR x)
cam_poses = [Tr @ P @ np.linalg.inv(Tr) for P in lidar_poses]  # inverse of the loader's conversion
(out / "poses.txt").write_text("".join(" ".join(f"{v:.9e}" for v in P[:3].ravel()) + "\n" for P in cam_poses))

# Reference network input for scan 0, via SalsaNext's own LaserScan + the parser's normalisation.
scan = laserscan.LaserScan(project=True, H=H, W=W, fov_up=sensor["fov_up"], fov_down=sensor["fov_down"])
scan.open_scan(str(out / "velodyne" / "000000.bin"))
proj = np.concatenate([scan.proj_range[None], scan.proj_xyz.transpose(2, 0, 1), scan.proj_remission[None]])
proj = (proj - means[:, None, None]) / stds[:, None, None]
proj = proj * scan.proj_mask.astype(np.float32)
mx.save_safetensors(str(out / "reference.safetensors"), {
    "input": mx.array(proj.transpose(1, 2, 0)[None].astype(np.float32)),   # NHWC [1, H, W, 5]
    "proj_x": mx.array(scan.proj_x.astype(np.int32)),
    "proj_y": mx.array(scan.proj_y.astype(np.int32)),
    "lidar_pose1": mx.array(lidar_poses[1].astype(np.float32)),
})
print("wrote", out)
