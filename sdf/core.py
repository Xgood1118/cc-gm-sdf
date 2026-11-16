# -*- coding: utf-8 -*-
"""有符号距离场（SDF）生成器，仅依赖 Python 标准库。

符号约定（全图一致）：

* 形状内部为负距离；
* 形状外部为正距离；
* 正好落在边界上为 0。

距离取像素中心到形状边界的真实欧氏最短距离（保留小数），
直接由几何解析式给出（圆用圆方程，矩形/多边形用点到线段距离），
不做网格扫描，因此斜向距离不会被高估、圆的等距线仍是圆，
形状贴边或越出图像边界时距离也不会被截断。
"""

import json
import math
import os
import struct
import zlib

# 小于该阈值的距离视为正好落在边界上，规避浮点投影的舍入误差。
_EPS = 1e-12


class Image(object):
    def __init__(self, width, height, pixels=None):
        self.width = int(width)
        self.height = int(height)
        self.pixels = pixels or [0] * (self.width * self.height)

    def at(self, x, y):
        return self.pixels[y * self.width + x]

    def set(self, x, y, v):
        self.pixels[y * self.width + x] = v


def in_shape(shape, x, y):
    """判断像素 (x, y) 是否在形状内。射线法对凹多边形同样成立。"""
    kind = shape["type"]
    if kind == "circle":
        d = math.hypot(x - shape["cx"], y - shape["cy"])
        return d <= shape["r"]
    if kind == "rect":
        return (shape["x"] <= x < shape["x"] + shape["w"] and
                shape["y"] <= y < shape["y"] + shape["h"])
    if kind == "polygon":
        pts = shape["points"]
        inside = False
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / float(yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    raise ValueError("未知形状 %s" % kind)


def _point_segment_dist(x, y, ax, ay, bx, by):
    """点 (x, y) 到线段 AB 的真实欧氏最短距离。"""
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    if l2 == 0.0:
        return math.hypot(x - ax, y - ay)
    t = ((x - ax) * vx + (y - ay) * vy) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(x - qx, y - qy)


def _rect_segments(shape):
    x0, y0 = float(shape["x"]), float(shape["y"])
    x1, y1 = x0 + float(shape["w"]), y0 + float(shape["h"])
    return [
        (x0, y0, x1, y0),
        (x1, y0, x1, y1),
        (x1, y1, x0, y1),
        (x0, y1, x0, y0),
    ]


def _polygon_segments(shape):
    pts = [(float(p[0]), float(p[1])) for p in shape["points"]]
    n = len(pts)
    return [(pts[i][0], pts[i][1], pts[(i + 1) % n][0], pts[(i + 1) % n][1])
            for i in range(n)]


def _boundary_distance(shape, x, y):
    """像素中心到形状边界的无符号最短距离（圆直接用带符号圆方程）。"""
    kind = shape["type"]
    if kind == "circle":
        return math.hypot(x - shape["cx"], y - shape["cy"]) - float(shape["r"])
    if kind == "rect":
        segments = _rect_segments(shape)
    elif kind == "polygon":
        segments = _polygon_segments(shape)
    else:
        raise ValueError("未知形状 %s" % kind)
    return min(_point_segment_dist(x, y, *seg) for seg in segments)


def compute_sdf(shape, width, height):
    """计算带符号距离场。

    返回 ``height`` 行、每行 ``width`` 个浮点数的二维表，
    内部为负、外部为正、边界为 0。距离是解析几何给出的真实
    欧氏最短距离；形状允许贴边或越出图像，距离不会被截断。
    """
    dist = []
    for y in range(height):
        row = []
        for x in range(width):
            d = _boundary_distance(shape, x, y)
            if abs(d) < _EPS:
                row.append(0.0)
            elif in_shape(shape, x, y):
                row.append(-abs(d))
            else:
                row.append(abs(d))
        dist.append(row)
    return dist


def normalize_sdf(dist, max_dist=8.0):
    """把带符号距离线性映射到 0..255 灰度，保留内外差异。

    ``-max_dist``（最深内部）映射为 0（黑），边界 0 映射为 128（中灰），
    ``+max_dist``（最远外部）映射为 255（白）；内外明暗不对称，
    看图即可分辨内外。``max_dist`` 是可配置的距离参考范围，
    超出范围的距离按端点截断。
    """
    if max_dist <= 0:
        raise ValueError("max_dist 必须为正数")
    out = []
    for row in dist:
        grays = []
        for v in row:
            if v < -max_dist:
                v = -max_dist
            elif v > max_dist:
                v = max_dist
            t = (v / max_dist + 1.0) * 0.5
            g = int(round(t * 255.0))
            grays.append(0 if g < 0 else 255 if g > 255 else g)
        out.append(grays)
    return out


def outline(dist, width=2.0):
    """取边界附近的等值带做描边。

    仅保留 ``|distance| <= width`` 的像素，即跨边界两侧、厚度各为
    ``width`` 的环带；内部深处距离绝对值很大，不会被选中。
    ``width`` 可配置，越大环带越厚、描边像素越多。
    返回与 ``dist`` 同尺寸的 0/1 掩码（行宽与原图一致）。
    """
    if width < 0:
        raise ValueError("width 不能为负")
    mask = []
    for row in dist:
        mask.append([1 if abs(v) <= width else 0 for v in row])
    return mask


def boundary_stats(shape, dist, width, height):
    """统计边界环像素的距离最小值与最大值。

    边界环指与任一四邻域的内外属性不同（图像外一律视为外部）的像素，
    即紧贴边界的内外两圈像素。返回 ``(min_dist, max_dist)``。
    """
    vals = []
    for y in range(height):
        for x in range(width):
            here = in_shape(shape, x, y)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    neighbor_inside = False
                else:
                    neighbor_inside = in_shape(shape, nx, ny)
                if neighbor_inside != here:
                    vals.append(dist[y][x])
                    break
    if not vals:
        return 0.0, 0.0
    return min(vals), max(vals)


def save_png_v(img, path):
    """把灰度 Image 存成 RGBA PNG（三通道同值），纯标准库实现。"""
    raw = bytearray()
    for y in range(img.height):
        raw.append(0)
        for x in range(img.width):
            v = img.at(x, y)
            raw.extend((v, v, v, 255))
    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", img.width, img.height, 8, 6, 0, 0, 0)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(data)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="sdf")
    ap.add_argument("shapes")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--max-dist", type=float, default=8.0,
                    help="归一化用的距离参考范围（像素），默认 8")
    ap.add_argument("--outline-width", type=float, default=2.0,
                    help="描边环带半宽（像素），默认 2")
    args = ap.parse_args(argv)
    with open(args.shapes, "r", encoding="utf-8") as f:
        shapes = json.load(f)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    for s in shapes:
        dist = compute_sdf(s, args.size, args.size)
        gray = normalize_sdf(dist, max_dist=args.max_dist)
        img = Image(args.size, args.size, [v for row in gray for v in row])
        save_png_v(img, os.path.join(args.out_dir, "%s_sdf.png" % s["name"]))

        mask = outline(dist, width=args.outline_width)
        edge = Image(args.size, args.size,
                     [v * 255 for row in mask for v in row])
        save_png_v(edge, os.path.join(args.out_dir, "%s_outline.png" % s["name"]))

        count = sum(sum(r) for r in mask)
        lo, hi = boundary_stats(s, dist, args.size, args.size)
        print("%s 描边像素 %d；边界环距离 min=%.4f max=%.4f"
              % (s["name"], count, lo, hi))
    return 0
