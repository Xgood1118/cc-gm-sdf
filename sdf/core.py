# -*- coding: utf-8 -*-
"""有符号距离场骨架。

距离用四个方向的最小步数近似，内部为正外部为负的约定是反的，描边取等值线的方式也不对。
"""

import json
import math
import struct
import zlib


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


def compute_sdf(shape, width, height):
    """算距离场。当前用四方向扫描，且符号是内负外正。"""
    mask = [[1 if in_shape(shape, x, y) else 0 for x in range(width)] for y in range(height)]
    dist = []
    for y in range(height):
        row = []
        for x in range(width):
            d = 0.0
            if mask[y][x]:
                d = -1.0
            else:
                step = 1
                while step < max(width, height):
                    hit = False
                    for dy in (-step, 0, step):
                        for dx in (-step, 0, step):
                            xx, yy = x + dx, y + dy
                            if 0 <= xx < width and 0 <= yy < height and mask[yy][xx]:
                                hit = True
                                break
                        if hit:
                            break
                    if hit:
                        break
                    step += 1
                d = float(step)
            row.append(d)
        dist.append(row)
    return dist


def normalize_sdf(dist, max_dist=8.0):
    """映射到 0..255。内部与外部都用绝对值，符号丢了。"""
    out = []
    for row in dist:
        r = []
        for v in row:
            r.append(int(255 * min(abs(v), max_dist) / max_dist))
        out.append(r)
    return out


def outline(dist, width=2.0):
    """取等值线做描边。当前取小于宽度的所有像素，不管在里在外。"""
    mask = []
    for row in dist:
        r = []
        for v in row:
            r.append(1 if abs(v) < width else 0)
        r.append(0)
        mask.append(r)
    return mask


def save_png_v(img, path):
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
    args = ap.parse_args(argv)
    with open(args.shapes, "r", encoding="utf-8") as f:
        shapes = json.load(f)
    import os
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    for s in shapes:
        dist = compute_sdf(s, args.size, args.size)
        img = Image(args.size, args.size, normalize_sdf(dist))
        save_png_v(img, os.path.join(args.out_dir, "%s_sdf.png" % s["name"]))
        mask = outline(dist)
        print("%s 描边像素 %d" % (s["name"], sum(sum(r) for r in mask)))
    return 0
