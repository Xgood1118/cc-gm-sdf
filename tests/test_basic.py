# -*- coding: utf-8 -*-
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdf.core import (  # noqa: E402
    Image,
    compute_sdf,
    in_shape,
    normalize_sdf,
    outline,
)


# ---------------------------------------------------------------- 基础接口

def test_in_circle():
    s = {"type": "circle", "cx": 5, "cy": 5, "r": 2}
    assert in_shape(s, 5, 5)
    assert not in_shape(s, 9, 5)


def test_in_polygon():
    s = {"type": "polygon", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    assert in_shape(s, 5, 5)
    assert not in_shape(s, 12, 5)


def test_sdf_shape():
    s = {"type": "circle", "cx": 10, "cy": 10, "r": 5}
    dist = compute_sdf(s, 20, 20)
    assert len(dist) == 20
    assert len(dist[0]) == 20
    assert dist[10][10] < 0


def test_image_default():
    img = Image(4, 4)
    assert img.at(0, 0) == 0


# ---------------------------------------------------------------- 距离精度

def test_diagonal_distance_is_euclidean_not_octagonal():
    """斜向距离必须等于真实欧氏距离，不能出现棋盘式的八角形高估。"""
    # 半径 10 的圆，圆心放在 (20, 20)
    s = {"type": "circle", "cx": 20, "cy": 20, "r": 10}
    dist = compute_sdf(s, 40, 40)

    # 圆外各点的距离 = 到圆心距离 - r，斜向点也一样精确
    checks = [
        (30, 20, 0.0),    # 正东，贴在圆上
        (20, 30, 0.0),    # 正南
        (30, 30, 10 * math.sqrt(2) - 10),  # 45 度斜向，约 4.14 而非整数 5
        (27, 24, math.hypot(7, 4) - 10),
        (34, 20, 4.0),
    ]
    for x, y, expected in checks:
        assert abs(dist[y][x] - expected) < 1e-9, (x, y, dist[y][x], expected)

    # 等距线仍是圆：同半径圆周上一圈采样点距离都应近似 0，
    # 八角形算法会让斜向点距离明显偏大。
    max_err = 0.0
    for k in range(16):
        a = 2 * math.pi * k / 16
        x = int(round(20 + 10 * math.cos(a)))
        y = int(round(20 + 10 * math.sin(a)))
        max_err = max(max_err, abs(dist[y][x]))
    assert max_err < math.sqrt(2.0) / 2.0 + 1e-9  # 像素中心最大偏差不超过半格

    # 矩形斜向同样精确：外点到角点是斜边，不是曼哈顿距离
    r = {"type": "rect", "x": 10, "y": 10, "w": 10, "h": 10}
    rd = compute_sdf(r, 40, 40)
    assert abs(rd[10][0] - 10.0) < 1e-9        # 正西 10 格
    assert abs(rd[0][0] - math.hypot(10, 10)) < 1e-9  # 对角点


def test_distances_are_fractional():
    """距离必须保留小数，不能整幅只有整数级。"""
    s = {"type": "circle", "cx": 20, "cy": 20, "r": 10}
    dist = compute_sdf(s, 40, 40)
    values = set(v for row in dist for v in row)
    assert any(abs(v - round(v)) > 1e-6 for v in values)


# ---------------------------------------------------------------- 符号

def test_sign_consistent_all_shapes():
    """内部为负、外部为正、边界为 0，全图符号一致。"""
    shapes = [
        {"type": "circle", "cx": 12, "cy": 12, "r": 6},
        {"type": "rect", "x": 6, "y": 8, "w": 12, "h": 8},
        {"type": "polygon",
         "points": [[4, 4], [28, 6], [24, 26], [12, 22], [6, 28]]},
    ]
    for s in shapes:
        dist = compute_sdf(s, 32, 32)
        for y in range(32):
            for x in range(32):
                v = dist[y][x]
                if in_shape(s, x, y):
                    assert v <= 0.0, (s["type"], x, y, v)
                else:
                    assert v >= 0.0, (s["type"], x, y, v)


# ---------------------------------------------------------------- 可视化

def test_normalize_keeps_inside_outside_difference():
    """归一化后内外明暗不同，不是对称图；参考范围可配置。"""
    s = {"type": "circle", "cx": 16, "cy": 16, "r": 8}
    dist = compute_sdf(s, 40, 40)

    g8 = normalize_sdf(dist, max_dist=8.0)
    assert g8[16][16] == 0          # 圆心（深内部，d=-8）-> 黑
    assert g8[16][32] == 255        # 距离恰好 +8 的外点 -> 白
    assert g8[16][24] == 128        # 边界（d=0）-> 中灰
    assert g8[16][31] > 128         # 外部整体偏亮
    # 内外等距的点：内部暗、外部亮，灰度关于 128 镜像而非同值
    assert g8[16][12] < 128 < g8[16][28]
    assert 255 - g8[16][28] == g8[16][12]

    g4 = normalize_sdf(dist, max_dist=4.0)   # 参考范围缩小，更多点被截断
    assert g4[16][16] == 0 and g4[16][31] == 255
    assert g4[16][28] == 255 and g8[16][28] < 255
    assert all(0 <= v <= 255 for row in g8 for v in row)
    assert all(0 <= v <= 255 for row in g4 for v in row)


# ---------------------------------------------------------------- 描边

def test_outline_excludes_deep_interior():
    """描边只取边界附近环带，内部深处不入选，且宽度可配置。"""
    s = {"type": "circle", "cx": 24, "cy": 24, "r": 12}
    dist = compute_sdf(s, 48, 48)

    narrow = outline(dist, width=1.0)
    wide = outline(dist, width=3.0)

    # 圆心是内部深处，任何合理宽度都不应被选中
    assert narrow[24][24] == 0
    assert wide[24][24] == 0
    # 远离形状的外部角落同样不入选
    assert narrow[0][0] == 0 and wide[0][0] == 0
    # 边界像素入选
    assert narrow[24][36] == 1
    # 宽度变大，描边像素单调变多
    assert sum(sum(r) for r in wide) > sum(sum(r) for r in narrow)
    # 尺寸与原图一致（骨架里曾错误地多出一列）
    assert len(narrow) == 48 and all(len(r) == 48 for r in narrow)
    # 零宽只剩距离恰为 0 的边界像素
    zero = outline(dist, width=0.0)
    assert all(v == 0 or dist[y][x] == 0.0
               for y, row in enumerate(zero) for x, v in enumerate(row))


# ---------------------------------------------------------------- 凹多边形

def test_concave_polygon_dent_is_outside():
    """U 形凹多边形的凹陷区域必须判为外部，距离符号为正。"""
    # 外框 20x20，顶部开口、中间向下凹出 10 深 12 宽的槽
    u = {"type": "polygon", "points": [
        [0, 0], [4, 0], [4, 10], [16, 10], [16, 0], [20, 0],
        [20, 20], [0, 20],
    ]}
    dist = compute_sdf(u, 24, 24)

    assert not in_shape(u, 10, 5)      # 槽内：凹陷区是外部
    assert dist[5][10] > 0.0           # 符号为正
    assert abs(dist[5][10] - 5.0) < 1e-9   # 到槽底 y=10 恰好 5 格
    assert in_shape(u, 10, 15)         # 槽底下方仍是内部
    assert dist[15][10] < 0.0
    assert in_shape(u, 2, 5)           # 左臂内部
    assert dist[5][2] < 0.0
    assert not in_shape(u, 22, 22)     # 形状外
    assert dist[22][22] > 0.0

    # 题目自带的五边形凹点同样成立
    pent = {"type": "polygon",
            "points": [[30, 40], [150, 30], [160, 150], [90, 120], [50, 160]]}
    assert in_shape(pent, 100, 60)
    assert not in_shape(pent, 95, 140)  # 凹点 (90,120) 下方的凹陷区


# ---------------------------------------------------------------- 贴边不截断

def test_shape_touching_border_not_truncated():
    """形状贴边/越出图像时，边界像素距离仍正确、不被扫描截断。"""
    # 矩形左上角与图像左上角重合
    r = {"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10}
    rd = compute_sdf(r, 16, 16)
    assert rd[0][0] == 0.0
    assert rd[5][0] == 0.0             # 左边界上的点
    assert abs(rd[0][13] - 3.0) < 1e-9  # 上方外部，到 y=0 为 3

    # 形状越出左边界：内部点到真实边 x=-4 的距离不被图像边缘截断
    c = {"type": "circle", "cx": -4, "cy": 8, "r": 8}
    cd = compute_sdf(c, 16, 16)
    assert cd[8][0] < 0.0
    assert abs(cd[8][0] - (-4.0)) < 1e-9
    assert cd[8][4] == 0.0             # 圆右边界穿过 (4, 8)
    assert abs(cd[8][8] - 4.0) < 1e-9  # 再往外 4 格，距离仍连续、未被截断
