# -*- coding: utf-8 -*-
"""生成三个形状：圆、矩形、凹多边形。"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SHAPES = [
    {"name": "circle", "type": "circle", "cx": 96, "cy": 96, "r": 48},
    {"name": "rect", "type": "rect", "x": 40, "y": 60, "w": 100, "h": 70},
    {"name": "poly", "type": "polygon",
     "points": [[30, 40], [150, 30], [160, 150], [90, 120], [50, 160]]},
]

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", "shapes.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(SHAPES, f, ensure_ascii=False, indent=2)
    print("已生成", path)
