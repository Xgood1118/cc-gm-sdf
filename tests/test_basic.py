# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdf.core import Image, compute_sdf, in_shape  # noqa: E402


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
