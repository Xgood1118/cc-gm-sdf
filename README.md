# 有符号距离场生成

骨架能判断点是否在形状内、能扫出一张距离图，但距离与符号都不可用。

## 已知缺口

- `compute_sdf` 用八个方向的环扫描，得到的是棋盘距离，斜向距离被高估，圆形边缘出成八角形。
- 符号约定是内部为负，与常规 SDF 相反。
- 扫描步长为整数，距离场只有整数级，靠近边界一片平。
- `normalize_sdf` 取绝对值映射，内外符号全丢，可视化看起来是对称的。
- `outline` 取 `abs(v) < width` 的所有像素，内部深处也会被选中。
- 没有返回值域外距离，形状贴边时边界像素的距离被截断。

## 运行

```
python gen_samples.py
python -m sdf.cli data/shapes.json --size 192 --out-dir out
python -m pytest tests -q
```

三个形状：半径 48 的圆、100x70 的矩形、五边形凹多边形。
