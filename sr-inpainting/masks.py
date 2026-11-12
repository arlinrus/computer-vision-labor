import numpy as np
from PIL import Image

def make_mask(kind:str, h:int, w:int) -> np.array:
    m = np.zeros((h,w), np.unit8)
    if kind == "ones":
        m[:]=1
    elif kind == "zeros":
        pass
    elif kind == "combined": # эллипс + прямоугольник
        yy, xx = np.mgrid[0:h, 0:w]
        yy, xx = np.mgrid[0:h, 0:w]
        ellipse = ((yy - 0.35 * h) / (0.22 * h)) ** 2 + ((xx - 0.35 * w) / (0.25 * w)) ** 2 <= 1
        rect = (yy >= 0.55 * h) & (yy < 0.9 * h) & (xx >= 0.55 * w) & (xx < 0.9 * w)
        m[ellipse|rect] = 1
    elif kind == "half":
        m[:, : w //2] = 1
    else:
        raise ValueError(kind)
    return m

def load_mask(path:str, h:int, w:int) -> np.ndarray:
    m = (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)
    assert m.shape == (h, w), f"маска {m.shape}, а картинка {(h, w)}"
    return m


