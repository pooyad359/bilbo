from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

Point = Tuple[float, float]
Quadrilateral = Tuple[Point, Point, Point, Point]


def create_rectangle(top_left: Tuple[float, float], width: float, height: float) -> Quadrilateral:
    x, y = top_left
    return (
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    )


def create_line(
    x_start: float, x_end: float, line_y: float, height: float, width_range: Tuple[float, float]
) -> List[Quadrilateral]:
    counter = 0
    line = []
    x = x_start
    while True:

        counter += 1
        _width = random.randint(int(width_range[0]), int(width_range[1]))
        _height = rand_between(height * 0.75, height * 1.25)
        new_box = create_rectangle((x, line_y), _width, _height)
        line.append(new_box)
        space = rand_between(0, 4) * height
        x = new_box[1][0] + space
        if counter > 50 or x > x_end:
            break
    return line


def create_page(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    num_lines: int,
    height: float,
    width_range: Tuple[float, float],
    line_spacing: float = 0.5,
) -> List[Quadrilateral]:
    page = []
    line_y = y_range[0]
    for i in range(num_lines):
        if line_y > y_range[1]:
            break
        line = create_line(x_range[0], x_range[1], line_y, height, width_range)
        page.extend(line)
        line_y += height * (1 + line_spacing)
    return page


def render_page(page, width, height):
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for box in page:
        draw.polygon(box, outline=(0, 0, 0))
    return canvas


def rand_between(a: float, b: float) -> float:
    return np.random.rand() * (b - a) + a


def generate_page_data(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    num_lines: int,
    height: float,
    width_range: Tuple[float, float],
    line_spacing: float = 0.5,
    shuffle: bool = True,
):
    page = create_page(x_range, y_range, num_lines, height, width_range, line_spacing)
    order = np.arange(len(page))
    matrix = order[:, None] - order[None, :]
    y = (matrix < 0) * 0.5 + (matrix <= 0) * 0.5
    x = np.reshape(page, (-1, 8))
    if shuffle:
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[:, idx][idx, :]
    return x, y


class BilboDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path = "./data", transforms=None):
        self.path = Path(path)
        self.files = list(self.path.glob("*.npz"))
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = data["x"]
        y = data["y"]
        if self.transforms:
            x, y = self.transforms(x, y)
        return x, y


class BilboDataLoader(torch.utils.data.DataLoader):
    def __init__(self, path: str | Path = "./data", batch_size=32, transforms=None, **kwargs):
        self.dataset = BilboDataset(path, transforms)
        super().__init__(self.dataset, batch_size=batch_size, **kwargs)
