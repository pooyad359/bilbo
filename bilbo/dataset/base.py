from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from pqdm import processes, threads

# from tqdm.cli import tqdm

Point = Tuple[float, float]
Quadrilateral = Tuple[Point, Point, Point, Point]
N_CPU = os.cpu_count()


def synthesize(*args, **kwargs):
    num_lines = np.random.randint(30, 150)
    width = np.random.choice([1024, 1536, 2048, 2560, 3072])
    ratio = np.random.choice([1.3, 1.5, 1.75, 2, 2.25, 2.5])
    height = int(width * ratio)
    x_pad = rand_between(0.1, 0.3)
    x_range = (int(width * x_pad), int(width * (1 - x_pad)))
    y_pad = rand_between(0.1, 0.3)
    y_range = (int(height * y_pad), int(height * (1 - y_pad)))
    box_hight = rand_between(height / num_lines / 2, height / num_lines)
    width_range = (int(box_hight), int(box_hight * rand_between(5, 15)))
    line_spacing = int(rand_between(-0.1, 0.5))
    params = {
        "x_range": x_range,
        "y_range": y_range,
        "num_lines": num_lines,
        "page_width": width,
        "page_height": height,
        "height": box_hight,
        "width_range": width_range,
        "line_spacing": line_spacing,
    }
    _x, _y = generate_page_data(
        x_range=x_range,
        y_range=y_range,
        num_lines=num_lines,
        height=box_hight,
        width_range=width_range,
        line_spacing=line_spacing,
        shuffle=False,
    )
    return params, _x, _y


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


def box_to_polygon(box):
    x1, y1, x2, y2 = box
    new_box = [x1, y1, x2, y1, x2, y2, x1, y2]
    return new_box


class BilboDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path | int = 100, transforms=None):
        if isinstance(path, int):
            self.path = path
            self.files = None
            self.data = [synthesize() for _ in range(path)]
        else:
            self.path = Path(path)
            self.files = list(self.path.glob("*.npz"))
            self.data = None
        self.transforms = transforms

    def __len__(self):
        return len(self.files) if self.data is None else len(self.data)

    def __getitem__(self, idx):
        if self.data is None:
            data = np.load(self.files[idx])
        else:
            data = self.data[idx]
        x = data["x"]
        y = data["y"]
        if self.transforms:
            x, y = self.transforms(x, y)
        return x, y


class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, transforms=None):
        self.path = Path(path)
        self.transforms = transforms
        self.data = [json.loads(f.read_text()) for f in self.path.glob("*.json")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        x = np.array([a["box"] for a in data])
        n_boxes = len(data)
        order = np.arange(n_boxes)
        matrix = order[:, None] - order[None, :]
        y = (matrix < 0) * 0.5 + (matrix <= 0) * 0.5
        if self.transforms:
            x, y = self.transforms(x, y)
        return x, y


class BilboDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            start = i
            end = min(i + self.batch_size, len(indices))
            batch_indices = indices[start:end]
            batch = [self.dataset[j] for j in batch_indices]
            yield self.collate_fn(batch)

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)
        return x, y

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def _load_data(file_path: Path):
    data = np.load(file_path)
    x = data["x"]
    y = data["y"]
    return x, y


class BilboInMemoryDataLoader:
    def __init__(self, path: str | Path = "./data", batch_size=32, transforms=None, shuffle=True):
        self.dataset = BilboDataset(path, transforms)
        self.batch_size = batch_size
        self.shuffle = shuffle
        raw_data = threads.pqdm(
            self.dataset.files,
            _load_data,
            desc="Loading Data",
            n_jobs=N_CPU * 4,
            exception_behaviour="raise",
        )
        self.data = processes.pqdm(
            raw_data,
            self.dataset.transforms,
            desc="Transforming Data",
            n_jobs=N_CPU,
            argument_type="args",
            exception_behaviour="raise",
        )
        # self.data = []

    def __iter__(self):
        n = len(self.dataset)
        x, y = self.collate_fn(self.data)
        indices = list(range(n))
        if self.shuffle:
            random.shuffle(indices)
        x, y = x[indices], y[indices]
        for i in range(0, len(indices), self.batch_size):
            start = i
            end = min(i + self.batch_size, len(indices))
            yield x[start:end], y[start:end]

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)
        return x, y

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


class BilboDataGenerator:
    def __init__(self, count: int = 100, batch_size=32, transforms=None):
        self.count = count
        self.batch_size = batch_size
        self.transforms = transforms
        self.raw_data = processes.pqdm(
            range(count),
            synthesize,
            desc="Loading Data",
            n_jobs=N_CPU - 1,
            exception_behaviour="raise",
        )
        self.data = processes.pqdm(
            [o[1:] for o in self.raw_data],
            self.transforms,
            desc="Transforming Data",
            n_jobs=N_CPU - 1,
            argument_type="args",
            exception_behaviour="raise",
        )

    def __iter__(self):
        x, y = self.collate_fn(self.data)
        indices = list(range(self.count))
        for i in range(0, len(indices), self.batch_size):
            start = i
            end = min(i + self.batch_size, len(indices))
            yield x[start:end], y[start:end]

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)
        return x, y

    def __len__(self):
        return math.ceil(self.count / self.batch_size)


class ReadingBankDataset:
    def __init__(self, path: str | Path = "./data", transforms=None, count=1000, skip=0):
        self.path = Path(path)
        self.transforms = transforms
        self.count = count
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist")
        if self.path.suffix != ".json":
            raise ValueError(f"Path {self.path} must be a json file")
        self.raw_data = []
        with self.path.open("r") as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                if i >= (count + skip):
                    break
                boxes = json.loads(line.strip())["src"]
                n_boxes = len(boxes)
                order = np.arange(n_boxes)
                matrix = order[:, None] - order[None, :]
                relation = (matrix < 0) * 0.5 + (matrix <= 0) * 0.5
                polygons = [box_to_polygon(b) for b in boxes]
                self.raw_data.append((np.array(polygons), relation))
        self.data = processes.pqdm(
            self.raw_data,
            self.transforms,
            desc="Transforming Data",
            n_jobs=N_CPU - 1,
            argument_type="args",
            exception_behaviour="raise",
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
