from uuid import uuid4

import numpy as np
import rich
import rich.json
import rich.progress
import rich.progress_bar

from bilbo.data.dataset import generate_page_data


def rand_between(a: float, b: float) -> float:
    return np.random.rand() * (b - a) + a


def synthesize():
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


def batch_synthesize(n):
    for _ in rich.progress.track(range(n)):
        params, x, y = synthesize()
        np.savez(f"data/val/{uuid4()}.npz", x=x, y=y)


if __name__ == "__main__":
    batch_synthesize(200)
