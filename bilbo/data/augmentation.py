from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from bilbo.utils import rand_between


class RandomRotate:
    def __init__(self, p=0.5, angle: float | Tuple[float, float] = 20):
        self.p = p
        if isinstance(angle, (tuple, list)):
            self.angle_range = angle
        else:
            self.angle_range = (-angle, angle)

    def __call__(self, boxes, relation):
        if np.random.rand() > self.p:
            return boxes, relation
        angle = rand_between(*self.angle_range)
        angle_rad = np.deg2rad(angle)
        xy = np.reshape(boxes, (-1, 2))
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        new_xy = np.matmul(xy, rotation_matrix)
        box_out = new_xy.reshape(-1, 8)
        return box_out, relation


class RandomRowSwap:
    def __init__(self, p=0.5, min_swap=2, max_swap=5):
        self.p = p
        self.min_swap = min_swap
        self.max_swap = max_swap

    def __call__(self, boxes, relation):
        if np.random.rand() > self.p:
            return boxes, relation
        if self.max_swap > self.min_swap:
            n_swap = np.random.randint(self.min_swap, self.max_swap)
        else:
            n_swap = self.min_swap
        for _ in range(n_swap):
            i, j = np.random.choice(range(boxes.shape[0]), 2, replace=False)
            boxes[[i, j], :] = boxes[[j, i], :]
            relation[[i, j], :] = relation[[j, i], :]
            relation[:, [i, j]] = relation[:, [j, i]]
        return boxes, relation


class RandomRowDropout:
    def __init__(self, p=0.5, ratio=None, final_count=None, drop_range=None):
        self.p = p
        if not any([ratio, final_count, drop_range]):
            raise ValueError("One of ratio, final_count or drop_range must be provided.")
        self.ratio = ratio
        self.final_count = final_count
        self.drop_range = drop_range

    def __call__(self, boxes, relation):
        if np.random.rand() > self.p:
            return boxes, relation
        if self.drop_range:
            self.min_drop, self.max_drop = self.drop_range
            if self.max_drop > self.min_drop:
                n_drop = np.random.randint(self.min_drop, self.max_drop)
            else:
                n_drop = self.min_drop
        elif self.final_count:
            n_drop = boxes.shape[0] - self.final_count
        else:
            n_drop = int(boxes.shape[0] * self.ratio)
        for _ in range(n_drop):
            i = np.random.choice(range(boxes.shape[0]))
            boxes = np.delete(boxes, i, axis=0)
            relation = np.delete(relation, i, axis=0)
            relation = np.delete(relation, i, axis=1)

        return boxes, relation


class RandomCrop:
    def __init__(self, p=0.5, final_size: int = 128):
        self.p = p
        self.crop_size = final_size

    def __call__(self, boxes, relation):
        if np.random.rand() > self.p:
            return boxes, relation
        if boxes.shape[1] <= self.crop_size:
            return boxes, relation

        n_rows = boxes.shape[0]
        drop_count = n_rows - self.crop_size
        i1 = np.random.randint(drop_count + 1)
        i2 = i1 + self.crop_size
        box_out = boxes[i1:i2, :]
        rel_out = relation[i1:i2, i1:i2]
        return box_out, rel_out


class PadToSize:
    def __init__(self, p=1, final_size: int = 128):
        self.p = p
        self.final_size = final_size

    def __call__(self, boxes, relation):
        if np.random.rand() > self.p:
            return boxes, relation
        n_rows = boxes.shape[0]
        if n_rows >= self.final_size:
            return boxes, relation
        pad_count = self.final_size - n_rows
        x_pad = np.zeros((pad_count, boxes.shape[1]))
        box_out = np.concatenate((boxes, x_pad), axis=0)
        yout = np.zeros((self.final_size, self.final_size))
        yout[:n_rows, :n_rows] = relation
        np.pad(relation, (0, pad_count), mode="constant", constant_values=0).shape
        return box_out, yout


class Normalize:
    def __init__(self, p=0.5, eps=1e-12):
        self.p = p
        self.eps = eps

    def __call__(self, boxes, relation):
        if np.random.rand() > self.p:
            return boxes, relation
        _mask = np.any(boxes, axis=1)
        _xy = np.reshape(boxes, (-1, 2))
        mask = np.stack([_mask] * 4).T.flatten()
        xmin = _xy[mask, 0].min()
        xmax = _xy[mask, 0].max()
        ymin = _xy[mask, 1].min()
        ymax = _xy[mask, 1].max()
        xy_min = np.array([xmin, ymin])
        xy_max = np.array([xmax, ymax])
        xy_out = mask[:, None] * (_xy - xy_min) / (xy_max - xy_min + self.eps)
        box_out = xy_out.reshape(-1, 8)
        return box_out, relation


class RandomShift:
    def __init__(
        self,
        p=0.5,
        x_range: float | Tuple[float, float] = 100,
        y_range: float | Tuple[float, float] = 100,
    ):
        self.p = p
        if isinstance(x_range, (tuple, list)):
            self.x_range = x_range
        else:
            self.x_range = (-x_range, x_range)
        if isinstance(y_range, (tuple, list)):
            self.x_range = y_range
        else:
            self.x_range = (-y_range, y_range)

    def __call__(self, boxes, relation):
        if np.random.rand() > self.p:
            return boxes, relation
        shift_x = rand_between(*self.x_range)
        shift_y = rand_between(*self.x_range)
        diff = np.array([shift_x, shift_y] * 4)
        box_out = boxes + diff[None, :]
        return box_out, relation


class RandomSinDistortion:
    def __init__(self, p=0.5, wave_length=(1000, 2000), amp_range=(50, 200)):
        self.p = p
        self.wave_length = wave_length
        self.amp_range = amp_range

    def __call__(self, boxes, relation):
        if np.random.rand() > self.p:
            return boxes, relation
        wave_length = np.random.randint(*self.wave_length)
        amp = np.random.randint(*self.amp_range)
        _xy = np.reshape(boxes, (-1, 2))
        _x = _xy[:, 0]
        _y = _xy[:, 1]
        y_out = _y + amp * np.sin(_x / wave_length * 2 * np.pi)
        box_out = np.stack([_x, y_out], axis=1).reshape(-1, 8)
        return box_out, relation


class SplineDistortion:
    def __init__(self, p=0.5, n_points=5, amp_range=(-200, 200)):
        self.p = p
        self.n_points = n_points
        self.amp_range = amp_range

    def __call__(self, boxes, relation):
        if np.random.rand() > self.p:
            return boxes, relation
        n_points = self.n_points

        _xy = np.reshape(boxes, (-1, 2))
        _x = _xy[:, 0]
        _y = _xy[:, 1]
        x_points = np.linspace(_x.min(), _x.max(), n_points)
        dy_points = np.random.randint(*self.amp_range, n_points)
        dy_out = np.interp(_x, x_points, dy_points)
        y_out = _y + dy_out
        box_out = np.stack([_x, y_out], axis=1).reshape(-1, 8)
        return box_out, relation


class ToTensor:
    def __init__(self, normalize=False):
        self.normalizer = Normalize(p=1) if normalize else None

    def __call__(self, boxes, relation):
        if self.normalizer is not None:
            boxes, relation = self.normalizer(boxes, relation)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(relation, dtype=torch.float32)


class Transform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, boxes, relation):
        for tsfm in self.transforms:
            boxes, relation = tsfm(boxes, relation)
        return boxes, relation
