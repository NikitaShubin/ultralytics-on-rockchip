from __future__ import annotations

from typing import NamedTuple


class Model(NamedTuple):
    family: str
    size: str
    task: str

    @property
    def name(self) -> str:
        if self.task == 'detect':
            return f'{self.family}{self.size}'
        return f'{self.family}{self.size}-{self.task}'

    @property
    def dataset(self) -> str:
        return {
            'detect': 'coco_val2017',
            'seg': 'coco_val2017',
            'pose': 'coco_val2017',
            'cls': 'imagenet',
            'obb': 'dotav1',
            'sem': 'cityscapes',
        }[self.task]

    @property
    def input_size(self) -> int:
        if self.task == 'cls':
            return 224
        if self.task in ('obb', 'sem'):
            return 1024
        if self.family == 'yolov5' and '6' in self.size:
            return 1280
        return 640


def _build(spec: list[tuple[str, list[str], list[str]]]) -> list[Model]:
    models: list[Model] = []
    for family, sizes, tasks in spec:
        for size in sizes:
            for task in tasks:
                models.append(Model(family, size, task))
    return models


FULL = _build([
    ('yolov3', ['u', '-sppu', '-tinyu'], ['detect']),
    ('yolov5', ['nu', 'su', 'mu', 'lu', 'xu'], ['detect']),
    ('yolov5', ['n6u', 's6u', 'm6u', 'l6u', 'x6u'], ['detect']),
    ('yolov8', ['n', 's', 'm', 'l', 'x'], ['detect', 'seg', 'cls', 'pose', 'obb']),
    ('yolov9', ['t', 's', 'm', 'c', 'e'], ['detect']),
    ('yolov9', ['c', 'e'], ['seg']),
    ('yolov10', ['n', 's', 'm', 'b', 'l', 'x'], ['detect']),
    ('yolo11', ['n', 's', 'm', 'l', 'x'], ['detect', 'seg', 'cls', 'pose', 'obb']),
    ('yolo12', ['n', 's', 'm', 'l', 'x'], ['detect']),
    ('yolo26', ['n', 's', 'm', 'l', 'x'], ['detect', 'seg', 'cls', 'pose', 'obb', 'sem']),
])

PILOT = _build([
    ('yolov3', ['u'], ['detect']),
    ('yolov5', ['nu'], ['detect']),
    ('yolov8', ['n'], ['detect', 'seg', 'pose', 'obb', 'cls']),
    ('yolov9', ['t'], ['detect']),
    ('yolov10', ['n'], ['detect']),
    ('yolo11', ['n'], ['detect', 'seg', 'cls', 'pose', 'obb']),
    ('yolo12', ['n'], ['detect']),
    ('yolo26', ['n'], ['detect', 'seg', 'cls', 'pose', 'obb', 'sem']),
])
