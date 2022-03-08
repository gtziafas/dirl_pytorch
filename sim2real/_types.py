from typing import List, Tuple, Dict, Any, Callable, Set, TypeVar, Sequence, Iterator
from typing import Optional as Maybe
from typing import Mapping as Map 
from dataclasses import dataclass 
from abc import ABC 

from numpy import array
from torch import Tensor 
from torch import float as floatt
from torch import long as longt
from torch.optim import Optimizer

QA = Tuple[str, str]

MayTensor = Maybe[Tensor]


@dataclass 
class Box:
    # coordinates in (x,y) cv2-like frame
    x: int 
    y: int
    w: int 
    h: int 


@dataclass
class Rectangle:
    # coordinates in (x,y) cv2-like frame
    x1: int 
    y1: int
    x2: int 
    y2: int 
    x3: int 
    y3: int
    x4: int 
    y4: int 


@dataclass
class Object:
    label: str
    category: str
    box: Box


@dataclass
class ObjectSim:
    label: str
    category: str
    contour: array
    box: Box
    rectangle: Rectangle
    center_of_mass: Tuple[int, int]
    position_2d: Tuple[float, float]


@dataclass
class ObjectCrop(Object):
    image: array 


@dataclass
class Scene:
    environment: str
    image_id: str
    #rgb: array
    #depth: array
    objects: Sequence[Object]

    @property
    def labels(self):
        return [o.label for o in self.objects]

    @property
    def categories(self):
        return list(set([o.category for o in self.objects]))

    @property
    def boxes(self):
        return [o.box for o in self.objects]

    # def get_crops(self):
    #     return [self.rgb[o.box.y : o.box.y+o.box.h, o.box.x : o.box.x+o.box.w] for o in self.objects]


@dataclass
class SceneRGB(Scene):
    image: array

    def get_crops(self):
        return [self.rgb[o.box.y : o.box.y+o.box.h, o.box.x : o.box.x+o.box.w] for o in self.objects]
    

@dataclass
class AnnotatedObject(Object):
    color: List[str]
    special: List[str]

@dataclass
class AnnotatedObjectSim(ObjectSim):
    color: str
    special: List[str]

@dataclass
class SceneGraph:
    nodes: Sequence[AnnotatedObjectSim]
    edges: array

    def print(self):
        ...

    def render(self):
        ...

@dataclass
class AnnotatedScene(Scene):
    query: List[List[str]]
    truth: List[List[int]]
    # qas: List[QA]
    # infs: List[QA]
    
