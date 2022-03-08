from _types import *
import numpy as np
from math import ceil
from random import sample 
import os
import cv2
import torch
import pandas as pd 
from functools import lru_cache
from torch import Tensor
from torch.utils.data import DataLoader
import imgaug.augmenters as iaa


def crop_box(img: array, box: Box) -> array:
    return img[box.y : box.y + box.h, box.x : box.x + box.w]


def crop_contour(img: array, contour: array) -> array:
    mask = np.zeros((img.shape[0], img.shape[1], 3))
    mask = cv2.drawContours(mask, [contour], 0, (0xff,0xff,0xff), -1)
    box = cv2.boundingRect(contour)
    mask = np.where(mask == 0xff, img, 0)
    return crop_box(mask, Box(*box))


def show(img: array, legend: Maybe[str] = None):
    legend = 'unlabeled' if legend is None else legend
    cv2.imshow(legend, img)
    while 1:
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyWindow(legend)


# pad image with zeros in the center of a desired resolution frame
def pad_with_frame(imgs: List[array], desired_shape: Tuple[int, int]) -> List[array]:
    H, W = desired_shape
    
    def _pad_with_frame(img: array) -> array:
        # construct a frame of desired resolution
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # paste image in the center of the frame
        starty, startx = max(0, (H - img.shape[0]) // 2), max(0, (W - img.shape[1]) // 2)
        frame[starty : starty + min(H, img.shape[0]), startx :  startx + min(W, img.shape[1]), :] = img
        return frame

    return list(map(_pad_with_frame, imgs))


def crop_boxes_fixed(desired_shape: Tuple[int, int]) -> Callable[[List[array]], List[array]]:
    H, W = desired_shape

    def _crop_boxes_fixed(imgs: List[array]) -> List[array]:
        imgs = list(imgs)

        # identify images larger than desired resolution
        large_idces = [idx for idx, i in enumerate(imgs) if i.shape[0] > H or i.shape[1] > W]

        # identify maximum dimension of large images
        large_maxdims = [max(imgs[i].shape[0:2]) for i in large_idces]

        for idx, img in enumerate(imgs):
            # if large, resize to desired shape while maintaining original ratio
            if idx in large_idces:
                maxdim = large_maxdims[large_idces.index(idx)]
                img = pad_with_frame([img], (maxdim, maxdim))[0]
                imgs[idx] = cv2.resize(img, (H, W))
            # if small, pad to desired shape
            else:
                imgs[idx] = pad_with_frame([img], (H, W))[0]

        return imgs
    
    return _crop_boxes_fixed


class SimScenesDataset:
    def __init__(self, images_path: str, csv_path: str):
        CATEGORY_MAP = {
        'mug'       :   'coffee mug',
        'bowl'      :   'bowl',
        'can'       :   'soda can',
        'cereal'    :   'cereal box',
        'cap'       :   'cap',
        'flashlight':   'flashlight'
        }
        self.root = images_path
        self.csv_path = csv_path
        self.table = pd.read_table(csv_path)
        self.image_ids = self.table['image_id'].tolist()
        self.labels = [row.split(',') for row in self.table['label'].tolist()]
        self.contours =  [[eval(x.strip("()")) for x in row.split("),")] for row in self.table["RGB_contour"].tolist()]
        self.contours = [[np.int0([[[c[i], c[i+1]]] for i in range(0, len(c)-1, 2)]) for c in row] for row in self.contours]
        self.pos_2d = [[float(x.strip("()")) for x in p.split(',')] for p in self.table['2D_position'].tolist()]
        self.pos_2d = [[(p[i], p[i+1]) for i in range(0, len(p)-1, 2)] for p in self.pos_2d]
        moments = [[cv2.moments(c) for c in row] for row in self.contours]
        self.centers = [[(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for M in row] for row in moments]
        self.boxes = [[Box(*cv2.boundingRect(c)) for c in row] for row in self.contours]
        self.rects = [[Rectangle(*sum(cv2.boxPoints(cv2.minAreaRect(c)).tolist(), [])) for c in row] for row in self.contours]          
        # self.centers = [[int(x.strip("()")) for x in c.split(',')] for c in self.table['RGB_center_of_mass'].tolist()]
        # self.centers = [[(c[i], c[i+1]) for i in range(0, len(c)-1, 2)] for c in self.centers]
        # self.boxes = [[int(x.strip("()")) for x in b.split(',')] for b in self.table['RGB_bounding_box'].tolist()]
        # self.boxes = [[Box(*b[i:i+4]) for i in range(0, len(b)-1, 4)] for b in self.boxes]
        # self.rects = [[int(x.strip("()")) for x in r.split(',')] for r in self.table['RGB_rotated_box'].tolist()]
        # self.rects = [[Rectangle(*r[i:i+8]) for i in range(0, len(r)-1, 8)] for r in self.rects]
        self.categories = [[CATEGORY_MAP[l.split('_')[0]] for l in labs] for labs in self.labels]
        self.objects = [[ObjectSim(l, cat, co, b, r, c, p) for l, cat, co, p, c, b, r in zip(ls, cats, cos, ps, cs, bs, rs)] 
                        for ls, cats, cos, ps, cs, bs, rs in zip(self.labels, self.categories, self.contours,
                        self.pos_2d, self.centers, self.boxes, self.rects)]  

    def get_image(self, n: int) -> array:
        return cv2.imread(os.path.join(self.root, str(self.image_ids[n]) + '.png'))

    def get_image_from_id(self, image_id: int) -> array:
        return cv2.imread(os.path.join(self.root, str(image_id) + '.png'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n: int) -> Scene:
        return Scene(environment="sim", 
                     image_id=self.image_ids[n], 
                     objects=self.objects[n])

    def show(self, n: int):
        scene = self.__getitem__(n)
        img = self.get_image(n).copy()
        for obj in scene.objects:
            x, y, w, h = obj.box.x, obj.box.y, obj.box.w, obj.box.h
            rect = obj.rectangle
            rect = np.int0([(rect.x1, rect.y1), (rect.x2, rect.y2), (rect.x3, rect.y3), (rect.x4, rect.y4)])
            img = cv2.putText(img, obj.label, (x, y), fontFace=0, fontScale=1, color=(0,0,0xff))
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0xff, 0, 0), 2)
            img = cv2.drawContours(img, [rect], 0, (0,0,0xff), 2)
            img = cv2.drawContours(img, [obj.contour], 0, (0,0xff, 0), 1)
        show(img, str(self.image_ids[n]) + ".png")

    def show_id(self, id: int):
        self.show(self.image_ids.index(id))

    def inspect(self):
        for n in range(self.__len__()):
            self.show(n)

    def massage(self, from_, to_):
        drop = []
        for i in range(from_, to_):
            scene = self.__getitem__(i)
            img = self.get_image(i).copy()
            for obj in scene.objects:
                x, y, w, h = obj.box.x, obj.box.y, obj.box.w, obj.box.h
                rect = obj.rectangle
                rect = np.int0([(rect.x1, rect.y1), (rect.x2, rect.y2), (rect.x3, rect.y3), (rect.x4, rect.y4)])
                img = cv2.putText(img, obj.label, (x, y), fontFace=0, fontScale=1, color=(0,0,0xff))
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0xff,0,0  ), 2)    
                img = cv2.drawContours(img, [rect], 0, (0,0,0xff), 2)
                img = cv2.drawContours(img, [obj.contour], 0, (0,0xff,0), 2)
            cv2.imshow(str(self.image_ids[i]), img)
            while True:
                key = cv2.waitKey(1) & 0xff
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    drop.append(self.image_ids[i])
                    break
            cv2.destroyWindow(str(self.image_ids[i]))
        return drop


def get_sim_rgbd_scenes():
    return SimScenesDataset("/home/p300488/dual_arm_ws/DATASET/Images", "/home/p300488/dual_arm_ws/DATASET/data.tsv")


def get_sim_rgbd_objects(size: int = -1):
    ds = get_sim_rgbd_scenes()
    size = size if size > -1 else len(ds)
    keep_idces = sample(range(len(ds)), size)
    crops, labels = [], []
    for i, scene in enumerate(ds):
        if i  not in keep_idces:
            continue
        rgb = ds.get_image(i)
        crops.extend([crop_contour(rgb, o.contour) for o in scene.objects])
        labels.extend([o.label for o in scene.objects])
    return crops, labels


class FromTableDataset:
    def __init__(self, 
                root: str = "/home/p300488/sim2realVL/datasets/rgbd-scenes", 
                table_file: str ='rgbd-scenes_boxes.tsv'):
        self.name = table_file.split('_')[0]
        self.root = root
        self.table = pd.read_table(os.path.join(root, table_file))
        self.parse_table()
        self.unique_scenes = sorted(list(set(self.rgb_paths)))

    def parse_table(self):
        self.scenes = self.table['scene'].tolist()
        self.environments = self.table['subfolder'].tolist()
        self.image_ids = self.table['image_id'].tolist()
        self.categories = [' '.join(cat.split('_')) for cat in self.table['object'].tolist()]
        self.labels = self.table['label'].tolist()
        self.boxes = [Box(*tuple(map(int, b.strip('()').split(',')))) for b in self.table['box']]
        self.rgb_paths = [os.path.join(self.root, 
                                        scene,
                                        env,
                                        'rgb',
                                        '_'.join([env, str(iid)]) + '.png')
                            for scene, env, iid in zip(self.scenes, self.environments, self.image_ids)]

    @lru_cache(maxsize=None)
    def get_image(self, n: int) -> array:
        return cv2.imread(self.unique_scenes[n])

    @lru_cache(maxsize=None)
    def get_depth(self, n: int) -> array:
        scene_path = self.unique_scenes[n].split('/')
        token = scene_path[-1].split('.png')[0] + '_depth.png'
        depth_path = '/'.join(scene_path[:-2] + ['depth', token])
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        return (depth / depth.max() * 0xff).astype('uint8')


class RGBDObjectsDataset(FromTableDataset):

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, n: int) -> Object:
        unique_idx = self.unique_scenes.index(self.rgb_paths[n])
        img = self.get_image(unique_idx)
        return ObjectCrop(label = self.labels[n], 
                      category = self.categories[n],
                      image = crop_box(img, self.boxes[n]),
                      box = self.boxes[n])


def get_real_rgbd_objects():
    ds = RGBDObjectsDataset()
    xs, ys = zip(*[(o.image, o.label) for o in ds])
    return list(xs), list(ys)


def few_labels(data: Tensor, labels: Tensor, num_pts: int, num_classes: int=10):
    num_samples = data.shape[0]
    data_subset, labels_subset = [], []
    for label in range(num_classes):
        filterr = torch.where(labels.argmax(-1) == label)
        select_pts = sample(list(range(filterr[0].shape[0])), num_pts)
        #data_subset.append(data[filterr][0:num_pts])
        #labels_subset.append(labels[filterr][0:num_pts])
        data_subset.append(data[filterr][select_pts])
        labels_subset.append(labels[filterr][select_pts])
    return torch.cat(data_subset, dim=0), torch.cat(labels_subset, dim=0) 


def load_datasets(img_resize: Tuple[int, int] = (75, 75),
                  augment: bool = False,
                  save: Maybe[str] = None) -> Tuple[Tuple[Tensor, Tensor], ...]:
    LABELMAP = {
        'bowl_3'    :   'bowl_2',
        'bowl_4'    :   'bowl_2',
        'cap_1'     :   'cap_white',
        'cap_3'     :   'cap_black',
        'cap_4'     :   'cap_red',
        'cereal_box_4'  : 'cereal_box_1',
        'cereal_box_2'  : 'cereal_box_3',
        'cereal_box_1'  : 'cereal_box_2',
        'flashlight_1'  : 'flashlight_black',
        'flashlight_2'  : 'flashlight_red',
        'flashlight_3'  : 'flashlight_blue',
        'flashlight_5'  : 'flashlight_yellow',
        'soda_can_1'    : 'can_pepsi',
        'soda_can_3'    : 'can_sprite',
        'soda_can_6'    : 'can_fanta',
        'coffee_mug_1'  : 'mug_red',
        'coffee_mug_6'  : 'mug_yellow'
    }
    y_tokenizer = {v: k for k,v in enumerate(set(LABELMAP.values()))}

    x_sim, y_sim = get_sim_rgbd_objects()
    x_real, y_real = get_real_rgbd_objects()
    labelset = sorted(set(y_sim))
    categoryset = sorted(set([y.split('_')[0] for y in labelset]))
    c_tokenizer = {v: k for k, v in enumerate(categoryset)}
    print(f'Labelset: {labelset}')
    print(f'Categoryset: {categoryset}')

    x_sim, y_sim = zip(*[(x_sim[i], y) for i,y in enumerate(y_sim) if y in LABELMAP.values()])
    x_real, y_real = zip(*[(x_real[i], LABELMAP[y]) for i,y in enumerate(y_real) if y in LABELMAP.keys()])
    c_sim = [y.split('_')[0] for y in y_sim]
    c_real = [y.split('_')[0] for y in y_real]
    
    x_sim = crop_boxes_fixed(img_resize)(x_sim)
    x_real = crop_boxes_fixed(img_resize)(x_real)

    x_sim = torch.stack([torch.tensor(x, dtype=floatt).transpose(2,0).div(0xff) for x in x_sim])
    x_real = torch.stack([torch.tensor(x, dtype=floatt).transpose(2,0).div(0xff) for x in x_real])
    y_sim = torch.tensor([y_tokenizer[y] for y in y_sim], dtype=longt)
    y_real = torch.tensor([y_tokenizer[y] for y in y_real], dtype=longt)
    c_sim = torch.tensor([c_tokenizer[c] for c in c_sim], dtype=longt)
    c_real = torch.tensor([c_tokenizer[c] for c in c_real], dtype=longt)
    
    if save is not None:
        torch.save({'sim': [x_sim, y_sim, c_sim], 'real': [x_real, y_real, c_real]}, save)
        
    return (x_sim, y_sim, c_sim), (x_real, y_real, c_real)