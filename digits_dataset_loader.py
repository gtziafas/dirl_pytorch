from typing import *
import pickle as pkl
import scipy.io as sio
import h5py
import numpy as np
from math import ceil
from random import sample 
from skimage.transform import resize
import os
import cv2
import torch
from torch import Tensor
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def reshape_many(imgs: np.array) -> List[np.array]:
    imgs = list(imgs)
    def _reshape(img, size_t=28):
        return cv2.resize(img, (size_t, size_t))
    return np.array(list(map(_reshape, imgs)))


def load_dataset(name: str, datasets_directory: str = './resources') -> Tuple[List[Tensor], ...]:
    crop = transforms.CenterCrop(28)

    if name == "mnist":
        tf = transforms.Compose([crop, transforms.ToTensor()])
        
        #read the data in
        x_train, y_train = zip(*MNIST(datasets_directory, train=True, transform=tf))
        x_test, y_test = zip(*MNIST(datasets_directory, train=False, transform=tf))
        
        # convert grayscale to rgb
        x_train = [x.repeat(3, 1, 1) for x in x_train]
        x_test = [x.repeat(3, 1, 1) for x in x_test]
        
        # convert labels to one-hot
        n_values = 10
        y_train = torch.eye(n_values)[torch.tensor(y_train)]
        y_test = torch.eye(n_values)[torch.tensor(y_test)]

        return x_train, list(y_train), x_test, list(y_test)

    elif name == "mnistm":
          # ds =  pkl.load(open(os.path.join(datasets_directory, "mnistm_data.pkl"), 'rb'))
          # x_train = ds['train'] / 255  + ds['valid'] / 255 
          # x_test = ds['test'] / 255 
          # return x_train, None, x_test, None 
          raise NotImplementedError

    elif name == "svhn":
        train_data = sio.loadmat(os.path.join(datasets_directory, 'svhn/train_32x32.mat'))
        test_data = sio.loadmat(os.path.join(datasets_directory, 'svhn/test_32x32.mat'))
        
        # x_train = torch.tensor(reshape_many(train_data['X'].transpose(3, 0, 1, 2))).view(-1, 3, 28, 28)
        # x_test = torch.tensor(reshape_many(test_data['X'].transpose(3, 0, 1, 2))).view(-1, 3, 28, 28)
        x_train = crop(torch.tensor(train_data['X'].transpose(3, 2, 0, 1)))
        x_test = crop(torch.tensor(test_data['X'].transpose(3, 2, 0, 1)))
        
        #convert to onehot
        y_train = train_data['y'] % 10
        y_test = test_data['y'] % 10 
        
        n_values = 10
        y_train = torch.from_numpy(np.eye(n_values)[y_train].squeeze())
        y_test = torch.from_numpy(np.eye(n_values)[y_test].squeeze())

        return list(x_train), list(y_train), list(x_test), list(y_test)
    
    elif name == "usps":
          pad = transforms.Pad(6, fill=0) # from 16x16 to 28x28

          with h5py.File(os.path.join(datasets_directory, "usps/usps.h5"), 'r') as hf:
              train = hf.get('train')
              test = hf.get('test')

              x_train = torch.from_numpy(train.get('data')[:])
              x_train = list(pad(x_train.view(-1, 16, 16).unsqueeze(1).repeat(1, 3, 1, 1)))
              x_test = torch.from_numpy(test.get('data')[:])
              x_test = list(pad(x_test.view(-1, 16, 16).unsqueeze(1).repeat(1, 3, 1, 1)))
              
              y_train = train.get('target')[:]
              y_test = test.get('target')[:]
              
              n_values = 10
              y_train = torch.from_numpy(np.eye(n_values)[y_train].squeeze())
              y_test = torch.from_numpy(np.eye(n_values)[y_test].squeeze())
            
              return x_train, list(y_train), x_test, list(y_test)


def few_labels(data: List[Tensor], labels: List[Tensor], num_pts: int, num_classes: int=10):
    data = torch.stack(data, dim=0)
    labels = torch.stack(labels, dim=0)
    num_samples = data.shape[0]
    data_subset, labels_subset = [], []
    for label in range(num_classes):
        filterr = torch.where(labels.argmax(-1) == label)
        data_subset.append(data[filterr][0:num_pts])
        labels_subset.append(labels[filterr][0:num_pts])

    return tuple(torch.cat(data_subset, dim=0)), tuple(torch.cat(labels_subset, dim=0)) 


def prepare_dataset(source: str, 
                    target: str,
                    examples_per_class: int = 10,
                    num_classes: int = 10
                  ):
    source_data, source_data_test, source_labels, source_labels_test = load_dataset(source)
    target_data, target_data_test, target_labels, target_labels_test = load_dataset(target)

    target_sup_size = examples_per_class * num_classes
    sizing = [target_sup_size * 4, target_sup_size, target_sup_size * 3]
    batch_size = sum(sizing)

    target_data_sup, target_labels_sup = few_labels(target_data, target_labels, examples_per_class, num_classes)

    source_dl = DataLoader(list(zip(source_data, source_labels)), shuffle=True, batch_size=sizing[0])
    target_dl = DataLoader(list(zip(target_data, target_labels)), shuffle=True, batch_size=sizing[2])
    source_dl_test = DataLoader(list(zip(source_data_test, source_labels_test)), shuffle=False, batch_size=sizing[0])
    target_dl_test = DataLoader(list(zip(target_data_test, target_labels_test)), shuffle=False, batch_size=sizing[2])

    return (source_dl, source_dl_test), (target_dl, target_dl_test, target_data_sup, target_labels_sup)


# def batch_generator(data: List[Tensor], batch_size: int, shuffle=True, iter_shuffle=False):
#     data = [torch.stack(d) for d in data]
#     num = data[0].shape[0]
    
#     def shuffle_aligned_list(data):
#         p = np.random.permutation(num)
#         return [d[p] for d in data]

#     if shuffle:
#         data = shuffle_aligned_list(data)

#     batch_count = 0
#     while True:
#         if batch_count * batch_size + batch_size >= num:
#           batch_count = 0
#           if iter_shuffle:
#             print("batch shuffling")
#             data = shuffle_aligned_list(data)

#     start = batch_count * batch_size
#     end = start + batch_size
#     batch_count += 1
#     yield [d[start:end] for d in data]