# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import Dataset
from torchvision import transforms
from scipy import misc
import numpy as np
import imageio
import torch
import os
import random

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

def extract_bayer_channels(raw):
    # Reshape the input bayer image
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = np.maximum(RAW_combined.astype(np.float32)-63, 0) / (4 * 255-63)

    return RAW_norm

class LoadTrainData(Dataset):
    def __init__(self, dataset_dir, dataset_size, test=False):
        if test:
            self.raw_dir = os.path.join(dataset_dir, 'test', 'huawei_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'test', 'canon')
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'huawei_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'canon')

        self.dataset_size = dataset_size
        self.test = test

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
        raw_image = extract_bayer_channels(raw_image)
        raw_image = raw_image.transpose((2, 0, 1))

        dslr_image = np.asarray(misc.imread(os.path.join(self.dslr_dir, str(idx) + ".jpg")))
        dslr_image = np.float32(dslr_image) / 255.0
        dslr_image = dslr_image.transpose((2, 0, 1))

        if self.test is False:
            raw_image, dslr_image = self._augment(raw_image, dslr_image)    
        raw_image = torch.from_numpy(raw_image)  
        dslr_image = torch.from_numpy(dslr_image)
        return raw_image, dslr_image

    def _augment(self, *imgs):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5
        def _augment_func(img, hflip, vflip, rot90):
            if hflip:   img = img[:, :, ::-1]
            if vflip:   img = img[:, ::-1, :]
            if rot90:   img = img.transpose(0, 2, 1) # CHW
            return np.ascontiguousarray(img)
        return (_augment_func(img, hflip, vflip, rot90) for img in imgs)

class LoadTestData(Dataset):
    def __init__(self, dataset_dir, dataset_size, denoising=True, fullres=False):
        self.raw_dir = dataset_dir
        self.denoising = denoising
        self.fullres = fullres
        self.list_raw, self.names = self._scan()
        # self.dataset_size = dataset_size
        self.dataset_size = len(self.list_raw)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # if self.fullres is False:
        #     name = str(idx) + '.png'
        # else:
        #     name = str(idx+1) + '.png'
        # raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, name)))
        raw_image = np.asarray(imageio.imread(self.list_raw[idx]))
        if self.denoising:
            raw_image = np.expand_dims(raw_image, 0)  # 1 * H * W
            raw_image = torch.from_numpy(raw_image.astype(np.float32) / (4 * 255))
        else:
            raw_image = extract_bayer_channels(raw_image)
            raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))
        return raw_image, self.names[idx]
    
    def _scan(self):
        list_raw = []
        names = []
        for filename in os.listdir(self.raw_dir):
            list_raw.append(os.path.join(self.raw_dir, filename))
            names.append(filename)
        return list_raw, names

class LoadVisualData(Dataset):
    def __init__(self, data_dir, size, scale, level, full_resolution=False):
        self.raw_dir = os.path.join(data_dir, 'test', 'huawei_full_resolution')
        self.dataset_size = size
        self.scale = scale
        self.level = level
        self.full_resolution = full_resolution
        self.test_images = os.listdir(self.raw_dir)

        self.image_height = 960
        self.image_width = 960

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, self.test_images[idx])))
        raw_image = extract_bayer_channels(raw_image)
        raw_image = raw_image[240:self.image_height + 240, 512:self.image_width + 512, :]
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))
        return raw_image
