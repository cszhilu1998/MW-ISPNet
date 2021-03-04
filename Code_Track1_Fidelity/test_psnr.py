# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import imageio
import cv2
import time
from load_data import LoadTestData
from mwcnn_model import MWRCAN
from ckpt.raw_denoising.code.urcan import Net
from utils import BaseOptions
from tqdm import tqdm
import math

def calc_psnr(sr, hr):
    diff = (sr.astype(np.float32) - hr.astype(np.float32)) / 255.
    mse = np.power(diff, 2).mean()
    return -10 * math.log10(mse)

def test_psnr():
    dir_re = '/home/zzl/Code/ISP/MW-ISPNet/Code_Track1_Fidelity/results/Track1/track1_x8'
    dir_re_de = '/home/zzl/Code/ISP/MW-ISPNet/Code_Track1_Fidelity/results/Track1/track1'
    dir_gt = '/data/dataset/Zurich-RAW-to-DSLR/test/canon'
    psnr = [0.0] * 1204
    psnr_de = [0.0] * 1204

    for i in tqdm(range(0, 1204)):
        re = cv2.imread(os.path.join(dir_re, str(i) + ".png"), cv2.IMREAD_COLOR)
        re_de = cv2.imread(os.path.join(dir_re_de, str(i) + ".png"), cv2.IMREAD_COLOR)
        gt = cv2.imread(os.path.join(dir_gt, str(i) + ".jpg"), cv2.IMREAD_COLOR)
        #print(re.shape)
        #print(re_de.shape)
        #print(gt.shape)
        psnr[i] = calc_psnr(re, gt)
        psnr_de[i] = calc_psnr(re_de, gt)
    print('psnr=%.6f, denoising_psnr=%.6f'%(np.mean(psnr), np.mean(psnr_de)))


if __name__ == '__main__':
    test_psnr()
