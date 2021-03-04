import numpy as np
import sys
import argparse
import os
import re
import torch
import time

def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't', '1')

inf = float('inf')

class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # data parameters
        parser.add_argument('--dataroot', type=str, default='/share/Dataset/Zurich-RAW-to-DSLR/Zurich-RAW-to-DSLR/')
        parser.add_argument('--data_size', type=int, default=1342)
        parser.add_argument('--batch_size', type=int, default=24)
        #parser.add_argument('--patch_size', type=int, default=None)
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--lr', type=float, default=0.0001)

        # device parameters
        parser.add_argument('--gpu_ids', type=str, default='all',
                help='Separate the GPU ids by `,`, using all GPUs by default. '
                     'eg, `--gpu_ids 0`, `--gpu_ids 2,3`, `--gpu_ids -1`(CPU)')

        parser.add_argument('--save_model_path', type=str, default='./ckpt')
        parser.add_argument('--load_path', type=str, default='')
        parser.add_argument('--save_img_path', type=str, default='./results')
        parser.add_argument('--fullres', type=str2bool, default=False)
        #parser.add_argument('--name', type=str, required=True, help='Name of the folder to save models and logs.')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are difined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=
                         argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # get the basic options
        opt, _ = parser.parse_known_args()
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        mkdirs(opt.save_model_path)
        mkdirs(opt.save_img_path)
        # file_name = os.path.join(opt.save_model_path, 'opt_%s.txt'
        #         % ('train' if self.isTrain else 'test'))
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')
    def parse(self):
        opt = self.gather_options()
        self.print_options(opt)
        # set gpu ids
        cuda_device_count = torch.cuda.device_count()
        if opt.gpu_ids == 'all':
            # GT 710 (3.5), GT 610 (2.1)
            gpu_ids = [i for i in range(cuda_device_count)]
        else:
            p = re.compile('[^-0-9]+')
            gpu_ids = [int(i) for i in re.split(p, opt.gpu_ids) if int(i) >= 0]
        opt.gpu_ids = [i for i in gpu_ids \
                       if torch.cuda.get_device_capability(i) >= (4,0)]

        if len(opt.gpu_ids) == 0 and len(gpu_ids) > 0:
            opt.gpu_ids = gpu_ids
            prompt('You\'re using GPUs with computing capability < 4')
        elif len(opt.gpu_ids) != len(gpu_ids):
            prompt('GPUs(computing capability < 4) have been disabled')

        if len(opt.gpu_ids) > 0:
            assert torch.cuda.is_available(), 'No cuda available !!!'
            torch.cuda.set_device(opt.gpu_ids[0])
            print('The GPUs you are using:')
            for gpu_id in opt.gpu_ids:
                print(' %2d *%s* with capability %d.%d' % (
                        gpu_id,
                        torch.cuda.get_device_name(gpu_id),
                        *torch.cuda.get_device_capability(gpu_id)))
        else:
            prompt('You are using CPU mode')

        self.opt = opt
        return self.opt

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, mid = %3.3f, std=%3.3f'
              % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def prompt(s, width=66):
    print('='*(width+4))
    ss = s.split('\n')
    if len(ss) == 1 and len(s) <= width:
        print('= ' + s.center(width) + ' =')
    else:
        for s in ss:
            for i in split_str(s, width):
                print('= ' + i.ljust(width) + ' =')
    print('='*(width+4))

def split_str(s, width):
    ss = []
    while len(s) > width:
        idx = s.rfind(' ', 0, width+1)
        if idx > width >> 1:
            ss.append(s[:idx])
            s = s[idx+1:]
        else:
            ss.append(s[:width])
            s = s[width:]
    if s.strip() != '':
        ss.append(s)
    return ss

def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std
