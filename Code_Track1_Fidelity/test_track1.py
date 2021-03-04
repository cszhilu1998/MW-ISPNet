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

to_image = transforms.Compose([transforms.ToPILImage()])

# Processing command arguments
opt = BaseOptions().parse()
TEST_SIZE = opt.data_size
dataset_dir = opt.dataroot
pre_denoising = opt.pre_denoising
model_pth = './ckpt/Track1/mwcnnvggssim4_epoch_60.pth'
save_pth = opt.save_img_path

def extract_bayer_channels(raw):
    # Reshape the input bayer image
    ch_B  = raw[..., 1::2, 1::2]
    ch_Gb = raw[..., 0::2, 1::2]
    ch_R  = raw[..., 0::2, 0::2]
    ch_Gr = raw[..., 1::2, 0::2]
    RAW_combined = torch.cat((ch_B, ch_Gb, ch_R, ch_Gr), dim=1)
    RAW_norm = torch.clamp(RAW_combined*255*4-63, min=0) / (4 * 255-63)
    return RAW_norm

def forward_x8(x, forward_function):
    def _transform(v, op):
        v = v.float()
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        ret = torch.Tensor(tfnp).cuda()
        #ret = ret.half()
        return ret

    lr_list = [x]
    for tf in 'v', 'h', 't':
        lr_list.extend([_transform(t, tf) for t in lr_list])

    sr_list = [forward_function(aug) for aug in lr_list]
    for i in range(len(sr_list)):
        if i > 3:
            sr_list[i] = _transform(sr_list[i], 't')
        if i % 4 > 1:
            sr_list[i] = _transform(sr_list[i], 'h')
        if (i % 4) % 2 == 1:
            sr_list[i] = _transform(sr_list[i], 'v')
    
    output_cat = torch.cat(sr_list, dim=0)
    output = output_cat.mean(dim=0, keepdim=True)
    return output

def test_model():
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    # Creating dataset loaders
    test_dataset = LoadTestData(dataset_dir, TEST_SIZE, denoising=pre_denoising, fullres=opt.fullres)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)

    # Creating and loading pre-trained model
    model_0 = MWRCAN().to(device)
    model_0 = torch.nn.DataParallel(model_0)
    model_0.load_state_dict(torch.load(model_pth), strict=True)
    model_0.eval()
    
    if pre_denoising:
        denoising_model = Net().to(device)
        denoising_model = torch.nn.DataParallel(denoising_model)
        denoising_model.load_state_dict(
                        torch.load('./ckpt/raw_denoising/raw_denoising.pth')['state_dict'], strict=True)
        denoising_model.eval()
        #model.half()

    time_val = 0
    with torch.no_grad():
        for j, raw_image in enumerate(tqdm(test_loader)) :
            #print("Processing image " + str(j))
            torch.cuda.empty_cache()          
            raw_image = raw_image.to(device)

            time_val_start = time.time()
            if pre_denoising:
                denoising_raw = denoising_model(raw_image.detach())      
                denoising_raw = extract_bayer_channels(denoising_raw)
                enhanced = model_0(denoising_raw.detach())# forward_x8(denoising_raw.detach(), model_0)
            else:
                enhanced = model_0(raw_image.detach(), model_0) # model_0(raw_image)# forward_x8
            time_val += time.time() - time_val_start    

            enhanced = np.asarray(torch.squeeze(enhanced.float().detach().cpu())).transpose((1,2,0))
            enhanced = np.clip(enhanced*255,0,255).astype(np.uint8)[..., ::-1]
            if opt.fullres:
                cv2.imwrite(save_pth + str(j+1) + ".png", enhanced)   
            else:
                cv2.imwrite(save_pth + str(j) + ".png", enhanced)      
        print('Time:%.4f',time_val/TEST_SIZE)
    
if __name__ == '__main__':
    test_model()
