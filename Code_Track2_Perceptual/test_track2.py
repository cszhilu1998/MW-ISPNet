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
from utils import BaseOptions
from tqdm import tqdm
from thop import profile

to_image = transforms.Compose([transforms.ToPILImage()])

# Processing command arguments
opt = BaseOptions().parse()
TEST_SIZE = opt.data_size
dataset_dir = opt.dataroot
# model_pth = './ckpt/Track2/G_epoch_46.pth'
model_pth = '/home/zzl/Code/ISP/MW-ISPNet/Code_Track1_Fidelity/ckpt/Track1/mwcnnvggssim4_epoch_60.pth'
# model_pth = '/home/zzl/Code/ISP/MW-ISPNet/Code_Track2_Perceptual/ckpt/Track2/G_epoch_46.pth'

save_pth = opt.save_img_path

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
    device = torch.device("cuda:0")

    # Creating dataset loaders
    test_dataset = LoadTestData(dataset_dir, TEST_SIZE, denoising=False, fullres=opt.fullres)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)

    # Creating and loading pre-trained model
  
    # input = torch.randn(1, 4, 224, 224)
    # macs, params = profile(MWRCAN(), inputs=(input, ))
    # print(macs, params)
    # from thop import clever_format
    # macs, params = clever_format([macs, params], "%.3f")

    model = MWRCAN().to(device)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    # print(net1)
    print('[Network] Total number of parameters : %.3f M'
            % (num_params / 1e6))

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_pth), strict=True)
    model.eval()

    time_val = 0
    with torch.no_grad():
        for j, [raw_image, name] in enumerate(tqdm(test_loader)) :
            #print("Processing image " + str(j))
            #print(name)
            torch.cuda.empty_cache()
            raw_image = raw_image.to(device)
            torch.cuda.synchronize()
            time_val_start = time.time()
            #enhanced = forward_x8(raw_image.detach(), model)
            enhanced = model(raw_image.detach())
            torch.cuda.synchronize()
            time_val += time.time() - time_val_start
      
            enhanced = np.asarray(torch.squeeze(enhanced.float().detach().cpu())).transpose((1,2,0))
            enhanced = np.clip(enhanced*255,0,255).astype(np.uint8)[..., ::-1]
            # if opt.fullres:
            #     cv2.imwrite(save_pth + name[0], enhanced)   
            # else:
            #     cv2.imwrite(save_pth + name[0], enhanced)  
        print('Time:%.4f',time_val/1204*1000)
    
if __name__ == '__main__':
    test_model()
