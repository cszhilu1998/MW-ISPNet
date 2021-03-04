# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import torch
import imageio
import numpy as np
import math
import sys
from mssim import MSSSIM
from load_data import LoadTrainData
from mwcnn_model import MWRCAN, Discriminator
from vgg import vgg_19
from utils import normalize_batch, BaseOptions

to_image = transforms.Compose([transforms.ToPILImage()])

np.random.seed(0)
torch.manual_seed(0)

# Processing command arguments
opt = BaseOptions().parse()

# Dataset size
TRAIN_SIZE = 46839
TEST_SIZE = 1204

def train_model():
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    # Creating dataset loaders
    train_dataset = LoadTrainData(opt.dataroot, TRAIN_SIZE, test=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1,
                              pin_memory=True, drop_last=True)
    test_dataset = LoadTrainData(opt.dataroot, TEST_SIZE, test=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)

    # Creating image processing network and optimizer
    generator = MWRCAN().to(device)
    generator = torch.nn.DataParallel(generator)
    #generator.load_state_dict(torch.load('./ckpt/Track1/mwcnnvggssim4_epoch_60.pth'))

    optimizer = Adam(params=generator.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50,100,150,200], gamma=0.5)

    # Losses
    VGG_19 = vgg_19(device)
    MSE_loss = torch.nn.MSELoss()
    MS_SSIM = MSSSIM()
    L1_loss = torch.nn.L1Loss()

    # Train the network
    for epoch in range(opt.epochs):
        print("lr =  %.8f" % (scheduler.get_lr()[0]))
        torch.cuda.empty_cache()
        generator.to(device).train()
        i = 0
        for x,y in train_loader:
            optimizer.zero_grad()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            enhanced = generator(x)

            loss_l1 = L1_loss(enhanced, y)

            enhanced_vgg = VGG_19(normalize_batch(enhanced))
            target_vgg = VGG_19(normalize_batch(y))
            loss_content = L1_loss(enhanced_vgg, target_vgg)
            
            loss_ssim = MS_SSIM(enhanced, y)

            total_loss = loss_l1 + loss_content + (1 - loss_ssim) * 0.15
            if i%100 == 0:
                print("Epoch %d_%d, L1: %.4f, vgg: %.4f, SSIM: %.4f, total: %.4f" % (epoch, i, loss_l1, loss_content, (1 - loss_ssim) * 0.15, total_loss))
            total_loss.backward()
            optimizer.step()
            i = i+1
        scheduler.step()   

        # Save the model that corresponds to the current epoch
        generator.eval().cpu()
        torch.save(generator.state_dict(), os.path.join(opt.save_model_path, "mwrcan_epoch_" + str(epoch) + ".pth"))      

        # Evaluate the model
        loss_psnr_eval = 0
        generator.to(device)
        generator.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                enhanced = generator(x)
                enhanced = torch.clamp(torch.round(enhanced*255), min=0, max=255) / 255
                y = torch.clamp(torch.round(y*255), min=0, max=255) / 255
                loss_mse_temp = MSE_loss(enhanced, y).item()
                loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))
        loss_psnr_eval = loss_psnr_eval / TEST_SIZE            
        print("Epoch %d, psnr: %.4f" % (epoch, loss_psnr_eval))

if __name__ == '__main__':
    train_model()

