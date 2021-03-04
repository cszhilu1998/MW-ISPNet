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
from torch.autograd import Variable

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
    generator.load_state_dict(torch.load('./ckpt/Track1/mwcnnvggssim4_epoch_60.pth'))
    # generator.load_state_dict(torch.load('./ckpt/Track2/G_epoch_46.pth'))
    disc = Discriminator().to(device)
    disc = torch.nn.DataParallel(disc)
    # disc.load_state_dict(torch.load('./ckpt/Track2/D_epoch_46.pth'))

    optimizer_g = Adam(params=generator.parameters(), lr=opt.lr)
    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, [50,100,150,200], gamma=0.5)
    optimizer_d = Adam(params=disc.parameters(), lr=opt.lr*2)
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, [50,100,150,200], gamma=0.5)

    VGG_19 = vgg_19(device)
    MSE_loss = torch.nn.MSELoss()
    MS_SSIM = MSSSIM()
    L1_loss = torch.nn.L1Loss()

    # Train the network
    for epoch in range(opt.epochs):
        generator.to(device).train()
        disc.to(device).train()
        print("generator lr =  %.8f; discriminator lr =  %.8f" % (scheduler_g.get_lr()[0], scheduler_d.get_lr()[0]))
        torch.cuda.empty_cache()
        i = 0
        for x,y in train_loader:   
            one = Variable(torch.cuda.FloatTensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
            zero = Variable(torch.cuda.FloatTensor(x.shape[0], 1).fill_(0.0), requires_grad=False)        
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer_g.zero_grad()
            enhanced = generator(x)
            fake_label = disc(enhanced).mean()

            loss_l1 = L1_loss(enhanced, y)
            enhanced_vgg = VGG_19(normalize_batch(enhanced))
            target_vgg = VGG_19(normalize_batch(y))
            loss_content = L1_loss(enhanced_vgg, target_vgg)
            loss_ssim = MS_SSIM(enhanced, y)
            adversarial_loss = MSE_loss(one, fake_label)

            g_loss = loss_l1 + loss_content + (1 - loss_ssim) * 0.15 + adversarial_loss * 0.1
            g_loss.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            real_label = disc(y).mean()
            fake_label = disc(enhanced.detach()).mean()
            d_loss = MSE_loss(one, real_label) + MSE_loss(fake_label, zero)
            d_loss.backward()
            optimizer_d.step()

            # Perform the optimization step
            if i%100 == 0:
                #print(loss_ssim)
                print("Epoch %d_%d, L1: %.4f, vgg: %.4f, SSIM: %.4f, adv: %.4f, g_loss: %.4f" %
                      (epoch, i, loss_l1, loss_content, (1 - loss_ssim) * 0.15, adversarial_loss*0.1, g_loss))
                print("Epoch %d_%d, d_loss: %.4f" % (epoch, i, d_loss))
            i = i+1
        
        scheduler_g.step()
        scheduler_d.step()

        # Save the model that corresponds to the current epoch
        generator.eval().cpu()
        disc.eval().cpu()
        torch.save(generator.state_dict(), os.path.join(opt.save_model_path, "g_epoch_" + str(epoch) + ".pth"))
        torch.save(disc.state_dict(), os.path.join(opt.save_model_path, "d_epoch_" + str(epoch) + ".pth"))

        # Evaluate the model
        generator.to(device)
        disc.to(device)
        generator.eval()
        disc.eval()
        loss_psnr_eval = 0
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

