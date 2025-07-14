import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import transforms
import random
import os
import glob 
#計算通訊成本 時間\
import time as t
from utils import get_data
from wdcgan512 import weights_init, Generator, Discriminator

# Set random seed for reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define the model.
params = {
    "bsize" : 4,# Batch size during training.
    'imsize' : 512,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.RGB
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 1000,# Number of training epochs.
    'lr' : 0.0001,# Learning rate for optimizers
    'class' : 1 #0-4
}

# Use GPU if available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Get the data.
dataloader = get_data(params)

# Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()

# Create the generator.
netG = Generator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netG.apply(weights_init)
# Print the model.
print(netG)

# Create the discriminator.
netD = Discriminator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netD.apply(weights_init)
# Print the model.
print(netD)


# Binary Cross Entropy loss function.
#criterion = nn.BCELoss()
def generator_loss(fake_scores):
    return -fake_scores.mean()
def discriminator_loss(real_scores, fake_scores, gradient_penalty, refu=1):
    print(f"grad_penalty: {gradient_penalty.item()*refu:.4f}")
    gp_clamped = torch.clamp(gradient_penalty, max=10.0)
    return fake_scores.mean() - real_scores.mean() + refu * gp_clamped
def compute_gradient_penalty(D, real_data, fake_data, label, device='cuda', clamp_norm=True):
    batch_size = real_data.size(0)
    
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates, label)

    fake = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    if clamp_norm:
        gradient_norm = gradient_norm.clamp(0, 10)

    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


# Optimizer for the discriminator.
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(0.0,0.9))
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(0.0,0.9))

# Optimizer for the generator.
# Stores generated images as training progresses.
img_list = []
# Stores generator losses during training.

G_losses = []
# Stores discriminator losses during training.
D_losses = []
Wasserstein_D = []

iters = 0

print("Starting Training Loop...")
print("-"*25)

# 初始化最佳 loss 差值為無限大
best_loss_gap = float('inf')
start_t = t.time()
for epoch in range(params['nepochs']):
    Wasserstein = 0.0
    err_D=0.0
    err_g=0.0
    dataloader_iter = iter(dataloader)
    for i, data in enumerate(dataloader, 0):
        # real_data = data[0].to(device)
        # real_labels = data[1].to(device)
        # b_size = real_data.size(0)
        for k in range (5) :

            try:
                real_data, real_labels = next(dataloader_iter)
            except:
                dataloader_iter = iter(dataloader)
                real_data, real_labels = next(dataloader_iter)

            real_data = real_data.to(device)
            real_labels = real_labels.to(device)
            b_size = real_data.size(0)
            netD.zero_grad()
            # real data score
            real_scores = netD(real_data, real_labels).view(-1)
            #to make sure label same with real
            fake_labels = real_labels.clone()
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            fake_data = netG(noise, fake_labels)

            # fake data score
            fake_scores = netD(fake_data.detach(), fake_labels).view(-1)

            # Gradient Penalty
            gradient_penalty = compute_gradient_penalty(
            netD,
            real_data.detach(),
            fake_data.detach(),
            real_labels,
            device
            )


            # Discriminator  loss (Wasserstein loss + gradient penalty)
            errD = discriminator_loss(real_scores, fake_scores, gradient_penalty)
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=10.0)
            

            errD.backward()
            optimizerD.step()

        # score for print
        D_x = real_scores.mean().item()
        D_G_z1 = fake_scores.mean().item()
        Wasserstein  += (D_x - D_G_z1)
        err_D+=errD

        #freeze d
        for param in netD.parameters():
            param.requires_grad = False
       	
       	
        # Make accumalted gradients of the generator zero.

#         # 取 real/fake 的中間特徵
# features_real = netD.feature_extractor(real_data, real_labels)
# features_fake = netD.feature_extractor(fake_data, fake_labels)

# # 計算 L1 或 L2 距離
# fm_loss = F.l1_loss(features_fake, features_real.detach())
# # 或用 F.mse_loss(...)

# # 加進 generator loss（加權合併）
# errG = generator_loss(fake_scores) + alpha * fm_loss

        netG.zero_grad()

        # We want the fake data to be classified as real. Hence
        # real_label are used. (label=1)
        #label.fill_(real_label)
        # No detach() is used here as we want to calculate the gradients w.r.t.
        # the generator this time.
        fake_scores = netD(fake_data, fake_labels).view(-1)
        features_real = netD.feature_extractor(real_data, real_labels)
        features_fake = netD.feature_extractor(fake_data, fake_labels)
        fm_loss = F.l1_loss(features_fake, features_real.detach())
        errG = generator_loss(fake_scores) + 10 * fm_loss
        print((10 * fm_loss).item())
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=10.0)
        # Gradients for backpropagation are calculated.
        # Gradients w.r.t. both the generator and the discriminator
        # parameters are calculated, however, the generator's optimizer
        # will only update the parameters of the generator. The discriminator
        # gradients will be set to zero in the next iteration by netD.zero_grad()
        errG.backward()# will release computation graph
            #torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            
        # Update generator parameters.
        optimizerG.step()
        err_g+=errG
            
        D_G_z2 = fake_scores.mean().item()
        # fd
        for param in netD.parameters():
            param.requires_grad = True
        
        if i % 1 == 0:
            del real_data, fake_data, real_scores, fake_scores, gradient_penalty, noise, features_real, features_fake
            torch.cuda.empty_cache()


        # Check progress of training.
        if i%1 == 0:
            print(torch.cuda.is_available())
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save the losses for plotting.
        # G_losses.append(errG.item())
        # D_losses.append(errD.item())

        iters += 1

    avg_wasserstein = Wasserstein/len(dataloader)
    Wasserstein_D.append(avg_wasserstein)
    err_avg_D = err_D/len(dataloader)
    D_losses.append(err_avg_D)
    err_avg_G = err_g/len(dataloader)
    G_losses.append(err_avg_G)

    if epoch % 100 == 0:
        model_path = f'model/model_epoch{epoch}.pth'
        print(f"= 儲存模型於 epoch {epoch}=")
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        },model_path)

end_t = t.time()
total_t = end_t - start_t
print(total_t//3600)
# Plot the training losses.
import os
os.makedirs("outputloss", exist_ok=True)

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.plot(Wasserstein_D,label="Wasserstein_D")
plt.xlabel("epochs")
plt.ylabel("Wasserstein_D")
plt.legend()
plt.savefig("outputloss/loss.png")
plt.show()

# Animation showing the improvements of the generator.
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000, blit=True)
plt.show()
anim.save('result.gif', dpi=80, writer='imagemagick')
