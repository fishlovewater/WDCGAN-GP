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
from torchvision.utils import make_grid
import random
import glob 
from torchvision.models import vgg16
import time as t
from utils import get_data
from wdcgan128 import weights_init, Generator, Discriminator

# Set random seed for reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define the model.
params = {
    "bsize" : 64,# Batch size during training.
    'imsize' : 128,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 1,# Number of channles in the training images. For coloured images this is 3.RGB
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 20000,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    # 'class' : 1, #0-4
    'Gnorm': 'instance'
}

# Use GPU if available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Get the data.
dataloader = get_data(params)

# Plot the training images.
sample_batch = next(iter(dataloader))
os.makedirs("inputimage",exist_ok=True)
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.savefig("inputimage/inputimage.png")
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

# vgg = vgg16(weights='VGG16_Weights.DEFAULT')

# feature_extractor = nn.Sequential(*list(vgg.features.children())[:8])
# feature_extractor = feature_extractor.to(device)
# for p in feature_extractor.parameters():
#     p.requires_grad = False


# Binary Cross Entropy loss function.
#criterion = nn.BCELoss()
def generator_loss(fake_scores):
    return -fake_scores.mean()
def discriminator_loss(real_scores, fake_scores, gradient_penalty, refu=10):
    # gp_clamped = torch.clamp(gradient_penalty, max=10.0)
    # print(f"grad_penalty: {refu * gp_clamped.item():.4f}")
    print(f"real score: {real_scores.mean()}  fake :{fake_scores.mean()} gp : {10*gradient_penalty.item()}")
    return fake_scores.mean() - real_scores.mean() + refu * gradient_penalty
def compute_gradient_penalty(D, real_data, fake_data,  device='cuda', clamp_norm=True):
    batch_size = real_data.size(0)
    
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)

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
        print(data[0].shape)
        real_data = data[0].to(device)
        b_size = real_data.size(0)
        for param in netD.parameters():
                param.requires_grad = True
        netD.zero_grad()

        
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        fake_data = netG(noise)

        real_scores = netD(real_data).view(-1)
        fake_scores = netD(fake_data).view(-1)

        gradient_penalty = compute_gradient_penalty(
            netD, real_data, fake_data, device
        )
        torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=10)
        errD = discriminator_loss(real_scores, fake_scores, gradient_penalty)
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
        netG.zero_grad()

        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        fake_data = netG(noise)
        fake_scores = netD(fake_data).view(-1)
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=10)

        # if real_data.shape[1] == 1:
        #     real_data = real_data.repeat(1, 3, 1, 1)
        #     fake_data = fake_data.repeat(1, 3, 1, 1)


        features_real = netD.feature_extractor(real_data)
        features_fake = netD.feature_extractor(fake_data)
        fm_loss = F.mse_loss(features_fake, features_real.detach())

        # fm_loss = torch.norm(features_real.detach() - features_fake, p=2)
        print(f"fm_loss : {fm_loss.item()/500}")
        fm_loss = torch.clamp(fm_loss, max=801.0)
        print(f"modify fm_loss : {fm_loss.item()}")
        errG = generator_loss(fake_scores)
        #generator_loss(fake_scores)

        # if epoch >= 1000 :
        errG += fm_loss/500
        errG.backward()
        optimizerG.step()

        err_g+=errG
            
        D_G_z2 = fake_scores.mean().item()

        
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
    err_avg_D = err_avg_D.item()
    D_losses.append(err_avg_D)
    err_avg_G = err_g/len(dataloader)
    err_avg_G = err_avg_G.item()
    G_losses.append(err_avg_G)

    if (epoch+1) %50 == 0:
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[GPU] Memory Allocated: {mem_alloc:.1f} MB, Reserved: {mem_reserved:.1f} MB")


    if (epoch+1) % 100 == 0:
        model_path = f'model/model_epoch{epoch}.pth'
        print(f"= 儲存模型於 epoch {epoch}=")
        torch.save({
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'params': params
        },model_path)
    if ((epoch+1) % 100 == 0) or epoch == 1 :
        os.makedirs("outputimage", exist_ok=True)

        num_sample = 9
        z = torch.randn(num_sample,100, device=device).view(num_sample,100, 1, 1)
        # labels = torch.zeros(num_sample, dtype=torch.long, device=device)  # label=0

        sample_images = netG(z).detach().cpu()

        grid = make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
        plt.imshow(grid.squeeze(), cmap='gray')  
        plt.axis('off')
        plt.savefig(f"outputimage/outputimage_{epoch}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved sample image: outputimage/outputimage_{epoch}.png")

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