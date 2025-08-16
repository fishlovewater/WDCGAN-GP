import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image  # 為了存圖

from wdcgan256 import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_epoch14899.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', type=int, default=100, help='Number of generated outputs')
parser.add_argument('-out_dir', default='output', help='Output directory to save images')  # 新增輸出資料夾參數
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path, map_location='cpu')

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
netG.load_state_dict(state_dict['generator'])
netG.eval()
print(netG)

print(f"Generating {args.num_output} images...")

# Get latent vector Z from unit normal distribution.
noise = torch.randn(100, params['nz'], 1, 1, device=device)
# fake_labels = torch.arange(params['class'], device=device).repeat_interleave(100)  # 每類重複 10 次



# Generate images
with torch.no_grad():
    generated_imgs = netG(noise).detach().cpu()

# Create output directory if not exists
os.makedirs(args.out_dir, exist_ok=True)

# Save each image individually
for idx, img in enumerate(generated_imgs):
    img = (img * 0.5 + 0.5).clamp(0, 1)  # Unnormalize from [-1,1] to [0,1]
    np_img = (img.numpy() * 255).astype(np.uint8)

    if np_img.shape[0] == 1:  # 灰階圖
        np_img = np_img.squeeze(0)  # [1, H, W] → [H, W]
        img_pil = Image.fromarray(np_img, mode='L')
    else:  # 彩色圖
        np_img = np.transpose(np_img, (1, 2, 0))  # [C, H, W] → [H, W, C]
        img_pil = Image.fromarray(np_img)

    img_path = os.path.join(args.out_dir, f"gen_{idx+1:03d}.png")
    img_pil.save(img_path)
    print(f"✅ Saved {img_path}")


# Display the generated image as a grid
plt.axis("off")
plt.title("Generated Images Grid")
plt.imshow(np.transpose(vutils.make_grid(generated_imgs, padding=2, normalize=True), (1, 2, 0)))
plt.show()

