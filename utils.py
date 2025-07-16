import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

# Directory containing the data.
root = 'data/bulldozer'
def get_data(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.

    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create the dataset.
    dataset = dset.ImageFolder(root=root, transform=transform)
    #is dictionary
    with open(r"C:\Users\user\OneDrive\桌面\WDCGAN-GP\class.txt", 'w', encoding='utf-8') as f:
        for class_name, class_idx in dataset.class_to_idx.items():
            f.write(f"{class_name}: {class_idx}\n")
    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader