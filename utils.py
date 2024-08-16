import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms 
from torch.utils.data import Dataset
from natsort import natsorted
import numpy as np
import tifffile as tiff
from PIL import Image
# from skimage.transform import resize


# def plot_images(images):
#     plt.figure(figsize=(32, 32))
#     plt.imshow(torch.cat([
#         torch.cat([i for i in images.cpu()], dim=-1),
#     ], dim=-2).permute(1, 2, 0).cpu())
#     plt.show()


def save_images(dem_images , satellite_images, path, **kwargs):
    os.makedirs(path, exist_ok=True)

    dem_ndarr = dem_images.permute(0, 2, 3, 1).to('cpu').numpy().astype('>i2')
    satellite_ndarr = satellite_images.permute(0, 2, 3, 1).to('cpu').numpy()

    # Save each channel as a separate TIFF file
    for i in range(satellite_ndarr.shape[0]):
        dem_path = os.path.join(path, f"dem_{i}.bin")
        satellite_path = os.path.join(path, f"satellite_{i}.tif")
        dem_ndarr[i].tofile(dem_path)
        tiff.imwrite(satellite_path, satellite_ndarr[i])

def get_data(args):
    terrain_dataset = TerrainDataset(args.classifier_path, args.DEM_path, args.satellite_path,args.num_samples, args.start_samples, args.end_samples, args.image_size)
    terrain_loader = DataLoader(terrain_dataset, batch_size=args.batch_size, shuffle=True)
    return terrain_loader
    

class TerrainDataset(Dataset):
    
    def __init__(self, classifier_dir, dem_dir, satellite_dir, num_samples, start_samples,end_samples,resize_size):
        self.dem_dir = dem_dir
        self.satellite_dir = satellite_dir
        self.classifier_dir = classifier_dir
        self.num_samples = num_samples
        self.resize_size = resize_size

        self.dem_files = natsorted([f for f in os.listdir(dem_dir)])
        self.satellite_files = natsorted([f for f in os.listdir(satellite_dir)])
        self.classifier_files = natsorted([f for f in os.listdir(classifier_dir)])

        if start_samples:
            self.dem_files = self.dem_files[start_samples:end_samples]
            self.satellite_files = self.satellite_files[start_samples:end_samples]
            self.classifier_files = self.classifier_files[start_samples:end_samples]

        if num_samples:
            self.dem_files = self.dem_files[:num_samples]
            self.satellite_files = self.satellite_files[:num_samples]
            self.classifier_files = self.classifier_files[:num_samples]

        self.resize_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.resize_size,self.resize_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return len(self.dem_files)
    
    def __getitem__(self, idx):
        dem_path = os.path.join(self.dem_dir, self.dem_files[idx])
        satellite_path = os.path.join(self.satellite_dir, self.satellite_files[idx])
        classifier_path = os.path.join(self.classifier_dir, self.classifier_files[idx])
       
        dem_image = np.fromfile(dem_path, dtype='>i2').reshape((64, 64))
        dem_image = np.rot90(np.fliplr(dem_image)) # rotate to correct orientation 
        dem_image = dem_image.byteswap().newbyteorder()  # Convert to Little-Endian 
        # dem_image = resize(dem_image, (self.resize_size, self.resize_size), order=0, preserve_range=True, anti_aliasing=False).astype(dem_image.dtype)
        if dem_image.max() - dem_image.min() == 0:
            dem_image = dem_image - dem_image.min()
        else:
            dem_image = (dem_image - dem_image.min()) / (dem_image.max() - dem_image.min()) * 2 - 1  # Normalize to [-1, 1]
        dem_image = torch.tensor(dem_image).float().unsqueeze(0)  # Add channel dimension
        # print(dem_image.min())
        # print(dem_image.max())
        # print(dem_image.shape)
        
        # Load and preprocess classifier image
        classifier_image = tiff.imread(classifier_path)
        # classifier_image = resize(classifier_image, (self.resize_size, self.resize_size), order=0, preserve_range=True, anti_aliasing=False).astype(classifier_image.dtype)
        classifier_image = classifier_image / 20.0
        classifier_image = classifier_image * 2 - 1 # Normalize to [-1, 1]
        classifier_image = torch.tensor(classifier_image).float().unsqueeze(0)  # Add channel dimension
        # print(classifier_image.min())
        # print(classifier_image.max())
        # print(classifier_image.shape)
       
        # Load and preprocess satellite image
        satellite_image = tiff.imread(satellite_path)
        satellite_image = Image.fromarray(satellite_image)
        satellite_image = self.resize_transform(satellite_image)  # Resize to 64x64
        # print(satellite_image.min())
        # print(satellite_image.max())
        # print(satellite_image.shape)
    
        combined_image = torch.cat([dem_image, satellite_image], dim=0)  # Combine along channel dimension (4, H, W))
        # print(combined_image.shape)
        return combined_image,classifier_image

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


# if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.run_name = "DDPM_Uncondtional2"
    # args.epochs = 20
    # args.batch_size = 8
    # args.image_size = 64
    # args.num_samples = 1
    # args.dataset_path = "patches_64x64"
    # args.device = "cuda"
    # args.lr = 3e-4
    # get_data(args)
    # classifier_dir = r"C:\Users\USER\Desktop\Tiles\Classifier_512"
    # dem_dir = r"C:\Users\USER\Desktop\Tiles\DEM_512"
    # satellitle_dir = r"C:\Users\USER\Desktop\Tiles\Satellite_512"
    # terraindataset = TerrainDataset(classifier_dir,dem_dir,satellitle_dir,1,None,None)
    # combined_image = terraindataset[0]
