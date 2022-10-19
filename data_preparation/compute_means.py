from PIL import Image
Image.MAX_IMAGE_PIXELS =  600000000
from torchvision import transforms

import os
import sys

prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(prj_root)

from utils.image_dataset_reader import HistImagesDataset, samples_per_location_from_samples_per_class
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

root_folder = os.path.join(prj_root, 'data/')

normal_images_paths = (

    {'folder': root_folder + "train/mt_mouse_liver", 'label': 'liver', 'ext': 'png'},

)

img_extension = 'png'
patch_size = (256, 256)
n_patches = 6921 #None # None means all # number of images per class


def image_means(normal_images_paths, patch_size, n_patches=None):

    transforms_seq = transforms.Compose([transforms.RandomCrop(patch_size), transforms.ToTensor()])

    n_patches_per_location = None
    if n_patches:
        n_patches_per_location = samples_per_location_from_samples_per_class(*normal_images_paths, samples_per_class=n_patches)

    images_dataset = HistImagesDataset(*normal_images_paths, n_samples=n_patches_per_location, transform=transforms_seq)

    means, stds = comp_means(images_dataset, sum(n_patches_per_location))

    return means, stds


def comp_means(images_dataset, n_patches):

    im_loader = DataLoader(images_dataset, num_workers=0)

    progress = tqdm(im_loader, total=n_patches)

    means = 0
    stds = 0
    n_patches_read = 0
    for samples in progress:
        image = samples['image']

        image = torch.squeeze(image)
        std, mean = torch.std_mean(image, dim=(1, 2))
        means += mean
        stds += std
        n_patches_read += 1

        pass

    means = means / n_patches_read
    stds = stds / n_patches_read
    print('average value: {}, std value: {}, based on {} images'.format(means, stds, n_patches_read))

    return means, stds

#---------------

image_means(normal_images_paths, patch_size, n_patches=n_patches)