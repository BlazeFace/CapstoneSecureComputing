import os
import random
from typing import List
import requests
from io import BytesIO
import numpy as np
import PIL
from PIL import Image
from numpy.random import default_rng
from matplotlib import pyplot as plt
import math
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from facenet_pytorch import InceptionResnetV1
import torchattacks
from facelibtest import getScores
from torch.utils.data import Dataset

# If running on system with GPU use that, otherwise use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# This custom dataset is necessary to use torch's built in multithreading
# on our dataset of a list of frames
# we are ignoring the labels for now TODO: maybe there's a better way to handle that
class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)#.to(device)
        return image, 0

# PGD settings
epsilon = 4 / 255
epsilon_iter = 1 / 225
nb_iter = 12

def torchattacks_facenet_pgd(x, pretrain_set):
    # Transform x to be usable by Facenet
    preprocess_ = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
    ])

    x = preprocess_(x)
    x = x.unsqueeze(0).to(device)

    # For a facenet model pretrained on VGGFace2 or casia-webface
    model = InceptionResnetV1(pretrained=pretrain_set).eval().to(device)

    # Run through PGD
    atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon_iter, steps=nb_iter)
    atk.set_return_type(type='int')
    adv_images = atk(x, torch.tensor([0]))

    # Reshape the tensor
    x = adv_images[0].permute((1, 2, 0))

    return x

def torchattacks_facenet_pgd_batched(images, labels, pretrain_set):
    # Here x is batched and pre-transformed.

    # For a facenet model pretrained on VGGFace2 or casia-webface
    model = InceptionResnetV1(pretrained=pretrain_set).eval().to(device)

    # Run through PGD
    atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon_iter, steps=nb_iter)
    atk.set_return_type(type='int')
    adv_images = atk(images, labels)

    # Reshape the tensor
    #  (n, c, h, w) --> (n, h, w, c)
    adv_images = adv_images.permute((0, 2, 3, 1))

    return adv_images

# Helper function to display an image (TEMPORARY -- THIS SHOULD NOT BE A PART OF THE PIPELINE)
def display(x, filename):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()  # put tensor on CPU
    img = Image.fromarray(x.astype(np.uint8)).convert('RGB')
    img.save(filename)

# Return list of strings of all algorithms
def methods() -> List[str]:
    return ["torchattacks_facenet_vggface2", "torchattacks_facenet_casiawebface"]

# alg -- string
# images -- Numpy 4D array in format (n, h, w, c)
# n - number of images
# h - height of images
# w - width of images
# c - channels (3)
# return score of effectiveness TODO: maybe not
def evaluate(alg: str, images) -> int:
    attacks = {
        "torchattacks_facenet_vggface2": lambda x, y: torchattacks_facenet_pgd_batched(x, y, "vggface2"),
        "torchattacks_facenet_casiawebface": lambda x, y: torchattacks_facenet_pgd_batched(x, y, "casiawebface")
    }
    index = 0
    output_path = "output/test/"

    # Create the output directory if it does not already exist TODO: we dont want this long term
    try:
        os.mkdir(output_path)
    except FileExistsError:
        None
        # intentionally left blank

    # Transform x to be usable by Facenet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=160),
        transforms.CenterCrop(size=160),  # TODO: crop to the face rather than arbitrarily in the middle
    ])

    dataset = ImageDataset(images, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    for images, labels in dataloader:
        adv_images = attacks[alg](images, labels)

        # save all to folder
        for adv_image in adv_images:
            #x = adv_image.permute((1, 2, 0))
            display(adv_image, "%sframe%d.jpg" % (output_path, index))
            index += 1

    return 0