import os
from typing import List
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torchattacks
from torch.utils.data import Dataset

# TODO:
# /home/ugrads/majors/jamespur/securecomputing/CapstoneSecureComputing/.venv/lib64/python3.6/site-packages/torchvision/transforms/functional.py:126: 
# UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying 
# (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting 
# it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:189.)
#   img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()

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
            image = self.transform(image).to(device)
        return image, 0

# PGD settings
epsilon = 4 / 255
epsilon_iter = 1 / 225
nb_iter = 12

def torchattacks_facenet_pgd(image, pretrain_set):
    # Transform x to be usable by Facenet
    preprocess_ = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
    ])

    image = preprocess_(image)
    image = image.unsqueeze(0).to(device)

    # For a facenet model pretrained on VGGFace2 or casia-webface
    model = InceptionResnetV1(pretrained=pretrain_set).eval().to(device)

    # Run through PGD
    atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon_iter, steps=nb_iter)
    atk.set_return_type(type='int')
    adv_images = atk(image, torch.tensor([0]))

    # Reshape the tensor
    adv_image = adv_images[0].permute((1, 2, 0))

    return adv_image

def torchattacks_facenet_pgd_batched(images, labels, pretrain_set):
    # Here images is batched and pre-transformed.

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

    """FACELIB TEST CODE"""

    # import fl
    # cropped_images = np.asarray(fl.crop_faces(images))
    # print(cropped_images.shape)

    # Transform x to be usable by Facenet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=160),
        transforms.CenterCrop(size=160),  # TODO: crop to the face rather than arbitrarily in the middle
    ])

    dataset = ImageDataset(images, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # a little sloppy - could probably be better
    # This creates an empty tensor and appends each batch of adversarial images to it
    (c, h, w) = dataset[0][0].shape
    adv_images = torch.empty((0, h, w, c)).to(device)
    for images, labels in dataloader:
        adv_images = torch.cat((adv_images, attacks[alg](images, labels)), dim=0)
        
    # save all to folder -- TEMPORARY THIS IS NOT WHAT WE WANT DONE
    for adv_image in adv_images:
        #x = adv_image.permute((1, 2, 0))
        display(adv_image, "%sframe%d.jpg" % (output_path, index))
        index += 1

    # At this point -- adv_images is a (n,h,w,c) tensor with the adversarial images
    # dataset is a ImageDataset with the original images post transformations contained
    # From here evaluation and reassembly into a video needs to be done
    
    return 0 # TODO: non-arbitrary return value