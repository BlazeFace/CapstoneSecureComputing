import os
from typing import List
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torchattacks
from torch.utils.data import Dataset

import fl
from facelibtest import getScores

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
    # atk.set_mode_targeted_least_likely(500)
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
def evaluate(alg: str, original_images, debug=1, faceOnly=False) -> int:
    attacks = {
        "torchattacks_facenet_vggface2": lambda x, y: torchattacks_facenet_pgd_batched(x, y, "vggface2"),
        "torchattacks_facenet_casiawebface": lambda x, y: torchattacks_facenet_pgd_batched(x, y, "casia-webface")
    }
    index = 0
    output_path = "output/test/"

    if debug >= 2:
        for original_image in original_images:
            original_image = Image.fromarray(original_image.astype(np.uint8))
            # x = adv_image.permute((1, 2, 0))
            # display(adv_image, "%sframe%d.jpg" % (output_path, index))
            original_image.save("%sframe%d_original.jpg" % (output_path, index))
            index += 1
            quit()
        index = 0

    # Create the output directory if it does not already exist TODO: we dont want this long term
    try:
        os.mkdir(output_path)
    except FileExistsError:
        var = None
        # intentionally left blank

    if faceOnly:
        # Crop out the faces for perturbations
        cropped_images, boxes = fl.crop_faces(original_images)
        # cropped_images = [np.array(img) for img in cropped_images]
        # cropped_images = np.asarray(cropped_images)

        images = cropped_images
    else:
        # Uses the entire image for perturb
        images = [Image.fromarray(x.astype(np.uint8)) for x in original_images]

    # Transform x to be usable by Facenet
    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(images, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # a little sloppy - could probably be better
    # This creates an empty tensor and appends each batch of adversarial images to it
    (c, h, w) = dataset[0][0].shape
    adv_images = torch.empty((0, h, w, c)).to(device)
    for images, labels in dataloader:
        adv_images = torch.cat((adv_images, attacks[alg](images, labels)), dim=0)

    if faceOnly:
        # put adversarial cropped face back on original images
        # returns a list of PIL images
        adv_images = fl.restore_images(original_images, adv_images.cpu().numpy(), boxes)
    else:
        # Change all the tensors in adv_images to PIL images for saving / evaluating
        adv_images = [Image.fromarray(x.detach().cpu().numpy().astype(np.uint8)) for x in adv_images]

    if debug:
        filenames = []

        # save all to folder -- TEMPORARY THIS IS NOT WHAT WE WANT DONE
        for adv_image in adv_images:
            # x = adv_image.permute((1, 2, 0))
            # display(adv_image, "%sframe%d.jpg" % (output_path, index))
            adv_image.save("%sframe%d.jpg" % (output_path, index))
            filenames.append("%sframe%d.jpg" % (output_path, index))
            index += 1

        # temporary method of evaluating the images
        total_faces, total_scores, total_boxes, total_landmarks = getScores(filenames, False)
        print("Detected: %d" % len(total_scores))
        if len(total_scores) > 0:
            for score in total_scores:
                if len(score) > 0:
                    print(score[0].item())
                else:
                    print("no score")

    # At this point -- adv_images is a list of PIL images with the adversarial images
    # original_images is a numpy array of size (n, h, w, c) of the original images
    # From here evaluation and reassembly into a video needs to be done

    return 0  # TODO: non-arbitrary return value
