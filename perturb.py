import os
from typing import List
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torchattacks
from torch.utils.data import Dataset
import time
from datetime import timedelta

import fl
from facelibtest import getScores

# PGD settings
epsilon = 4 / 255
epsilon_iter = 1 / 225
nb_iter = 12

# If running on system with GPU use that, otherwise use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# This custom dataset is necessary to use torch's built in batch processing
# All it does is apply a given transform composition to each element in data when batches are being created
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
        return image, 0 # 0 = label

"""Not being used in current pipeline"""
# Generates adversarial image using PGD for a single image
#Takes a PIL image and a string for the set FaceNet should be trained on
#Returns a numpy array (h,w,c) representing an adversarial image
def torchattacks_facenet_pgd(image, pretrain_set):
    # Transform x to be usable by Facenet
    preprocess_ = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(),
        transforms.ToTensor(),
    ])
    image = preprocess_(image)
    image = image.unsqueeze(0).to(device)

    # For a facenet model pretrained on VGGFace2 or casia-webface
    model = InceptionResnetV1(pretrained=pretrain_set).eval().to(device)

    # Run through PGD
    atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon_iter, steps=nb_iter)
    atk.set_return_type(type='int')
    adv_image = atk(image, torch.tensor([0]))[0]

    # Reshape the tensor
    # (c, h, w) --> (h, w, c)
    adv_image = adv_image.permute((1, 2, 0))

    return adv_image

# Generates adversarial images using PGD for a batch
#Takes a batch of images and labels and a string for the set FaceNet should be trained on
#Returns a numpy array (n,h,w,c) of adversarial images
def torchattacks_facenet_pgd_batched(images, labels, pretrain_set):
    #The batched version assumes the batches have been preprocessed

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

# Return list of strings of all algorithms
# The plan was to make the CLI and possibly the cloud integration modular
# If multiple perturbation methods were added, this was how the CLI or cloud would know
# what perturbation methods were available
def methods() -> List[str]:
    return ["torchattacks_facenet_vggface2", "torchattacks_facenet_casiawebface"]

# The 'api' call to perturb a video
# alg -- string representing which algorithhm to use
# images -- Numpy 4D array in format (n, h, w, c)
# n - number of images
# h - height of images
# w - width of images
# c - channels (3)
# faceOnly -- if True, crop to the face, perturb that, and paste back onto video. Else, perturb whole video
# The end state of this function is still undetermined TODO: save a video, return the images, etc
def evaluate(alg: str, original_images, debug=1, faceOnly = False) -> int:
    start_time = time.time() # record time to know how long perturbation pipeline took

    #The only thing that will change based on the alg string is which attack to run
    attacks = {
        "torchattacks_facenet_vggface2": lambda x, y: torchattacks_facenet_pgd_batched(x, y, "vggface2"),
        "torchattacks_facenet_casiawebface": lambda x, y: torchattacks_facenet_pgd_batched(x, y, "casia-webface")
    }

    output_path = "output/test/"
    scaled_path = "output/test_scaled/"

    if debug >= 3:
        index = 0
        for original_image in original_images:
            original_image = Image.fromarray(original_image.astype(np.uint8))
            # x = adv_image.permute((1, 2, 0))
            # display(adv_image, "%sframe%d.jpg" % (output_path, index))
            original_image.save("%sframe%d_original.jpg" % (output_path, index))
            index += 1
            quit()

    # Create the output directory if it does not already exist TODO: we dont want this long term
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    
    #If debug level is >= 2, create scaled to 160x160 versions of every frame and save in scaled_path
    #Used for manual SSIM eval
    if debug >= 2:
        try:
            os.mkdir(scaled_path)
        except FileExistsError:
            pass

        transform = transforms.Compose([
            transforms.Resize(160),
            transforms.CenterCrop(160),
                ])
        scaled_images = [transform(Image.fromarray(img)) for img in original_images]

        index = 0
        for image in scaled_images:
            image.save("%sframe%d.jpg" % (scaled_path, index))
            index += 1
    
    if faceOnly:
        # Crop out the faces for perturbations
        cropped_images, boxes = fl.crop_faces(original_images)

        images = cropped_images
    else:
        # Uses the entire image for perturb
        images = [Image.fromarray(x.astype(np.uint8)) for x in original_images]

    # Transform x to be usable by FaceNet
    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
    ])

    # Torch's batch handling
    # ImageDataset is our custom dataset defined at the top of the file
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
        # save all to folder -- TEMPORARY THIS IS NOT WHAT WE WANT DONE
        index = 0
        filenames = []
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
    
    end_time = time.time()
    print("Time Elapsed in permutations:")
    print(timedelta(seconds=end_time-start_time))
    return 0 # TODO: non-arbitrary return value
