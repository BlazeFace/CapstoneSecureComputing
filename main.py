import random
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
from torchvision import transforms
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from facenet_pytorch import InceptionResnetV1
import torchattacks
from facelibtest import getScores

device = "cuda" if torch.cuda.is_available() else "cpu"

mobilenet_mean=[0.485, 0.456, 0.406]
mobilenet_std=[0.229, 0.224, 0.225]

class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

# FaceLib outputs scores so we can use the same L function as Yolo
def loss(z, m):
    loss_sum = 0
    for j in range(0, m):
        loss_sum += np.log(z)
    return loss_sum

def mask(mask1, org):
    with np.nditer(mask1, flags=['multi_index'], op_flags=['readwrite']) as it:
        for m_x in it:
            if random.randint(1, 1000) <= 1:
                mask1[it.multi_index] = 1
                if org is not None:
                    org[it.multi_index] = 0
    return mask1, org

def cleverhans_mobilenet_pgd(x):

    # Transform x to be usable by Mobilenet
    preprocess_ = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    x = preprocess_(x)
    x = x.unsqueeze(0).to(device)

    #Acquire the model to do PGD on
    model = torch.hub.load('pytorch/vision:v0.8.0', 'mobilenet_v2', pretrained=True) # Facelib uses the Mobilenet model
    model.to(device)

    norm_layer = Normalize(mean=mobilenet_mean, std=mobilenet_std)

    model = nn.Sequential(
        norm_layer,
        model
    ).to(device)

    model = model.eval()

    # Run through PGD
    epsilon = 8/255
    epsilon_iter = 2/225
    nb_iter = 100
    x = projected_gradient_descent(model, x, epsilon, epsilon_iter, nb_iter, np.inf)

    # Reshape the tensor
    x = x[0].permute((1,2,0))

    #scale tensor
    x = x * 255

    return x

def cleverhans_facenet_pgd(x, pretrain_set):

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
    epsilon = 8/255
    epsilon_iter = 2/225
    nb_iter = 100
    x = projected_gradient_descent(model, x, epsilon, epsilon_iter, nb_iter, np.inf)

    # Reshape the tensor
    x = x[0].permute((1,2,0))

    #scale tensor
    x = x * 255
    
    return x

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
    epsilon = 8/255
    epsilon_iter = 2/225
    nb_iter = 100
    atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon_iter, steps=nb_iter)
    atk.set_return_type(type='int')
    adv_images = atk(x, torch.tensor([0]))

    # Reshape the tensor
    x = adv_images[0].permute((1,2,0))

    return x

#Projected Gradient Descent
def pgdv1(x):
    x = np.array(x)
    SHAPE = x.shape
    random.seed()

    #Create mask
    m, org = mask(mask1=np.zeros(SHAPE), org=x.copy())

    #Create Delta
    rng = default_rng()
    ranints = rng.integers(low=0, high=255, size=SHAPE)
    delta = np.multiply(ranints, m)
    delta = np.where(delta > 5, delta % 256, delta)
    
    #Display Delta
    plt.imshow(delta, interpolation='nearest')
    plt.savefig("delta.jpeg")
    plt.figure(figsize=(20, 4))
    delta = np.absolute(delta)
    
    #Apply delta to the image
    dm = np.multiply(delta, m)
    array = dm + org
    array = array.astype(np.uint8)
    array = np.where(array > 256, array%256, array )
    return array

def pgdv2(x):
    #x = torch.tensor(np.array(x)).to(device)
    x = mobilenet_preprocess(x).float()
    
    SHAPE = x.shape
    random.seed()
    # PGD settings as per paper
    p = 40 # number of PGD iterations P
    alpha = 16/255 # PGD step size α

    #Acquire the model to do PGD on
    f = torch.hub.load('pytorch/vision:v0.8.0', 'mobilenet_v2', pretrained=True) # Facelib uses the Mobilenet model
    f.to(device)

    #Create mask (Random mask in place of Half-Neighbor for now)
    m, _ = mask(mask1=np.zeros(SHAPE), org=None)
    m = torch.tensor(m).float().to(device)

    #Create Delta -- Random initialize 𝛿
    rng = default_rng()
    ranints = rng.integers(low=0, high=255, size=SHAPE)
    ranints = torch.tensor(ranints).to(device)

    # apply delta to mask -- 𝛿 = 𝛿 × M
    delta = torch.mul(ranints, m) 
    delta = torch.where(delta > 5, delta % 256, delta)
    
    # for 𝑖 = 1 . . . 𝑃 do
    for i in range(1,p+1):
        # 𝛿 = 𝛿 + 𝛼 · sign (∇𝛿𝐿 (𝑓 (𝑥 + 𝛿)))
        fres = f(x + delta)
        delta = torch.add(delta, (torch.mul(alpha, torch.sign(loss(fres))))) 
        #𝛿 = 𝛿 × 𝑀
        delta = torch.mul(delta, m)
        # 𝛿 = max(min(𝛿, 0 − 𝑥), 1 − 𝑥)
        delta = torch.max(torch.min(delta, 0 - x), 1 - x)

    return delta

def display(x, filename):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy() # put tensor on CPU
    img = Image.fromarray(x.astype(np.uint8)).convert('RGB')
    img.save(filename)

def createPermutation(filename, url=None):
    if url is None:
        img = PIL.Image.open(filename)
    else:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save("input/" + filename + ".jpg")

    filenames = ["input/" + filename + ".jpg"]
    filename = "output/" + filename.split(".")[0]
    
    # x_v1 = pgdv1(img)
    # display(x_v1, "output_v1.jpeg")

    x_cleverhans_mobilenet = cleverhans_mobilenet_pgd(img)
    display(x_cleverhans_mobilenet, filename + "_cleverhans_mobilenet.jpeg")
    filenames.append(filename + "_cleverhans_mobilenet.jpeg")

    x_cleverhans_facenet_vggface2 = cleverhans_facenet_pgd(img, "vggface2")
    display(x_cleverhans_facenet_vggface2, filename + "_cleverhans_facenet_vggface2.jpeg")
    filenames.append(filename + "_cleverhans_facenet_vggface2.jpeg")

    x_cleverhans_facenet_casiawebface = cleverhans_facenet_pgd(img, "casia-webface")
    display(x_cleverhans_facenet_casiawebface, filename + "_cleverhans_facenet_casiawebface.jpeg")
    filenames.append(filename + "_cleverhans_facenet_casiawebface.jpeg")

    x_torchattacks_facenet_vggface2 = torchattacks_facenet_pgd(img, "vggface2")
    display(x_torchattacks_facenet_vggface2, filename + "_torchattacks_facenet_vggface2.jpeg")
    filenames.append(filename + "_torchattacks_facenet_vggface2.jpeg")

    x_torchattacks_facenet_casiawebface = torchattacks_facenet_pgd(img, "casia-webface")
    display(x_torchattacks_facenet_casiawebface, filename + "_torchattacks_facenet_casiawebface.jpeg")
    filenames.append(filename + "_torchattacks_facenet_casiawebface.jpeg")

    print(filenames)
    scores = getScores(filenames)
    
    print("="*30 + "\n")
    for filename,score in zip(filenames, scores):
        print("File: %s" % filename)
        print("Score: %s" % score)
    print("="*30 + "\n")

#createPermutation(filename="output", url="https://thoughtcatalog.com/wp-content/uploads/2018/06/oopsface.jpg?w=1140")

images = {'oops':'https://thoughtcatalog.com/wp-content/uploads/2018/06/oopsface.jpg?w=1140',
'pose':'https://images.hivisasa.com/1200/It9Rrm02rE20.jpg',
'elderlycouple':'https://www.latrobe.edu.au/jrc/jri-images/empowering-older-people.jpg/1680.jpg'}

for key,value in images.items():
    createPermutation(filename=key, url=value)