import random
import numpy as np
import PIL
from PIL import Image
from numpy.random import default_rng
from matplotlib import pyplot as plt
import math

P = 10
alpha = .005


# FaceLib outputs scores so we can use the same L function as Yolo
def loss(z, m):
    loss_sum = 0
    for j in range(0, m):
        loss_sum += np.log(z)
    return loss_sum


def mask(mask1, org):
    with np.nditer(mask1, flags=['multi_index'], op_flags=['readwrite']) as it:
        for m_x in it:
            if random.randint(1, 100) <= 1:
                m_x[...] = 1
                org[it.multi_index] = 0
    return mask1, org

#Projected Gradient Descent
def pgd(image_arr):
    x = image_arr
    SHAPE = x.shape
    random.seed(238)
    phi = 1
    n = random.randint(500, 1000000)

    m, x = mask(mask1=np.zeros(SHAPE), org=x)
    rng = default_rng()
    delta = np.multiply(rng.integers(low=0, high=256, size=SHAPE), m)
    for i in range(1, P + 1):
        #delta = np.add(delta, alpha * np.gradient(loss(delta, 1)))
        delta = np.multiply(delta, m)
        delta = np.maximum(np.minimum(delta, np.subtract(np.zeros(SHAPE), x)),
                        np.subtract(np.ones(SHAPE), x))

    plt.imshow(delta, interpolation='nearest')
    plt.savefig("delta.jpeg")
    plt.figure(figsize=(20, 4))
    delta = np.absolute(delta)
    dm = np.multiply(delta, m)
    array = dm + x
    return array


img = PIL.Image.open("example.jpg")

impre = np.array(img)
array = pgd(impre)
print(array.shape)

#im = Image.fromarray(array, "RGB")
#im.convert('RGB')
#im.save('output.jpeg')
im2 = Image.fromarray(impre)
im2.save('out2.jpg')