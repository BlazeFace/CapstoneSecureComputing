import random
import numpy as np
import PIL
from PIL import Image
from numpy.random import default_rng
from matplotlib import pyplot as plt

P = 10
alpha = 1


# FaceLib outputs scores so we can use the same L function as Yolo
def loss(z, m):
    loss_sum = 0
    for j in range(0, m):
        loss_sum += np.log(z)
    return loss_sum


def mask(mask1):
    with np.nditer(mask1, op_flags=['readwrite']) as it:
        for m_x in it:
            if random.randint(1, 100) <= 3:
                m_x[...] = 1
    return mask1

img = PIL.Image.open("example.jpg")
x = np.array(img.getdata())
SHAPE = x.shape
random.seed(238)
phi = 1
n = random.randint(500, 1000000)

m = mask(np.zeros(SHAPE))
rng = default_rng()
delta = np.multiply(rng.integers(low=0, high=256, size=SHAPE), m)
for i in range(1, P + 1):
    #delta = np.add(delta, alpha * np.gradient(loss(delta, 1)))
    delta = np.multiply(delta, m)
    delta = np.maximum(np.minimum(delta, np.subtract(np.zeros(SHAPE), x)),
                       np.subtract(np.ones(SHAPE), x))

plt.imshow(delta, interpolation='nearest')
plt.show()

array = np.multiply(delta, x)
array = np.absolute(array)
array = array % 256
im = Image.fromarray(array)
im = im.convert('RGB')
im.save("test.jpeg")
