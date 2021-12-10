# This file is responsible for cropping images to just the face before perturbations
# and putting the face back on the original image after perturbations

from facelib import FaceDetector
from PIL import Image
import numpy as np

#TODO: Be seriously diligent about the rounding and math to ensure no off-by-one errors

# Crops a series of images to a face using Facelib and scales them to 160x160
# images -- Numpy 4D array in format (n, h, w, c)
def crop_faces(images, debug = True):
    if debug:
        print("=== Cropping faces out of input images for perturbing ===")

    # Facelib to detect faces TODO: we can consider using MTCNN instead which is what facenet recommends using
    if debug:
        print("Detecting faces...")
    detector = FaceDetector()
    boxes = [detector.detect_faces(img)[0] for img in images]

    # Crop each image to its bounding box and round the bounding boxes to an int
    if debug:
        print("Cropping images...")
    cropped_images = []
    new_boxes = []
    for img, box in zip(images, boxes):
        cropped_image, new_box = crop_image(img, box)
        cropped_images.append(cropped_image)
        new_boxes.append(new_box)
    
    # apply a resize to each image to ensure they are all the same size
    if debug:
        print("resizing faces...")
    cropped_resized_images = [resize(img) for img in cropped_images]

    if debug:
        print("done")
    return cropped_resized_images, new_boxes

# Crop an image to the given bouding box and return the rounded bounding box for use later
# img - Numpy 3D array in format (h, w, c)
# box - list of bounding boxes from facelib (x,y,w,h)
# if no boxes are found, use the dimensions of the image
# otherwise, use the first bouding box in the list # TODO: what if multiple faces (Out of scope)
# Return the cropped image (h,w,c) and the bounding box [x,y,w,h]
def crop_image(img, box): # TODO: What if face isnt detected?
    if len(box > 0):
        x,y,w,h = box[0].int().tolist()
    else:
        x,y = 0, 0
        w,h,_ = img.shape
    return img[y:h, x:w], [x,y,w,h]

# Resize the cropped image to be the size needed by the Network
# img - Numpy 3D array in format (h, w, c)
# Return the image as a PIL Image (we need it to be PIL to do the image resizing and pasting)
def resize(img, final_size=160): #TODO: We know it's 160 because the networks, we should have a way to change this in case we used another network
    img = Image.fromarray(img)

    #Find new scale
    size = img.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])

    #Resize image
    img = img.resize(new_image_size, Image.ANTIALIAS)

    #Paste image onto an empty canvas (creating black boxes, TODO: THIS MAY NOT BE A GOOD WAY TO HANDLE THIS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(img, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    
    return new_im

#Restores an image to its original size
# image - original frame - Numpy 3D array in format (h, w, c)
# cropped_image - cropped frame - Numpy 3D array in format (h, w, c)
# box - the box the cropped image was pulled from [x,y,w,h]
# Return the original image with the restored face pasted back on the image as a PIL Image
def restore_image(image, cropped_image, box):
    size = 160 # temp
    x,y,x2,y2 = box
    w = x2 - x
    h = y2 - y

    #crop image
    ratio = float(size) / float(max(w, h))
    crop_size = tuple([int(x*ratio) for x in [w, h]])
    crop_pos = tuple([(size - x)//2 for x in crop_size])
    
    img = cropped_image[crop_pos[1]:crop_pos[1]+crop_size[1], crop_pos[0]:crop_pos[0]+crop_size[0]]
    img = Image.fromarray(img.astype(np.uint8))

    #Resize cropped image
    img = img.resize((w, h), Image.ANTIALIAS)

    # Paste onto original image
    image = Image.fromarray(image.astype(np.uint8))
    image.paste(img, (x, y))
    return image

# Applys restore_image to a list of images, cropped images, and boxes
# images - list of orginal frames - Numpy 3D array in format (h, w, c)
# cropped_images - list of cropped frames - Numpy 3D array in format (h, w, c)
# boxes - list of the boxes the cropped image was pulled from [x,y,w,h]
# Return the original images with the restored face pasted back on the image as a PIL Image
def restore_images(images, cropped_images, boxes):
    print("=== Image Restoration ===")
    print("%d images to restore" % len(boxes))

    return [restore_image(img, crop, box) for img, crop, box in zip(images, cropped_images, boxes)]