from facelib import FaceDetector
import PIL
from PIL import Image
import numpy as np

#TODO: document
def crop_faces(images):
    print("=== Cropping faces out of input images for perturbing ===")

    # Facelib to detect faces TODO: we can consider using MTCNN instead which is what facenet recommends using
    detector = FaceDetector()
    boxes = [detector.detect_faces(img)[0] for img in images]

    # Crop each image to its bounding box and round the bounding boxes to an int
    cropped_images = []
    new_boxes = []
    for img, box in zip(images, boxes):
        cropped_image, new_box = crop_image(img, box)
        cropped_images.append(cropped_image)
        new_boxes.append(new_box)
    
    # apply a resize to each image to ensure they are all the same size
    cropped_resized_images = [resize(img) for img in cropped_images]

    return cropped_resized_images, new_boxes

#TODO: document
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

#TODO: document
def crop_image(img, box): # TODO: What if face isnt detected?
    x,y,w,h = box[0].int().tolist()
    return img[y:h, x:w], [x,y,w,h]

#TODO: document
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

# TODO: document
def restore_images(images, cropped_images, boxes):
    print("=== Image Restoration ===")
    print("%d images to restore" % len(boxes))

    return [restore_image(img, crop, box) for img, crop, box in zip(images, cropped_images, boxes)]