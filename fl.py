from facelib import FaceDetector
import PIL
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np


output_filename = "output/example_cropped.jpg"

def crop_faces(images):
    print(images.shape)
    detector = FaceDetector()
    boxes = [detector.detect_faces(img)[0] for img in images]
       
    #print(boxes, scores, landmarks)

    #x,y,w,h = [int(a) for a in boxes[0].tolist()]
    #print(x,y,w,h)
    #cv2.imwrite(output_filename, img[y:h, x:w])

    #cropped_images = [crop(img, box) for img, box in zip(images, boxes)]
    cropped_images = []
    new_boxes = []
    for img, box in zip(images, boxes):
        cropped_image, new_box = crop(img, box)
        cropped_images.append(cropped_image)
        new_boxes.append(new_box)
    
    # im = Image.fromarray(cropped_images[100])
    # im.save(output_filename)
    cropped_resized_images = [resize(img) for img in cropped_images]
    # im2 = cropped_resized_images[100]
    # im2.save("output/example_cropped_resized.jpg")
    
    # for cropped_resized_image in cropped_resized_images:
    #     print(cropped_resized_image.size)
    return cropped_resized_images, new_boxes

def resize(img, final_size=160):
    img = Image.fromarray(img)
    size = img.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    img = img.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(img, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

def crop(img, box): # TODO: What if face isnt detected?
    x,y,w,h = box[0].int().tolist()
    return img[y:h, x:w], [x,y,w,h]

def restore_image(image, cropped_image, box):
    # print(image.shape)
    # print(cropped_image.shape)
    # print(box)

    size = 160 # temp
    x,y,x2,y2 = box
    w = x2 - x
    h = y2 - y
    # print("Width: ", w)
    # print("Height:", h)

    #crop image
    ratio = float(size) / float(max(w, h))
    crop_size = tuple([int(x*ratio) for x in [w, h]])
    #print(crop_size)

    crop_pos = tuple([(size - x)//2 for x in crop_size])
    #print(crop_pos)
    
    img = cropped_image[crop_pos[1]:crop_pos[1]+crop_size[1], crop_pos[0]:crop_pos[0]+crop_size[0]]
    img = Image.fromarray(img.astype(np.uint8))
    #img.save("example_restored_cropped.jpg")

    #Resize cropped image
    new_image_size = (w, h)
    img = img.resize(new_image_size, Image.ANTIALIAS)
    #new_im = Image.new("RGB", (w, h))
    #new_im.paste(img, None)
    #new_im.save("example_restored_cropped_resized.jpg")

    # Paste onto original image
    image = Image.fromarray(image.astype(np.uint8))
    image.paste(img, (x, y))
    #image.save("example_restored.jpg")
    return image

def restore_images(images, cropped_images, boxes):
    print("===Image Restoration===")
    print(type(images))
    print(images.shape)
    print(type(cropped_images))
    print(cropped_images.shape)
    print(type(boxes))
    print(len(boxes))
    restored_images = [restore_image(i, c, b) for i, c, b in zip(images, cropped_images, boxes)]
    restored_images[100].save("example_restored.jpg")
    
    return restored_images