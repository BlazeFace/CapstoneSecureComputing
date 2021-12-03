from facelib import FaceDetector
import PIL
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def crop_faces(images):
    print(images.shape)
    output_filename = "output/example_cropped.jpg"
    detector = FaceDetector()
    boxes = [detector.detect_faces(img)[0] for img in images]
       
    #print(boxes, scores, landmarks)

    #x,y,w,h = [int(a) for a in boxes[0].tolist()]
    #print(x,y,w,h)
    #cv2.imwrite(output_filename, img[y:h, x:w])
    cropped_images = [crop(img, box) for img, box in zip(images, boxes)]
    im = Image.fromarray(cropped_images[100])
    im.save(output_filename)

    return cropped_images

def crop(img, box): # TODO: What if face isnt detected?
    x,y,w,h = box[0].int().tolist()
    return img[y:h, x:w]

def main():
    input_filename = "input/example.jpg"
    #img = PIL.Image.open("input/example.jpg")
    img = cv2.imread(input_filename)
    crop_face(img)

if __name__ == "__main__":
    main()