from facelib import FaceDetector
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image


def getScores(filenames):
    detector = FaceDetector()
    images = []

    for filename in filenames:
        images.append(cv2.imread(filename))

    # image = cv2.imread('oops.jpg')
    # #/home/yakorde/capstone/FaceLib
    # image1 = cv2.imread('output/output_cleverhans_facenet_casiawebface.jpeg')
    # image2 = cv2.imread('output/output_cleverhans_facenet_vggface2.jpeg')
    # image3 = cv2.imread('output/output_cleverhans_mobilenet.jpeg')
    # image4 = cv2.imread('output/output_torchattacks_facenet_casiawebface.jpeg')
    # image5 = cv2.imread('output/output_torchattacks_facenet_vggface2.jpeg')

    total_scores = []
    for image, filename in zip(images, filenames):
        boxes, scores, landmarks = detector.detect_faces(image)

        faces, boxes, scores, landmarks = detector.detect_align(image)
        
        total_scores.append(scores)
        
        print(faces.shape)
        im2 = Image.fromarray(faces.detach().cpu().numpy()[0])
        im2.save(filename.split('.')[0] + "boundingbox.jpg")
    
    return total_scores

#boxes, scores, landmarks = detector.detect_faces(image)
#faces, boxes, scores, landmarks = detector.detect_align(image)




#im2 = Image.fromarray(faces.cpu()[0])
#im2.save('output_crop.jpg')

#print(boxes)