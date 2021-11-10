from facelib import FaceDetector
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image

detector = FaceDetector()
image = cv2.imread('/home/yakorde/capstone/CapstoneSecureComputing/example.jpg')
#/home/yakorde/capstone/FaceLib
image1 = cv2.imread('/home/yakorde/capstone/CapstoneSecureComputing/output/output_cleverhans_facenet_casiawebface.jpeg')
image2 = cv2.imread('/home/yakorde/capstone/CapstoneSecureComputing/output/output_cleverhans_facenet_vggface2.jpeg')
image3 = cv2.imread('/home/yakorde/capstone/CapstoneSecureComputing/output/output_cleverhans_mobilenet.jpeg')
image4 = cv2.imread('/home/yakorde/capstone/CapstoneSecureComputing/output/output_torchattacks_facenet_casiawebface.jpeg')
image5 = cv2.imread('/home/yakorde/capstone/CapstoneSecureComputing/output/output_torchattacks_facenet_vggface2.jpeg')


arr = [image1, image2, image3, image4, image5, image]

for x in arr:
    boxes, scores, landmarks = detector.detect_faces(x)

    faces, boxes, scores, landmarks = detector.detect_align(x)
    
    print(boxes)
    print(scores)

#boxes, scores, landmarks = detector.detect_faces(image)
#faces, boxes, scores, landmarks = detector.detect_align(image)




#im2 = Image.fromarray(faces.cpu()[0])
#im2.save('output_crop.jpg')

#print(boxes)