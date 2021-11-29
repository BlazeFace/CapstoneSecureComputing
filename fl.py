from facelib import FaceDetector
import PIL
from PIL import Image

img = PIL.Image.open("example.jpg")
detector = FaceDetector()
boxes, scores, landmarks = detector.detect_faces(image)