
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2 as cv

class FaceEyesDetector():
    def __init__(self):
        self.height = 1080
        self.width = 1920
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.indexes = {
            'face': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'left_eye' : [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'right_eye': [33, 7, 163, 144, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            }

    def detect(self, frame, object_to_detect = 'face'):
        try:
            results = self.face_mesh.process(frame)
            face_landmarks = results.multi_face_landmarks[0]
            face_points = np.array([[face_landmarks.landmark[x].x, face_landmarks.landmark[x].y] for x in self.indexes[object_to_detect]])
            x1 = int(face_points[:,0].min()*self.width)
            x2 = int(face_points[:,0].max()*self.width)
            y1 = int(face_points[:,1].min()*self.height)
            y2 = int(face_points[:,1].max()*self.height)
            if object_to_detect == 'face':
                return [x1,y1],[x2,y2]
            else:
                return [x1-20,y1-30],[x2+20,y2+30]
        except:
            return 0
        
    
    def return_img(self, frame, object_to_detect = 'face'):
        bb = self.detect(frame, object_to_detect)
        return frame[int(bb[0][1]):int(bb[1][1]),int(bb[0][0]):int(bb[1][0]),:]

