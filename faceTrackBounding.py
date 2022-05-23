import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

#initialize mediapipe's face detection model
mp_face_detection = mp.solutions.face_detection

# call detection function
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

#to  visualize
mp_drawing =mp.solutions.drawing_utils

sample_img = cv2.imread("girl.jpg")

face_detection_results = face_detection.process(sample_img[:,:,::-1])

image_copy = sample_img[:,:,::-1].copy()
# height, width, c
h, w, c = sample_img.shape

print('Height: ', h)
print('Width: ', w)
print("C:", c)
if face_detection_results.detections:
    # generate iterable with enum over faces in the image
    for face_no, face in enumerate(face_detection_results.detections):
        print(f'FACE NUMBER: {face_no+1}')
        print("===========================")

        print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')
        face_data = face.location_data
        # Values in relative bounding values are normalized to between 0.0 to 1.0
        print(f'FACE BOUNDING BOX: \n{face_data.relative_bounding_box}')
        
        # values oof range between 1 and 6
        for i in range(6):
            print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
            print(f'{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}')

        # to visualize
        mp_drawing.draw_detection(
            image=image_copy,
            detection=face,
            keypoint_drawing_spec = mp_drawing.DrawingSpec(
                color=(255, 0, 0),
                thickness = 2,
                circle_radius=2
            )
        )
plt.figure(figsize = [10, 10])

# plt.title("Girl");
# plt.axis("off");
# plt.imshow(sample_img[:,:,::-1])

#new Image
plt.title("Resultant Girl");
plt.axis('on');
plt.imshow(image_copy)

plt.show()