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
image_copy1 = sample_img[:,:,::-1].copy()
# height, width, c
h, w, c = sample_img.shape

print('Height: ', h)
print('Width: ', w)
print("C:", c)

def distFromCenter():
    boxDist = face_data.relative_bounding_box
    # finds center of the frame
    centerWidth= w*(0.5/1.0)
    centerHeight = h*(0.5/1.0)
    # (Xmin,Ymin) = top left
    # (Xmax, Ymin) = Top right
    # (Xmin, Ymax) = Bottom left
    #(Xmax, Ymax) = Bottom right
    # Xmax = Xmin + width
    # Ymax = Ymin + height 
    # To draw a center bounding box where the face should be at we should take the height/2 and width/2
    # and set up the xmin,ymin, xmax and ymax.
    print(f'{centerWidth, centerHeight}')
    newXmin = int((0.5 - (boxDist.width/2))*w)
    newYmin = int((0.5 - (boxDist.height/2))* w)
    newXmax = int((0.5 + (boxDist.width/2))* h)
    newYmax = int((0.5 + (boxDist.height/2))* h)
    print(f'new Xmin, Ymin, Xmax, Ymax{newXmin, newYmin, newXmax, newYmax }')
    x= int(boxDist.xmin)
    y=int(boxDist.ymin)
    cv2.rectangle(image_copy, (newXmin, newYmin), (newXmax, newYmax), (255,0,0), 2)
    print(f'FACE in Dst: \n{boxDist.xmin, boxDist.ymin, boxDist.width, boxDist.height}')

if face_detection_results.detections:
    # generate iterable with enum over faces in the image
    for face_no, face in enumerate(face_detection_results.detections):
        print(f'FACE NUMBER: {face_no+1}')
        print("===========================")

        print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')
        face_data = face.location_data
        # Values in relative bounding values are normalized to between 0.0 to 1.0
        print(f'FACE BOUNDING BOX: \n{face_data.relative_bounding_box}')
        distFromCenter()
        # values oof range between 1 and 6
        for i in range(1):
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


