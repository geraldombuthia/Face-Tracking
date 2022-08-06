import cv2
import mediapipe as mp

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

dframe = cv2.imread("girl.jpg")

image_input = cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB)
# load face detection model
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,  # model selection
    min_detection_confidence=0.5  # confidence threshold
)
results = mp_face.process(image_input)

image_rows, image_cols, _ = dframe.shape
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

detection = results.detections[0]

eye_left=mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
#
eye_right=mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
#Track  nose
nose_tip =mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP)



eye_r = _normalized_to_pixel_coordinates(eye_right.x,eye_right.y, image_cols,image_rows)
eye_l = _normalized_to_pixel_coordinates(eye_left.x,eye_left.y, image_cols,image_rows)
nose_t = _normalized_to_pixel_coordinates(nose_tip.x,nose_tip.y, image_cols,image_rows)
#
print(eye_r, eye_l, nose_t)
cv2.putText(image_input, '.', eye_r,cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(image_input, '.', eye_l,cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(image_input, '.', nose_t,cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


cv2.imwrite('this_output.png', image_input)