import cv2
import dlib
import numpy as np
import imutils

detector = dlib.get_frontal_face_detector()

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

# cap = cv.VideoCapture(0)
org_img = cv2.imread('person.jpg')

if org_img is None:
    print('File Not Available')
    exit(0)

org_img = imutils.resize(org_img, width=500)
gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)

faces = detector(gray, 1)
print("Number of Faces Detected: ", len(faces))

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

for face in faces:
    landmarks = predictor(gray, face)

    left_eye_points = []
    for i in range(36, 42):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        left_eye_points.append(point)

    left_eye_mask = np.array(left_eye_points, dtype=np.int32)
    # img_line = cv2.polylines(org_img, [left_eye_mask], True, COLOR_RED, thickness=2, lineType=cv2.LINE_8)
    # img_masked = cv2.fillPoly(img_line, [left_eye_mask], COLOR_RED, lineType=cv2.LINE_AA)
    img_line = cv2.polylines(org_img, [left_eye_mask], True, 0, thickness=2, lineType=cv2.LINE_8)
    img_masked = cv2.fillPoly(img_line, [left_eye_mask], 0, lineType=cv2.LINE_AA)

    right_eye_points = []
    for i in range(42, 48):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        right_eye_points.append(point)

    right_eye_mask = np.array(right_eye_points, dtype=np.int32)
    # img_line = cv2.polylines(org_img, [right_eye_mask], True, COLOR_GREEN, thickness=2, lineType=cv2.LINE_8)
    # img_masked = cv2.fillPoly(img_line, [right_eye_mask], COLOR_GREEN, lineType=cv2.LINE_AA)
    img_line = cv2.polylines(org_img, [right_eye_mask], True, 0, thickness=2, lineType=cv2.LINE_8)
    img_masked = cv2.fillPoly(img_line, [right_eye_mask], 0, lineType=cv2.LINE_AA)

    mouth_points = []
    for i in range(48, 60):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        mouth_points.append(point)

    mouth_mask = np.array(mouth_points, dtype=np.int32)
    # img_line = cv2.polylines(org_img, [mouth_mask], True, COLOR_BLUE, thickness=2, lineType=cv2.LINE_8)
    # img_masked = cv2.fillPoly(img_line, [mouth_mask], COLOR_BLUE, lineType=cv2.LINE_AA)
    img_line = cv2.polylines(org_img, [mouth_mask], True, 0, thickness=2, lineType=cv2.LINE_8)
    img_masked = cv2.fillPoly(img_line, [mouth_mask], 0, lineType=cv2.LINE_AA)

save_image_file = "masked_image.jpg"
print("Saving output image to", save_image_file)
cv2.imwrite(save_image_file, img_masked)
