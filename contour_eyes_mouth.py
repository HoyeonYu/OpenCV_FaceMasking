import os
import sys
import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

COLOR_WHITE = (255, 255, 255)

img_size_width = 300
img_size_height = 300

''''''''''''''''''''' Realtime Webcam Code '''''''''''''''''''''''
cap = cv2.VideoCapture(0)
masked_img = np.zeros((img_size_width, img_size_height, 3), np.uint8)

while True:
    ret, img = cap.read()
    if not ret:
        break

    org_img = cv2.resize(img, dsize=(img_size_width, img_size_height), interpolation=cv2.INTER_LINEAR)
    draw_img = np.zeros((org_img.shape[0], org_img.shape[1], org_img.shape[2]), np.uint8)
    gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    print("Number of Faces Detected: ", len(faces))

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_points = []
        for i in range(36, 42):
            point = [landmarks.part(i).x, landmarks.part(i).y]
            left_eye_points.append(point)

        left_eye_mask = np.array(left_eye_points, dtype=np.int32)
        img_line = cv2.polylines(draw_img, [left_eye_mask], True, COLOR_WHITE, thickness=2, lineType=cv2.LINE_8)
        draw_img = cv2.fillPoly(img_line, [left_eye_mask], COLOR_WHITE, lineType=cv2.LINE_AA)

        right_eye_points = []
        for i in range(42, 48):
            point = [landmarks.part(i).x, landmarks.part(i).y]
            right_eye_points.append(point)

        right_eye_mask = np.array(right_eye_points, dtype=np.int32)
        img_line = cv2.polylines(draw_img, [right_eye_mask], True, COLOR_WHITE, thickness=2, lineType=cv2.LINE_8)
        draw_img = cv2.fillPoly(img_line, [right_eye_mask], COLOR_WHITE, lineType=cv2.LINE_AA)

        mouth_points = []
        for i in range(48, 60):
            point = [landmarks.part(i).x, landmarks.part(i).y]
            mouth_points.append(point)

        mouth_mask = np.array(mouth_points, dtype=np.int32)
        img_line = cv2.polylines(draw_img, [mouth_mask], True, COLOR_WHITE, thickness=2, lineType=cv2.LINE_8)
        draw_img = cv2.fillPoly(img_line, [mouth_mask], COLOR_WHITE, lineType=cv2.LINE_AA)

        masked_img = cv2.bitwise_and(org_img, draw_img)

    draw_img_gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(draw_img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(draw_img, [contour], 0, (255, 0, 0), 3)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('org_img', org_img)
    cv2.imshow('contoured_img', draw_img)
    cv2.imshow('masked_img', masked_img)

    if cv2.waitKey(1) == ord('q'):
        sys.exit(1)


# ''''''''''''''''''''' Image Code '''''''''''''''''''''''
# # org_img = cv2.imread('person.jpg')
# org_img = cv2.imread('people.jpg')
#
# if org_img is None:
#     print('File Not Available')
#     exit(0)
#
# org_img = cv2.resize(org_img, dsize=(img_size_width, img_size_height), interpolation=cv2.INTER_LINEAR)
# draw_img = np.zeros((org_img.shape[0], org_img.shape[1], org_img.shape[2]), np.uint8)
# gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
# faces = detector(gray, 1)
# print("Number of Faces Detected: ", len(faces))
#
# for face in faces:
#     landmarks = predictor(gray, face)
#
#     left_eye_points = []
#     for i in range(36, 42):
#         point = [landmarks.part(i).x, landmarks.part(i).y]
#         left_eye_points.append(point)
#
#     left_eye_mask = np.array(left_eye_points, dtype=np.int32)
#     img_line = cv2.polylines(draw_img, [left_eye_mask], True, COLOR_WHITE, thickness=2, lineType=cv2.LINE_8)
#     draw_img = cv2.fillPoly(img_line, [left_eye_mask], COLOR_WHITE, lineType=cv2.LINE_AA)
#
#     right_eye_points = []
#     for i in range(42, 48):
#         point = [landmarks.part(i).x, landmarks.part(i).y]
#         right_eye_points.append(point)
#
#     right_eye_mask = np.array(right_eye_points, dtype=np.int32)
#     img_line = cv2.polylines(draw_img, [right_eye_mask], True, COLOR_WHITE, thickness=2, lineType=cv2.LINE_8)
#     draw_img = cv2.fillPoly(img_line, [right_eye_mask], COLOR_WHITE, lineType=cv2.LINE_AA)
#
#     mouth_points = []
#     for i in range(48, 60):
#         point = [landmarks.part(i).x, landmarks.part(i).y]
#         mouth_points.append(point)
#
#     mouth_mask = np.array(mouth_points, dtype=np.int32)
#     img_line = cv2.polylines(draw_img, [mouth_mask], True, COLOR_WHITE, thickness=2, lineType=cv2.LINE_8)
#     draw_img = cv2.fillPoly(img_line, [mouth_mask], COLOR_WHITE, lineType=cv2.LINE_AA)
#
# draw_img_gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
# contours, hierarchy = cv2.findContours(draw_img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
# for contour in contours:
#     cv2.drawContours(draw_img, [contour], 0, (255, 0, 0), 3)
#
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# contour_dir = 'contour_img'
# if not os.path.exists(contour_dir):
#     os.makedirs(contour_dir)
#
# save_image_file = "org_img.jpg"
# print("Save Image:", save_image_file)
# cv2.imwrite(contour_dir + '/' + save_image_file, org_img)
#
# save_image_file = "binary_masked.jpg"
# print("Save Image:", save_image_file)
# cv2.imwrite(contour_dir + '/' + save_image_file, draw_img_gray)
#
# save_image_file = "draw_contour.jpg"
# print("Save Image:", save_image_file)
# cv2.imwrite(contour_dir + '/' + save_image_file, draw_img)