import sys
import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

animal_img = []
animal_roi = []

COLOR_WHITE = (255, 255, 255)
bear_img = cv2.imread('animal_filter/hole/bear.png')
bear_img = cv2.cvtColor(bear_img, cv2.COLOR_BGR2BGRA)
bear_roi = [
    [118, 181, 50, 20], [79, 105, 28, 28], [144, 95, 28, 28]
]
animal_img.append(bear_img)
animal_roi.append(bear_roi)

cat_img = cv2.imread('animal_filter/hole/cat.png')
cat_img = cv2.cvtColor(cat_img, cv2.COLOR_BGR2BGRA)
cat_roi = [
    [114, 188, 54, 19], [83, 113, 35, 35], [157, 109, 35, 35]
]
animal_img.append(cat_img)
animal_roi.append(cat_roi)

dino_img = cv2.imread('animal_filter/hole/dino.png')
dino_img = cv2.cvtColor(dino_img, cv2.COLOR_BGR2BGRA)
dino_roi = [
    [27, 131, 128, 84], [26, 52, 28, 28], [125, 48, 28, 28]
]
animal_img.append(dino_img)
animal_roi.append(dino_roi)

dog_img = cv2.imread('animal_filter/hole/dog.png')
dog_img = cv2.cvtColor(dog_img, cv2.COLOR_BGR2BGRA)
dog_roi = [
    [24, 197, 111, 40], [28, 92, 28, 28], [126, 106, 28, 28]
]
animal_img.append(dog_img)
animal_roi.append(dog_roi)

fox_img = cv2.imread('animal_filter/hole/fox.png')
fox_img = cv2.cvtColor(fox_img, cv2.COLOR_BGR2BGRA)
fox_roi = [
    [66, 180, 40, 15], [51, 110, 25, 25], [111, 111, 25, 25]
]
animal_img.append(fox_img)
animal_roi.append(fox_roi)

horse_img = cv2.imread('animal_filter/hole/horse.png')
horse_img = cv2.cvtColor(horse_img, cv2.COLOR_BGR2BGRA)
horse_roi = [
    [13, 226, 37, 14], [22, 100, 25, 25], [86, 98, 25, 25]
]
animal_img.append(horse_img)
animal_roi.append(horse_roi)

quokka_img = cv2.imread('animal_filter/hole/quokka.png')
quokka_img = cv2.cvtColor(quokka_img, cv2.COLOR_BGR2BGRA)
quokka_roi = [
    [80, 131, 61, 41], [45, 68, 20, 20], [118, 53, 20, 20]
]
animal_img.append(quokka_img)
animal_roi.append(quokka_roi)

rabbit_img = cv2.imread('animal_filter/hole/rabbit.png')
rabbit_img = cv2.cvtColor(rabbit_img, cv2.COLOR_BGR2BGRA)
rabbit_roi = [
    [99, 283, 55, 21], [61, 185, 30, 30], [145, 188, 30, 30]
]
animal_img.append(rabbit_img)
animal_roi.append(rabbit_roi)

sea_elephant_img = cv2.imread('animal_filter/hole/sea_elephant.png')
sea_elephant_img = cv2.cvtColor(sea_elephant_img, cv2.COLOR_BGR2BGRA)
sea_elephant_roi = [
    [107, 154, 75, 26], [24, 42, 28, 28], [144, 46, 28, 28]
]
animal_img.append(sea_elephant_img)
animal_roi.append(sea_elephant_roi)

toad_img = cv2.imread('animal_filter/hole/toad.png')
toad_img = cv2.cvtColor(toad_img, cv2.COLOR_BGR2BGRA)
toad_roi = [
    [93, 61, 110, 28], [95, 28, 25, 25], [167, 19, 25, 25]
]
animal_img.append(toad_img)
animal_roi.append(toad_roi)

img_size_width = 300
img_size_height = 300

''''''''''''''''''''' Realtime Webcam Code '''''''''''''''''''''''
cap = cv2.VideoCapture(0)
masked_img = np.zeros((img_size_width, img_size_height, 3), np.uint8)
animal_idx = 0

while True:
    ret, img = cap.read()
    if not ret:
        break

    input_keycode = cv2.waitKey(1)
    if ord('0') <= input_keycode <= ord('9'):
        animal_idx = input_keycode - ord('0')

    org_img = cv2.resize(img, dsize=(img_size_width, img_size_height), interpolation=cv2.INTER_LINEAR)
    draw_img = np.zeros((org_img.shape[0], org_img.shape[1], org_img.shape[2]), np.uint8)
    blank_img = np.zeros((animal_img[animal_idx].shape[0], animal_img[animal_idx].shape[1],
                          animal_img[animal_idx].shape[2]), np.uint8)
    gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    print("Number of Faces Detected: ", len(faces))

    roi_parts = []

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

        roi_list = []
        for contour_idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = masked_img[y:y + h, x:x + w]
            roi = cv2.resize(roi, dsize=(animal_roi[animal_idx][contour_idx][2], animal_roi[animal_idx][contour_idx][3]),
                             interpolation=cv2.INTER_LINEAR)

            roi_tmp = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(roi_tmp, 0, 255, cv2.THRESH_BINARY)
            b, g, r = cv2.split(roi)
            rgba = [b, g, r, alpha]
            roi = cv2.merge(rgba, 4)
            roi_list.append(roi)

        if len(contours) > 2:
            roi_left_eye = roi_list[1]
            roi_right_eye = roi_list[2]

            if cv2.boundingRect(contours[1])[0] > cv2.boundingRect(contours[2])[0]:
                roi_left_eye = roi_list[2]
                roi_right_eye = roi_list[1]

            roi_parts.append(roi_list[0])
            roi_parts.append(roi_left_eye)
            roi_parts.append(roi_right_eye)

        for contour_idx in range(len(contours)):
            x, y, w, h = animal_roi[animal_idx][contour_idx]
            blank_img[y:y + h, x:x + w] = roi_parts[contour_idx]

    compose_img = cv2.bitwise_or(animal_img[animal_idx], blank_img)
    # cv2.imshow('org_img', org_img)
    # cv2.imshow('contoured_img', draw_img)
    cv2.imshow('masked_img', masked_img)
    cv2.imshow('composed_img', compose_img)

    # if len(roi_parts) > 0 and len(roi_parts) % 3 == 0:
    #     for roi_idx in range(0, len(roi_parts), 3):
    #         cv2.imshow('roi_mouth', roi_parts[roi_idx])
    #         cv2.imshow('roi_left_eye', roi_parts[roi_idx + 1])
    #         cv2.imshow('roi_right_eye', roi_parts[roi_idx + 2])

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
# masked_img = np.zeros((img_size_width, img_size_height, 3), np.uint8)
# print("Number of Faces Detected: ", len(faces))
#
# roi_parts = []
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
#     masked_img = cv2.bitwise_and(org_img, draw_img)
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
#     draw_img_gray = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
#     contours, hierarchy = cv2.findContours(draw_img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     roi_list = []
#
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         roi = masked_img[y:y + h, x:x + w]
#         # cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         roi = cv2.resize(roi, dsize=(img_size_width // 1, img_size_height // 1), interpolation=cv2.INTER_LINEAR)
#         roi_list.append(roi)
#
#     if len(contours) > 2:
#         roi_left_eye = roi_list[1]
#         roi_right_eye = roi_list[2]
#
#         if cv2.boundingRect(contours[1])[0] > cv2.boundingRect(contours[2])[0]:
#             roi_left_eye = roi_list[2]
#             roi_right_eye = roi_list[1]
#
#         roi_parts.append(roi_list[0])
#         roi_parts.append(roi_left_eye)
#         roi_parts.append(roi_right_eye)
#
# if len(roi_parts) > 0 and len(roi_parts) % 3 == 0:
#     for roi_idx in range(0, len(roi_parts), 3):
#         save_image_file = 'roi_idx%d_mouth.jpg' % (roi_idx // 3)
#         print("Save Image:", save_image_file)
#         cv2.imwrite(save_image_file, roi_parts[roi_idx])
#
#         save_image_file = 'roi_idx%d_left_eye.jpg' % (roi_idx // 3)
#         print("Save Image:", save_image_file)
#         cv2.imwrite(save_image_file, roi_parts[roi_idx + 1])
#
#         save_image_file = 'roi_idx%d_right_eye.jpg' % (roi_idx // 3)
#         print("Save Image:", save_image_file)
#         cv2.imwrite(save_image_file, roi_parts[roi_idx + 2])
