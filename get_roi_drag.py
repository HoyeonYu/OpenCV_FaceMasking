import cv2

isDragging = False

def onMouse(event, x, y, flags, param):
    global isDragging, x0, y0

    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = img.copy()
            cv2.rectangle(img_draw, (x0, y0), (x, y), (255, 0, 0), 2)
            cv2.imshow('img', img_draw)

    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w = x - x0
            h = y - y0
            print("x:%d, y:%d, w:%d, h:%d" % (x0, y0, w, h))


img = cv2.imread('animal_filter/hole/cat.png')
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)
cv2.waitKey()
cv2.destroyAllWindows()
