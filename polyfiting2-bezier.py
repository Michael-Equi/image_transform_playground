import numpy as np
import cv2

img = np.zeros((200, 200, 3))

drawing = False
pts = []


def draw_poly(p, start, stop):
    xs = np.linspace(start, stop, 100)
    poly_x = np.polyval(p[:, 0], xs)
    poly_y = np.polyval(p[:, 1], xs)
    poly_pts = np.concatenate((poly_x.reshape(-1, 1), poly_y.reshape(-1, 1)), axis=1).astype(np.int)
    cv2.polylines(img, [poly_pts], False, (1, 1, 1), 1)
    cv2.imshow('img', img)


def click(event, x, y, flags, param):
    global drawing, pts
    if (event == cv2.EVENT_LBUTTONDOWN):
        drawing = True
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points = np.array(pts)
        for i in range(0, len(points) - 2, 2):
            p0 = points[i]
            p1 = points[i+1]
            p2 = points[i+2]
            p = np.array([[(p2[0] - 2*p1[0] + p0[0]), (p2[1] - 2*p1[1] + p0[1])],
                          [(2 * p1[0] - 2 * p0[0]), (2 * p1[1] - 2 * p0[1])],
                          [p0[0], p0[1]]])
            draw_poly(p, 0, 1)
        pts = []
    if drawing == True:
        img[y, x, 2] = 1
        pts.append([x, y])
        cv2.imshow('img', img)


cv2.imshow('img', img)
cv2.setMouseCallback('img', click)
cv2.waitKey(0)
