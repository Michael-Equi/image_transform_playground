import numpy as np
import cv2

img = np.zeros((200, 200, 3))

drawing = False
pts = []


def draw_poly(p, start, stop):
    xs = np.linspace(start, stop, 100)
    poly_x = np.polyval(p[0], xs)
    poly_y = np.polyval(p[1], xs)
    poly_pts = np.concatenate((poly_x.reshape(-1, 1), poly_y.reshape(-1, 1)), axis=1).astype(np.int)
    cv2.polylines(img, [poly_pts], False, (1, 1, 1), 1)
    cv2.imshow('img', img)


def filtered_pts(pts):
    last_pt = pts.pop(0)
    while len(pts) > 0:
        curr_pt = pts.pop(0)
        if np.linalg.norm(np.array(last_pt) - np.array(curr_pt)) > 3:
            last_pt = curr_pt
            yield curr_pt


def click(event, x, y, flags, param):
    global drawing, pts
    if (event == cv2.EVENT_LBUTTONDOWN):
        drawing = True
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points = np.array(list(filtered_pts(pts)))
        for i in range(0, len(points) - 3, 3):
            p0 = points[i]
            p1 = points[i + 1]
            p2 = points[i + 2]
            p3 = points[i + 3]

            M = np.array([[-1, 3, -3, 1],
                          [3, -6, 3, 0],
                          [-3, 3, 0, 0],
                          [1, 0, 0, 0]])

            p = np.dot(np.array([[p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]]]), M)
            draw_poly(p, 0, 1)
        pts = []
    if drawing == True:
        img[y, x, 2] = 1
        pts.append([x, y])
        cv2.imshow('img', img)


cv2.imshow('img', img)
cv2.setMouseCallback('img', click)
cv2.waitKey(0)
