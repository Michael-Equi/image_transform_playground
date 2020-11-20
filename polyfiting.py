import numpy as np
import cv2

img = np.zeros((1000, 1000, 3))

drawing = False
pts = []


def draw_poly(p, start, stop):
    xs = np.linspace(start, stop, 1000).astype(np.int)
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
        current_points = []
        for point in points:
            if len(current_points) < 3:
                current_points.append(point)
            else:
                current_points.append(point)
                t = np.linspace(0, len(current_points) - 1, len(current_points)).astype(np.int)
                cpts = np.array(current_points)
                p = np.polyfit(t, cpts, 2)
                if np.sum((np.polyval(p[:, 0], t) - cpts[:, 0])**2) + \
                        np.sum((np.polyval(p[:, 1], t) - cpts[:, 1])**2) > 1:
                    draw_poly(p, 0, len(current_points))
                    current_points = [current_points[-1]]
        # error was not low enough to already trigger a draw
        if len(current_points) > 3:
            t = np.linspace(0, len(current_points) - 1, len(current_points)).astype(np.int)
            p = np.polyfit(t, np.array(current_points), 2)
            draw_poly(p, 0, len(current_points))
        pts = []
    if drawing == True:
        img[y, x, 2] = 1
        pts.append([x, y])
        cv2.imshow('img', img)


cv2.imshow('img', img)
cv2.setMouseCallback('img', click)
cv2.waitKey(0)
