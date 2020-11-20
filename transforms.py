import cv2
import numpy as np
from pupil_apriltags import Detector

img = cv2.imread('images/track.jpeg')
img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

at_detector = Detector(families='tagStandard41h12',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
results = at_detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

result = results[1]
points = np.array(result.corners)
pts1 = np.float32([points[3], points[2], points[1], points[0]])  # [top left, top right, bottom left, bottom right]
pts2 = np.float32([[0, 50], [0, 0], [50, 0], [50, 50]])  # [top left, top right, bottom left, bottom right]
# for i, pt in enumerate(pts1):
    # img = cv2.putText(img, str(i), tuple(pt.astype(np.int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
cv2.imshow('img1', img)
h = cv2.getPerspectiveTransform(pts1, pts2)

img3 = cv2.warpPerspective(img, np.dot(np.array([[1, 0, 25], [0, 1, 25], [0, 0, 1]]), h), (2000, 2000), cv2.INTER_CUBIC)
cv2.imshow('img3', img3)


"""
Determine the range of the transformation on a restricted domain
"""
min_x, min_y, max_x, max_y = 0, 0, 0, 0
def consider_point(x):
    global min_x, min_y, max_x, max_y
    b = np.dot(h, x)
    b /= b[2]
    if b[0] > max_x:
        max_x = b[0]
    elif b[0] < min_x:
        min_x = b[0]
    if b[1] > max_y:
        max_y = b[1]
    elif b[1] < min_y:
        min_y = b[1]


# handle left edge
for i in range(img.shape[0]):
    x = np.float32([0, i, 1])
    consider_point(x)

# handle right edge
for i in range(img.shape[0]):
    x = np.float32([img.shape[1], i, 1])
    consider_point(x)

# handle top edge
for i in range(img.shape[1]):
    x = np.float32([i, 0, 1])
    consider_point(x)

# handle bottom edge
for i in range(img.shape[1]):
    x = np.float32([i, img.shape[0], 1])
    consider_point(x)

# restrict between -3000, -3000, 3000, 3000
min_x, min_y, max_x, max_y = max(min_x, -3000), max(min_y, -3000), min(max_x, 3000), min(max_y, 3000)
print(min_x, min_y, max_x, max_y)

# translating the image so that it appears entirely in the positive region of the matrix
h_t = np.float32([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]])
dst = cv2.warpPerspective(img, np.dot(h_t, h), (int(-min_x + max_x), int(-min_y + max_y)))

point_1 = None
scale = 110.81 / 50  # mm/pixel


def click(event, x, y, flags, param):
    global point_1, dst
    if (event == cv2.EVENT_LBUTTONDOWN):
        if point_1 is None:
            point_1 = (x, y)
        else:
            image = cv2.line(dst, point_1, (x, y), (0, 0, 255), 5)
            print("distance between points is",
                  str(np.sqrt((point_1[0] - x) ** 2 + (point_1[1] - y) ** 2) * scale) + "mm")
            # dst = cv2.circle(dst, (x, y), 10, (0, 0, 255), 20)
            cv2.imshow('img', image)


cv2.imshow('img', dst)
cv2.setMouseCallback('img', click)
cv2.waitKey(0)
