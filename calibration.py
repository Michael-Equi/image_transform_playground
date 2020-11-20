import numpy as np
import cv2

# https://medium.com/analytics-vidhya/camera-calibration-with-opencv-f324679c6eb7
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

path = './video/calibration2.MOV'
cap = cv2.VideoCapture(path)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

img_shape = None
square_width = 0.024  # meters
chessboard_size = (8, 6)
world_spaced_corners = []


def createKnownBoardPosition():
    for i in range(chessboard_size[1]):
        for j in range(chessboard_size[0]):
            world_spaced_corners.append([j * square_width, i * square_width, 0])


createKnownBoardPosition()

skip_frames = 10
count = 0
printed_size = False
while cap.isOpened():
    ret, frame = cap.read()
    if not printed_size:
        print(frame.shape)
        printed_size = True
    if not ret:
        break
    if count >= skip_frames:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_shape = gray.shape

            # Find the chess board corners
            found, corners = cv2.findChessboardCorners(
                gray, chessboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If found, add object points, image points (after refining them)
            if found:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                gray = cv2.drawChessboardCorners(gray, chessboard_size, corners2, ret)

            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)
        count = 0
    else:
        count += 1

cap.release()
cv2.destroyAllWindows()

imgpoints = np.array(imgpoints)
objpoints = np.resize(np.array(world_spaced_corners, np.float32), (imgpoints.shape[0], 48, 3))

print("Computing  Calibration")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape[::-1], None, None)

print('\n###')
print('mtx:', mtx)
print('fx:', mtx[0, 0])
print('fy:', mtx[1, 1])
print('cx:', mtx[0, 2])
print('cy:', mtx[1, 2])
print('###\n')

print('dist:', dist, '\n')
print('rvecs:', rvecs, '\n')
print('tvecs:', tvecs, '\n')
