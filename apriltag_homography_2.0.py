from pupil_apriltags import Detector
import numpy as np
import cv2
import itertools
from scipy.spatial.transform import Rotation

tag_size = 0.11081
fx = 3075.0045348434996
fy = 3082.755795750154
cx = 1906.61235874168
cy = 1075.4475738294798
resolution = (2160, 3840, 3)  # shape of the frame with which the calibration was determined

camera_params = [fx, fy, cx, cy]
P = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])


def detect_tags(img1, img2):
    at_detector = Detector(families='tagStandard41h12',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)
    result1 = at_detector.detect(cv2.cvtColor(
        img1, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
    result2 = at_detector.detect(cv2.cvtColor(
        img2, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
    if len(result1) == len(result2) == 4:
        return result1, result2
    raise AttributeError("Not enough tags detected!")


def match_points(set1, set2):
    points = []
    for det1 in set1:
        for det2 in set2:
            if det1.tag_id == det2.tag_id:
                points.append([det1, det2])
    return points

def transform():
    global camera_params, P
    img1, img2 = cv2.imread('./images/4ft measure/6.jpeg'), cv2.imread('./images/vert_1_2.jpeg')
    # scale the calibration parameters based on the image resolution
    camera_params = [fx * img1.shape[1] / resolution[1], fy * img1.shape[0] / resolution[0],
                     cx * img1.shape[1] / resolution[1], cy * img1.shape[0] / resolution[0]]
    P = np.array([[camera_params[0], 0, camera_params[2]],
                  [0, camera_params[1], camera_params[3]],
                  [0, 0, 1]])

    result1, result2 = detect_tags(img1, img2)
    match_points(result1, result2)

    centers = []
    poses = []
    for result in result1:
        centers.append(result.center)
        poses.append(result.pose_t)
    top_left = np.array([0, 0])
    top_right = np.array([img1.shape[1], 0])
    bottom_left = np.array([0, img1.shape[0]])
    bottom_right = np.array([img1.shape[1], img1.shape[0]])

    targets = np.array([top_left, top_right, bottom_left, bottom_right])
    perms = np.array(list(itertools.permutations(centers)))
    pose_perms = np.array(list(itertools.permutations(poses)))

    best_combo = np.argmin(np.sum(np.linalg.norm(perms - targets, axis=1), axis=1))
    center_sorted = perms[best_combo]
    pose_sorted = pose_perms[best_combo][:, :, 0]

    print(pose_sorted, '\n')

    print('center 0', center_sorted[0], 'pose sorted 0', np.dot(P, pose_sorted[0]) / np.dot(P, pose_sorted[0])[2])
    print('center 1', center_sorted[1], 'pose sorted 1', np.dot(P, pose_sorted[1]) / np.dot(P, pose_sorted[1])[2])
    print('center 2', center_sorted[2], 'pose sorted 2', np.dot(P, pose_sorted[2]) / np.dot(P, pose_sorted[2])[2])
    print('center 3', center_sorted[3], 'pose sorted 3', np.dot(P, pose_sorted[3]) / np.dot(P, pose_sorted[3])[2], '\n')

    normals = []
    for combo in itertools.combinations(pose_sorted, 3):
        v12 = combo[1] - combo[0]
        v13 = combo[2] - combo[0]
        n = np.cross(v12, v13)
        n /= np.linalg.norm(n)
        normals.append(n)

    averaged_normal = np.array([0.0, 0.0, 0.0])
    print()
    for normal in normals:
        averaged_normal += np.abs(normal)  # This could cause some strange behavior, there is definitely a better way to
        # handle opposite normals
    averaged_normal /= len(normals)
    print('averaged normal', averaged_normal)
    n = averaged_normal

    #
    # v12 = pose_sorted[1] - pose_sorted[0]
    # v13 = pose_sorted[2] - pose_sorted[0]
    #
    # n = np.cross(v12, v13)
    # n /= np.linalg.norm(n)
    print('normal', n, '\n')
    # n_p = pose_sorted[0] + 1/4*n
    # draw the normal vector
    # img1 = cv2.arrowedLine(img1, tuple(center_sorted[0].astype(np.int)), tuple((np.dot(P, n_p) / np.dot(P, n_p)[2])[:2].astype(np.int)), (255, 150, 150), 10)

    z = np.array([0, 0, 1])
    axis_of_rot = np.cross(z, n) / np.linalg.norm(np.cross(z, n))
    rotation_deg = np.arccos(np.dot(z, n))
    rotation_vector = axis_of_rot * rotation_deg
    R = Rotation.from_rotvec(rotation_vector).as_matrix()

    print(np.dot(R, z))
    print(n)
    print()

    p0 = np.dot(np.linalg.inv(R), pose_sorted[0])
    p1 = np.dot(np.linalg.inv(R), pose_sorted[1])
    p2 = np.dot(np.linalg.inv(R), pose_sorted[2])
    p3 = np.dot(np.linalg.inv(R), pose_sorted[3])

    print(p0, p1, p2, p3)
    print()

    p0 = p0[:2]
    p1 = p1[:2]
    p2 = p2[:2]
    p3 = p3[:2]

    scale = 200
    top_left_pixels = np.array([0, 0])
    top_right_pixels = (p1 - p0) * scale
    bottom_left_pixels = (p2 - p0) * scale
    bottom_right_pixels = (p3 - p0) * scale

    arr = np.array([top_left_pixels, top_right_pixels, bottom_left_pixels, bottom_right_pixels])
    print(arr)
    arr += np.abs(np.min(arr, axis=0))
    print('\n', arr)
    target_points = arr

    h = cv2.getPerspectiveTransform(center_sorted.astype(np.float32), target_points.astype(np.float32))
    cv2.line(img1, tuple(center_sorted[0].astype(np.int)), tuple(center_sorted[1].astype(np.int)), (255, 0, 0), 5)
    cv2.line(img1, tuple(center_sorted[1].astype(np.int)), tuple(center_sorted[3].astype(np.int)), (255, 0, 0), 5)
    cv2.line(img1, tuple(center_sorted[3].astype(np.int)), tuple(center_sorted[2].astype(np.int)), (255, 0, 0), 5)
    out = cv2.line(img1, tuple(center_sorted[2].astype(np.int)), tuple(center_sorted[0].astype(np.int)), (255, 0, 0), 5)

    out = cv2.warpPerspective(out, h, tuple(np.max(arr, axis=0).astype(np.int)))
    cv2.imshow('out', out)
    print(out.shape)

    img1 = cv2.circle(img1, tuple(center_sorted[0].astype(np.int)), 5, (0, 255, 0), 5)
    img1 = cv2.circle(img1, tuple(center_sorted[1].astype(np.int)), 5, (0, 0, 0), 10)
    img1 = cv2.circle(img1, tuple(center_sorted[2].astype(np.int)), 5, (0, 0, 255), 10)
    img1 = cv2.circle(img1, tuple(center_sorted[3].astype(np.int)), 5, (255, 255, 0), 10)
    cv2.imshow('img', img1)

    point_1 = None
    def click(event, x, y, flags, param):
        nonlocal point_1, out
        if event == cv2.EVENT_LBUTTONDOWN:
            if point_1 is None:
                point_1 = (x, y)
            else:
                image = cv2.line(out, point_1, (x, y), (0, 0, 255), 5)
                print("distance between points is",
                      str(np.sqrt((point_1[0] - x) ** 2 + (point_1[1] - y) ** 2)) + "pixels")
                cv2.imshow('out', image)
    cv2.setMouseCallback('out', click)

    cv2.waitKey(0)


if __name__ == "__main__":
    transform()
