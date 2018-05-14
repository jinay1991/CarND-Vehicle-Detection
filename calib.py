import glob
import os
import pickle

import numpy as np

import cv2


def calibrate(image_dir, nx=9, ny=6, save=True):
    """
    Performs camera calibration

    Returns:
        mtx (camera matrix)
        dist (camera dist matrix)
    """
    assert os.path.exists(image_dir), "%s does not exist" % (image_dir)

    fileList = glob.glob("%s/*.jpg" % (image_dir))
    assert len(fileList) > 0, "No calibration images found"

    pattern_points = np.zeros(shape=(nx * ny, 3), dtype=np.float32)
    pattern_points[:, :2] = np.indices(dimensions=(nx, ny)).T.reshape(-1, 2)

    objectPoints = []
    imagePoints = []
    for fileName in fileList:
        image = cv2.imread(fileName)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape[:2]
        found, corners = cv2.findChessboardCorners(gray, (nx, ny))
        if not found:
            continue

        imagePoints.append(corners.reshape(-1, 2))
        objectPoints.append(pattern_points)

    assert len(objectPoints) > 0, "chessboard not found"
    assert len(imagePoints) > 0, "chessboard not found"

    rms, mtx, dist, rvec, tvec = cv2.calibrateCamera(objectPoints=objectPoints,
                                                     imagePoints=imagePoints,
                                                     imageSize=(w, h),
                                                     cameraMatrix=None,
                                                     distCoeffs=None)

    if save:
        pickle.dump({'rms': rms, 'mtx': mtx, 'dist': dist, 'rvec': rvec, 'tvec': tvec}, open('calib.p', 'wb'))

    return mtx, dist


def load():
    """
    Reads pickle for camera calibration matrix

    Returns
        mtx (camera matrix)
        dist (camera dist matrix)
    """
    assert os.path.exists("calib.p"), "Failed to locate 'calib.p'. Please perform calibration first"
    data = pickle.load(open('calib.p', 'rb'))
    return data['mtx'], data['dist']
