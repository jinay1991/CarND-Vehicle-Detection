# -*- coding: utf-8 -*-

import os
import time

import numpy as np

import calib
import cv2
from classifier import features, load, train
from detect import detect_cars, showImages
from lane import detect_lanes
from line import Line

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train classifier", action="store_true")
    parser.add_argument("--sample_size", help="classifier sample size for training", default=None, type=int)
    parser.add_argument("--fname", help="input video/image", default="test_video.mp4")
    parser.add_argument("--color_space", help="can be RGB, HSV, LUV, HLS, YUV, YCrCb", default="YCrCb", type=str)
    parser.add_argument("--orient", help="HOG orientations", default=9, type=int)
    parser.add_argument("--pix_per_cell", help="HOG pixels per cell", default=16, type=int)
    parser.add_argument("--cell_per_block", help="HOG cells per block", default=2, type=int)
    parser.add_argument("--hog_channel", help="HOG channels, can be 0, 1, 2 or 'ALL'", default='ALL', type=str)
    parser.add_argument("--spatial_size", help="Spatial binning dimensions (w, h)", nargs=2, default=(64, 64), type=int)
    parser.add_argument("--hist_bins", help="Number of histogram bins", default=32, type=int)
    parser.add_argument("--spatial_feat", help="Spatial features on or off", action="store_false")
    parser.add_argument("--hist_feat", help="Histogram features on or off", action="store_false")
    parser.add_argument("--hog_feat", help="HOG features on or off", action="store_false")
    parser.add_argument("--heat_threshold", help="Heatmap threshold", default=1, type=int)
    parser.add_argument("--method", help="car search method. can be 'find_cars' or 'search_windows'", default='find_cars', type=str)
    parser.add_argument("--detect_lanes", help="detects lane lines", action="store_true")
    args = parser.parse_args()

    # Accomodate hog_channel multi-type
    args.hog_channel = int(args.hog_channel) if args.hog_channel != 'ALL' else args.hog_channel

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Arguments values
    # -------------------------------------------------------------------------------------------------------------------------------------
    print(args)

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Train or use pre-trained classifier
    # -------------------------------------------------------------------------------------------------------------------------------------
    np.random.seed(100)
    if not args.train:
        print("loading pre-trained classifier...")
        clf, scaler = load()
    else:
        print("train classifier...")
        X, y = features(args.color_space, tuple(args.spatial_size), args.hist_bins, args.orient, args.pix_per_cell, args.cell_per_block,
                        args.hog_channel, args.hog_feat, args.spatial_feat, args.hist_feat, sample_size=args.sample_size)
        clf, scaler = train(X, y, save=True)

    assert os.path.exists(args.fname), "Failed to locate %s" % (args.fname)

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Prepare for lane line detection
    # -------------------------------------------------------------------------------------------------------------------------------------
    if args.detect_lanes:
        mtx, dist = calib.load()
        left_lane = Line()
        right_lane = Line()

    # -------------------------------------------------------------------------------------------------------------------------------------
    # Process on image/video frames
    # -------------------------------------------------------------------------------------------------------------------------------------
    cap = cv2.VideoCapture(args.fname)
    assert cap.isOpened(), "Failed to open %s" % (args.fname)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    isImage = True if args.fname.endswith(".jpg") or args.fname.endswith(".png") else False

    for idx in range(totalFrames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start = time.time()

        if args.detect_lanes:
            lane_img = detect_lanes(frame, mtx, dist, left_lane, right_lane, diag=False, display=False)

        result = detect_cars(frame, clf, scaler, args.orient, args.pix_per_cell, args.cell_per_block, tuple(args.spatial_size),
                             args.hist_bins, args.color_space, args.hog_channel, args.hog_feat, args.spatial_feat, args.hist_feat,
                             args.heat_threshold, display=False, method=args.method, lanes_img=lane_img if args.detect_lanes else None)

        fps = 1.0 / (time.time() - start)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        cv2.putText(result, "fps: %02.2f" % fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 120), 2)
        cv2.imshow("result", result)
        key = cv2.waitKey(30) if not isImage else cv2.waitKey(0)
        if key == 32:
            cv2.waitKey(0)
        elif key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
