# -*- coding: utf-8 -*-
import glob
import os
import pickle
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import cv2
from lesson_functions import (add_heat, apply_threshold, draw_boxes,
                              draw_labeled_bboxes, extract_features, find_cars,
                              search_windows, slide_window, single_img_features)


class TestSearchClassify(unittest.TestCase):

    def setUp(self):
        # TODO: Tweak these parameters and see how the results change.
        self.color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 0  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16)  # Spatial binning dimensions
        self.hist_bins = 16     # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        # y_start_stop = [None, None]  # Min and max in y to search in slide_window()

    def testSearchClassify(self):
        # Read in cars and notcars
        cars = glob.glob("C:\\workspace\\datasets\\vehicle_detection\\vehicles\\*\\*.png")
        notcars = glob.glob("C:\\workspace\\datasets\\vehicle_detection\\non-vehicles\\*\\*.png")
        # Reduce the sample size because
        # The quiz evaluator times out after 13s of CPU time
        sample_size = 500
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]
        print("car images: %s, notcar images: %s" % (len(cars), len(notcars)))

        car_features = extract_features(cars, color_space=self.color_space,
                                        spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                        orient=self.orient, pix_per_cell=self.pix_per_cell,
                                        cell_per_block=self.cell_per_block,
                                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        notcar_features = extract_features(notcars, color_space=self.color_space,
                                           spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                           orient=self.orient, pix_per_cell=self.pix_per_cell,
                                           cell_per_block=self.cell_per_block,
                                           hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                           hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # # Fit a per-column scaler
        # X_scaler = StandardScaler().fit(X)
        # # Apply the scaler to X
        # scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        # rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)

        print('Using:', self.orient, 'orientations', self.pix_per_cell, 'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t1 = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t1, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        test_img = cv2.imread(cars[100])
        test_img_features = single_img_features(test_img, color_space=self.color_space,
                                                spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                                orient=self.orient, pix_per_cell=self.pix_per_cell,
                                                cell_per_block=self.cell_per_block,
                                                hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                                hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        features = X_scaler.transform(np.array(test_img_features).reshape(1, -1))
        prediction = svc.predict(features)
        print("prediction: %s" % (prediction))

        image = cv2.imread('test_images/test1.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        draw_image = np.copy(image)
        print("Predict cars for %s" % ('test_images/test1.jpg'))
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        #image = image.astype(np.float32)/255

        windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                               xy_window=(96, 96), xy_overlap=(0.5, 0.5))

        # window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)
        # plt.imshow(window_img)
        # draw_image = np.copy(image)
        print("num of windows: %s" % (len(windows)))

        hot_windows = search_windows(image, windows, svc, X_scaler, color_space=self.color_space,
                                     spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                     orient=self.orient, pix_per_cell=self.pix_per_cell,
                                     cell_per_block=self.cell_per_block,
                                     hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                     hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        print("num of hot_windows: %s" % (len(hot_windows)))

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        plt.imshow(window_img)

        # ystart = int(image.shape[0] * 0.50)
        # ystop = int(image.shape[0] * 0.95)

        # out_img, boxes = find_cars(image, ystart, ystop, 1.5, svc, X_scaler, self.orient, self.pix_per_cell,
        #                            self.cell_per_block, self.spatial_size, self.hist_bins)

        # plt.imshow(out_img)

        # Add heat to each box in box list
        # heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        # heat = add_heat(heat, boxes)

        # Apply threshold to help remove false positives
        # heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        # heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        # labels = label(heatmap)
        # draw_img = draw_labeled_bboxes(np.copy(image), labels)

        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(draw_img)
        # plt.title('Car Positions')
        # plt.subplot(122)
        # plt.imshow(heatmap, cmap='hot')
        # plt.title('Heat Map')
        # fig.tight_layout()
        plt.show()


if __name__ == "__main__":

    runner = unittest.TextTestRunner()
    itersuite = unittest.TestLoader().loadTestsFromTestCase(TestSearchClassify)
    runner.run(itersuite)
