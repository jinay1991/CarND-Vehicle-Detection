# -*- coding: utf-8 -*-

import glob
import os
import pickle
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from lesson_functions import extract_features


def features(dataset_dir, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, hog_feat,
             spatial_feat, hist_feat, sample_size=None, augment=False):
    """
    Extracts features for car and notcar images
    """
    cars = glob.glob(os.path.join(dataset_dir, "vehicles", "*", "*.png"))
    notcars = glob.glob(os.path.join(dataset_dir, "non-vehicles", "*", "*.png"))
    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    if sample_size and sample_size < len(cars) and sample_size < len(notcars):
        # sample_size = 1500
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]

    print("car images: %s, notcar images: %s" % (len(cars), len(notcars)))

    print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell', cell_per_block, 'cells per block', color_space,
          'color_space', spatial_size, 'spatial_size', hist_bins, 'hist_bins and', hog_channel, 'hog_channel', augment, 'augment for features extract')

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat, augment=augment)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat, augment=augment)
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    return X, y

def train(X, y, save=True):
    """
    Train LinearSVC classifier for given feature set
    """
    print(len(X), "images", len(y), "labels")
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Feature vector length:', len(X_train[0]))

    # Note: To get even better classification results you might want to fine tune the C parameter of LinearSVC classifier
    #       which controls between the training accuracy and generalization. As the test and training images came from the
    #       same source, high test accuracy might imply overfitting. Against overfitting you need more generalization,
    #       meaning C < 1.0 values. You can try 0.01 or even 0.0001.
    svc = LinearSVC(C=0.01)

    # Check the training time for the SVC
    t1 = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print(round(t2 - t1, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    t1 = time.time()
    print('  Predictions:', svc.predict(X_test[0:10]))
    print('       Labels:', y_test[0:10])
    t2 = time.time()
    print()
    print(round(t2 - t1, 5), 'seconds to predict 10 labels with SVC.')

    if save:
        pickle.dump({"svc": svc, "X_scaler": X_scaler}, open("svc.p", "wb"))

    return svc, X_scaler

def load():
    """
    load pretrained classifier from pickle

    Returns:
        clf (LinearSVC classifier)
        scaler (training data scaler)
    """
    assert os.path.exists("svc.p"), "Failed to locate 'svc.p'"
    svc = pickle.load(open("svc.p", "rb"))
    return svc['svc'], svc['X_scaler']
