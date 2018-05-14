# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import label

import cv2
from lesson_functions import (add_heat, apply_threshold, draw_boxes,
                              draw_labeled_bboxes, find_cars, search_windows, slide_window)

# INTERESTING_WIN_PROP = [
#     # (ystart, ystop, scale, step, color)
#     (400, 464, 1.0, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),
#     (416, 480, 1.0, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),

#     (400, 496, 1.5, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),
#     (432, 528, 1.5, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),

#     (400, 528, 2.0, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),
#     (432, 560, 2.0, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),

#     (400, 628, 3.5, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),
#     (464, 690, 3.5, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
# ]
# INTERESTING_WIN_PROP = [  # image dimensions 1280x720
#     # (ystart, ystop, scale, step, color)
#     (400, 656, 1.5, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),

#     (410, 475, 1.0, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),
#     (410, 520, 1.5, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),
#     (410, 550, 2.0, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),
#     (410, 660, 3.5, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
# ]
# INTERESTING_WIN_PROP = [
    # #(416, 490, 1.0, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),
    # (416, 570, 1.5, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),
    # #(428, 620, 2.0, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),
    # #(458, 690, 3.0, 2, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
# ]
INTERESTING_WIN_PROP = [
    (396, 660, 1.0, 4, (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))),

]


def showImages(images, cols=4, rows=5, figsize=(15, 10), cmaps=None):
    """
    Display `images` on a [`cols`, `rows`] subplot grid.
    """
    imgLength = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.tight_layout()

    indexes = range(cols * rows)
    for ax, index in zip(axes.flat, indexes):
        if index < imgLength:
            imagePathName, image = images[index]
            if cmaps == None:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmaps[index])
            ax.set_title(imagePathName)
            ax.axis('off')

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def detect_cars(rgb_image, clf, scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space,
                hog_channel, hog_feat, spatial_feat, hist_feat, heat_thresh=1, display=False, method='find_cars', lanes_img=None):
    """
    Performs multi-scale find_cars for given feature configurations
    """
    height, width = rgb_image.shape[:2]
    boxes_img = rgb_image.copy()
    boxes = []
    i = 0
    if method.lower() == "find_cars":
        for ystart, ystop, scale, step, color in INTERESTING_WIN_PROP:

            _, boxes_ = find_cars(rgb_image, ystart=ystart, ystop=ystop, scale=scale, svc=clf, X_scaler=scaler,
                                  orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                  spatial_size=spatial_size, hist_bins=hist_bins, color_space=color_space,
                                  step=step, hog_feat=hog_feat, hog_channel=hog_channel, hist_feat=hist_feat,
                                  spatial_feat=spatial_feat)

            boxes_img = draw_boxes(boxes_img, boxes_, color=color)

            # Add legends to boxes image
            cv2.putText(boxes_img, str(scale), (50 * i, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            i += 1

            boxes += boxes_
    elif method.lower() == "search_windows":
        windows = slide_window(rgb_image, x_start_stop=[None, None], y_start_stop=[396, 598],
                               xy_window=(96, 96), xy_overlap=(0.75, 0.75))

        boxes = search_windows(rgb_image, windows, clf=clf, scaler=scaler, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                               orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                               spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    else:
        raise NotImplementedError("Provided method %s is not supported. Supported methods are ['find_cars', 'search_windows']" % (method))

    # Add heat to each box in box list
    heat = np.zeros_like(rgb_image[:, :, 0]).astype(np.float)

    heat = add_heat(heat, boxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heat_thresh)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(rgb_image), labels)

    if display:
        showImages([("boxes_img", boxes_img), ("heatmap", heatmap), ("draw_img", draw_img)], 3, 1,
                   cmaps=[None, 'hot', None])
        plt.figure(figsize=(40, 40))
        plt.imshow(boxes_img)
        return draw_img
    else:
        if lanes_img is not None:
            rw = width // 3
            rh = height // 3
            rlanes_img = cv2.resize(lanes_img, (rw, rh))
            draw_img = draw_labeled_bboxes(np.copy(lanes_img), labels)
        else:
            rw = width // 2
            rh = height // 2
        rboxes = cv2.resize(boxes_img, (rw, rh))
        heatmap_img = cv2.applyColorMap(heatmap.astype(np.uint8) * 10, cv2.COLORMAP_HOT)
        rheat = cv2.resize(heatmap_img, (rw, rh))

        if lanes_img is not None:
            stacked = np.vstack((rboxes, rheat, rlanes_img))
        else:
            stacked = np.vstack((rboxes, rheat))

        result = np.hstack((draw_img, stacked))
        return result
