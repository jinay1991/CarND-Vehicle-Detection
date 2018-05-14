import glob
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

import cv2
from detect import showImages


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Extracts X or Y Gradients using Sobel operation

    Returns:
        binary_output (ndarray) - 1D binary image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(gray)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    """
    Calculates and thresholds Gradient based on it's magnitude

    Returns:
        binary_output (ndarray) - 1D binary image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag > thresh[0]) & (gradmag < thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Calculates Gradient Direction and applies threshold (angle)

    Returns:
        binary_output (ndarray) - 1D binary image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(grad_direction)
    binary_output[(grad_direction > thresh[0]) & (grad_direction < thresh[1])] = 1
    return binary_output


def hls_select(img, thresh=(0, 255), index=2):
    """
    Selects channel from HLS color space and applies thresholds

    Returns:
        binary_output (ndarray) - 1D binary image
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    selected = hls[:, :, index]

    binary_output = np.zeros_like(selected)
    binary_output[(selected > thresh[0]) & (selected <= thresh[1])] = 1
    return binary_output


def apply_thresholds(image, display=True):
    """
    Apply different thresholding techiques to extract lane lines

    Returns:
        threshold (ndarray) - thresholded binary image
    """
    image = cv2.GaussianBlur(image, (5, 5), 0)

    height, width = image.shape[:2]

    # gradient threshold
    sx_binary = abs_sobel_thresh(image, 'x', sobel_kernel=7, thresh=(20, 100))
    sy_binary = abs_sobel_thresh(image, 'y', sobel_kernel=7, thresh=(80, 255))

    # gradient direction threshold
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))  # 13

    # magnitude gradient threshold
    mag_binary = mag_thresh(image, sobel_kernel=9, thresh=(30, 100))  # 9

    # combine threshold
    combine_binary = np.zeros_like(dir_binary)
    combine_binary[((sx_binary == 1) & (sy_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # color space - hls threshold
    s_binary = hls_select(image, thresh=(170, 255))

    # combine threshold
    color_binary = np.zeros_like(combine_binary)
    color_binary[(s_binary > 0) | (combine_binary > 0)] = 1

    # crop dst points
    left_bottom = (100, height)
    right_bottom = (width - 20, height)
    apex1 = (610, 410)
    apex2 = (680, 410)
    inner_left_bottom = (310, height)
    inner_right_bottom = (1150, height)
    inner_apex1 = (700, 480)
    inner_apex2 = (650, 480)
    vertices = np.array([[left_bottom, apex1, apex2,
                          right_bottom, inner_right_bottom,
                          inner_apex1, inner_apex2, inner_left_bottom]], dtype=np.int32)

    mask = np.zeros_like(color_binary)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(color_binary, mask)

    if display:
        showImages([("image", image), ("s_binary", s_binary), ("sx_binary", sx_binary),
                    ("sy_binary", sy_binary), ("mag_binary", mag_binary), ("dir_binary", dir_binary),
                    ("color_binary", color_binary), ("masked_image", masked_image)],
                   4, 2, figsize=(25, 7))

    return masked_image


def transform_points(width, height):
    """
    Forms Source/Destination points based on image dimensions

    Returns:
        src (ndarray) - Source Image points
        dst (ndarray) - Destination Image points
    """
    offset = 200
    src = np.float32([[580, 460], [710, 460], [1100, 720], [200, 720]])
    dst = np.float32([[offset, 0], [width - offset, 0], [width - offset, height], [offset, height]])

    return src, dst


def unwarp_image(image):
    """
    Performs Invert Perspective Transform to unwarp image

    Returns:
        unwarped (ndarray) - 2D unwarped image
    """
    height, width = image.shape[:2]

    src, dst = transform_points(width, height)

    Minv = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(image, Minv, (width, height))


def warp_image(image):
    """
    Performs Perspective Transform to warp image

    Returns:
        unwarped (ndarray) - 2D unwarped image
    """
    height, width = image.shape[:2]

    src, dst = transform_points(width, height)

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (width, height))


def fit_polynomial(binary_warped, left_fit, right_fit):
    """
    Fits polynomial on binary image to streamline/smoothen probable lane curve
    """
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return ploty, left_fitx, right_fitx


def find_curvature(yvals, fitx):
    """
    Computes curvature of line
    """
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(yvals)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meteres per pixel in x dimension
    fit_cr = np.polyfit(yvals*ym_per_pix, fitx*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad


def find_pos(l_fit, r_fit, w, h):
    """
    Computes position of car from left lane
    """
    xm_per_pix = 3.7/700  # meteres per pixel in x dimension
    car_position = w / 2
    l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
    r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center_dist = (car_position - lane_center_position) * xm_per_pix
    return center_dist


def find_peaks(binary_warped):
    """
    Computes Lane Point as start point by finding peaks in histogram of window
    """
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return leftx_base, rightx_base


def sliding_window(binary_warped, left_lane, right_lane, nwindows=9, margin=100, minpix=50):
    """
    Trace Lane line points by sliding window
    """
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    leftx_base, rightx_base = find_peaks(binary_warped)

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    if not (left_lane.detected or right_lane.detected):
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    else:
        left_fit = left_lane.current_fit[-1]
        right_fit = right_lane.current_fit[-1]
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)  # if len(leftx) != 0 else None
    right_fit = np.polyfit(righty, rightx, 2)  # if len(rightx) != 0 else None

    return left_fit, right_fit, left_lane_inds, right_lane_inds


def fitLines(binary_warped, left_lane, right_lane):
    """
    Fits Polynomial Lines on the probably lane points in top-down view
    """

    binary_warped[binary_warped > 0] = 255

    left_fit, right_fit, left_lane_inds, right_lane_inds = sliding_window(binary_warped, left_lane, right_lane)

    ploty, left_fitx, right_fitx = fit_polynomial(binary_warped, left_fit, right_fit)

    left_curverad = find_curvature(ploty, left_fitx)
    right_curverad = find_curvature(ploty, right_fitx)

    left_lane.add_line(left_fit, left_lane_inds)
    right_lane.add_line(right_fit, right_lane_inds)

    return left_fitx, right_fitx, left_fit, right_fit, left_lane_inds, right_lane_inds, left_curverad, right_curverad


def visualizeLines(binary_warped, left_lane_inds, right_lane_inds):
    """
    Visualizes lines
    """
    # Generate x and y values for plotting
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img


def visualizeLanes(undist, binary_warped, left_fitx, right_fitx, left_fit, right_fit, left_curve, right_curve):
    """
    Visualize lane line
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    newwarp = unwarp_image(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Put text on an image
    curvature = int((left_curve + right_curve) / 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: {} m".format(curvature)
    cv2.putText(result, text, (400, 100), font, 1, (255, 255, 255), 2)
    # Find the position of the car
    #pts = np.argwhere(newwarp[:,:,1])
    #position = find_position(pts, undist.shape[1])
    h, w = binary_warped.shape[:2]
    position = find_pos(left_fit, right_fit, w, h)
    if position < 0:
        text = "Vehicle is {:.2f} m left of center".format(-position)
    else:
        text = "Vehicle is {:.2f} m right of center".format(position)
    cv2.putText(result, text, (400, 150), font, 1, (255, 255, 255), 2)

    # plt.imshow(result)
    return result


def detect_lanes(image, mtx, dist, left_lane, right_lane, diag, display=False):
    """
    Complete Pipeline for finding lane lines
    """
    # Undistort image
    undist = cv2.undistort(image, mtx, dist)

    # Apply threshold schemes to extract Lanes
    threshold = apply_thresholds(image, False)

    # Perspective Transform to view top-down view of lanes
    warped = warp_image(threshold)

    # fit Lane Lines on top-down view of lanes
    lfx, rfx, lf, rf, l_ind, r_ind, lc, rc = fitLines(warped, left_lane, right_lane)

    # visualize lane lines on top-down view of lanes
    lanes_img = visualizeLines(warped, l_ind, r_ind)

    # Visualize lane lines on undistorted image 2D view
    result = visualizeLanes(undist, warped, lfx, rfx, lf, rf, lc, rc)

    if diag:
        threshold[threshold > 0] = 255
        thresh_3_channel = np.dstack((threshold, threshold, threshold))

        result1 = np.hstack((undist, result))
        result2 = np.hstack((thresh_3_channel, lanes_img))
        result = np.vstack((result1, result2))

    if display:
        showImages([("image", undist), ("thresh", threshold),
                    ("lanes_img", lanes_img), ("result", result)],
                   4, 1, figsize=(20, 9))
    return result
