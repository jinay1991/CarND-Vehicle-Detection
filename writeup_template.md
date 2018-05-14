## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image50]: ./output_images/test1_out.jpg
[image51]: ./output_images/test2_out.jpg
[image52]: ./output_images/test3_out.jpg
[image53]: ./output_images/test4_out.jpg

[image05]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./output_images/test6_out.jpg
[image07]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

#### Notes:
All the source code is form of multiple `*.py` files as below:

1. `main.py`: Starter with `main` function
2. `classifier.py`: Contains functions `features` and `train` for extracting features and training `svm` classifier for given feature set, respectively.
3. `detect.py`: Contains functions `detect_cars` - a high level API to perform detection of cars based on the method chosen `['find_cars', 'search_windows', 'ssd']`. More details given in section `XXXXX`
4. `lane.py` and `line.py`: Contains code for Lane detection from `P4-Advance Lane Finding` ([github]( https://github.com/jinay1991/CarND-Advanced-Lane-Lines`))
5. `dnn.py`: Contains code to use `Single Shot Detection MobileNet v2 COCO` model from TensorFlow Model Zoo ([link](`https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md`)) with TensorFlow API (Details provided in Section ``Extras``)
6. `calib.py`: Contains function `calibrate` and `load` for performing camera calibration and loading of exiting pickled matrix.

Pickle files `svc.p` and `calib.p` are also saved for reference as Trained Classifier and calibration matrix, respectively.

#### Usage of `main`

Refer below arguments to tune any parameters:

    usage: main.py [-h] [--train] [--sample_size SAMPLE_SIZE] [--fname FNAME]
               [--dataset_dir DATASET_DIR] [--color_space COLOR_SPACE]
               [--orient ORIENT] [--pix_per_cell PIX_PER_CELL]
               [--cell_per_block CELL_PER_BLOCK] [--hog_channel HOG_CHANNEL]
               [--spatial_size SPATIAL_SIZE SPATIAL_SIZE]
               [--hist_bins HIST_BINS] [--spatial_feat] [--hist_feat]
               [--hog_feat] [--heat_threshold HEAT_THRESHOLD]
               [--method METHOD] [--detect_vehicles] [--detect_lanes] [--save]
               [--augment]

    optional arguments:
    -h, --help                                  show this help message and exit
    --train                                     train classifier
    --sample_size SAMPLE_SIZE                   classifier sample size for training
    --fname FNAME                               input video/image
    --dataset_dir DATASET_DIR                   dataset directory (vehicles/non-vehicles)
    --color_space COLOR_SPACE                   can be RGB, HSV, LUV, HLS, YUV, YCrCb
    --orient ORIENT                             HOG orientations
    --pix_per_cell PIX_PER_CELL                 HOG pixels per cell
    --cell_per_block CELL_PER_BLOCK             HOG cells per block
    --hog_channel HOG_CHANNEL                   HOG channels, can be 0, 1, 2 or 'ALL'
    --spatial_size SPATIAL_SIZE SPATIAL_SIZE    Spatial binning dimensions (w, h)
    --hist_bins HIST_BINS                       Number of histogram bins
    --spatial_feat                              Spatial features on or off
    --hist_feat                                 Histogram features on or off
    --hog_feat                                  HOG features on or off
    --heat_threshold HEAT_THRESHOLD             Heatmap threshold
    --method METHOD                             algo ['find_cars', 'search_windows', 'ssd']
    --detect_vehicles                           vehicle detection on or off
    --detect_lanes                              lane detection on or off
    --save                                      saves result as result.mp4
    --augment                                   augment dataset to generate more features

*Example command lines:*

    $ python main.py --fname test_video.mp4 --color_space 'YCrCb' --orient 9 --pix_per_cell 8 --cell_per_block 2 --spatial_size 32 32 --hist_bins 32 --hog_channel 'ALL' --heat_threshold 6 --method 'search_windows' --detect_lanes --detect_vehicles --save --train

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines `16` through `53` of the file called `classifier.py` as function `features()`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

With this, while training classifier I found that adding more training data could help eliminating false positives for car detection hence I flipped the original images of dataset with `cv2.flip` and added features to respective class. This helped in improving classifier accuracy.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and here I mainly focused on accuracy with speed. Hence, fall for following combination of HOG Parameters most suited for me.

    color_space = `YCrCb`
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL'

Apart from HOG Features I have also used `color features` information with following parameters

    spatial_size = (32, 32)
    hist_bins = 32

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn.svm.LinearSVC` with default arguments. Code for this is in `classifier.py` written as function `train()`.

To train a classifier I have used HOG features along with color features (`spatial_feature` and `hist_feature`) containing feature vector of size `8560`. I have tried reducing further by setting `pix_per_cell = 16` but results were not favorable and lot of false positives for `svm` hence kept most optimal setting for training.

Training SVM took `35.77 seconds` for training `35520` images (with augmented dataset) which gave test accuracy of `99.17%`.

    car images: 8792, notcar images: 8968
    Using: 9 orientations 8 pixels per cell 2 cells per block YCrCb color_space (32, 32) spatial_size 32 hist_bins and ALL hog_channel True augment for features extract
    35520 image 35520 labels
    Feature vector length: 8460
    35.77 Seconds to train SVC...
    Test Accuracy of SVC =  0.9917
    Predictions: [0. 1. 1. 1. 0. 0. 0. 1. 1. 0.]
        Labels: [0. 1. 1. 1. 0. 0. 0. 1. 1. 0.]

    0.001 seconds to predict 10 labels with SVC.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Now to localizing the detections I chose sliding window search keeping fixed region where posibilities of finding vehicles is maximum. By tweaking, I ended up setting following `[xstart, xstop]` and `[ystart, ystop]`.

    [xstart, xstop] = [None, None]
    [ystart, ystop] = [396, 598]
    xy_window = (96, 96)
    xy_overlap = (0.75, 0.75)

I used `slide_window` function from lesson (code at line `103` to `145` in `lesson_functions.py`) for generating list of windows with above mentioned configurations and I then applied `search_windows` function from lesson (code at line `204` to `234` in `lesson_functions.py`) for finding cars with in each window by extracting feature vector and predicting with SVM.

I have implemented these in my pipeline which is at line `72` to `80` in `detect.py` within function `detect_cars`.

*Note that, `detect_cars()` performs various methods for obtaining car bounding boxes such as `find_cars`, `ssd` (Single Shot Detection) or `search_windows`. To run with any of the algorithm use `--method` argument of `main.py`.*

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on twp scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used `search_windows` method for generating project result video. Although `find_cars` works just fine with existing code. For `find_cars` I have chosen `4` scales to support multi-scale sub-sampling and their respective step size as given below

| ystart | ystop | scale | step |
|--------|:-----:|:-----:|:----:|
|  396   |  470  |  1.0  |   1  |
|  396   |  496  |  1.5  |   2  |
|  396   |  570  |  2.0  |   2  |
|  396   |  620  |  2.5  |   2  |

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.

I have used `deque` for storing past `3` heat signatures for detected vehicles, which I average over frames to get vehicle detection stable.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image50]
![alt text][image51]
![alt text][image52]
![alt text][image53]

Heatmaps for each of the test image can be seen in above images (center image of right column)

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have used two different approach during the project `find_cars` (performing `hog` feature extraction once and sub-sample the feature array to extract windows which contain vehicles) and `search_windows` (perform `hog` feature extraction for each window and extract windows which contain vehicles). Both of them have done decent job in identifying vehicles but there are few fall backs in the technique itself. They are -

1. Processing Speed: As `hog` feature extraction is very costly operation and running it for all the windows is not prefered for performance and even with `find_cars` where this is being done only once, we need to have multi-scale objects to be detected hence to find cars at multi-scale we iterate through each scale for this which is time consuming and at same time computionally expensives. To overcome this, I would highly recommend Deep Learning as they out-performs this technique in both factors accuracy as well as speed. (see extra section below)

2. Tracking Pipeline: I have used averaging heatmap over past 3 frames which helped me in reducing false positives but at the same time my latency for detecting new object dropped, which is not accepted in mosts of the cases as it may lead to hazardus situation in self-driven cars. To improve this I would like to try out something like Kalman Prediction techniques which allows me to track the detected object and detection pipeline gets slighly isolated from the tracking pipeline. (i.e. detection happens and later it gets tracked only. No detection for already detected object for certain period)

3. Lane/Vehicle Detection: Lane detection pipeline seems to throttle down the performance. To improve this I would recommend to use existing approach to generate labeled masks for Deep Learning datasets and train Sematic Segmantation Model (Masked-RCNN) to produce mask for Lane region as well as Car and other objects.

4. Distance objects: We can not detect distant objects with existing pipeline even though we add small scale, as they tend to produce false positives for Simple SVM classifier. We need to have strong classifier for detecting small objects such as LeNet.

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

---

### Extras

#### 1. Provide combined result containing pipeline of *P4: Advance Lane Finding* Pipeline to work alongside with Vehicle Detection Pipeline.

Here, I have integrated Lane detection pipeline with Vehicle Detection pipeline to work alongside to have complete solution.

Here's a [link to my video result](./project_video_out.mp4)

Couple of example frame where Lane Lines and Vehicle Detection Pipeline working together.

![alt text][image8]

#### 2. Discuss deep learning approach. How deep learning out-performs SVM results with real-time processing?

In above section I have learned how object localization/detection can be done with `hog` + `svm` + `search_window` technique but this technique seems to have lot of fall backs in terms of speed and accuracy. As you can see in my result video, entire pipeline process at `~1-2fps` without any parallism which is far from the real-time, whereas self-driving car needs to detect object in real-time (`~15-20fps`).

Although with Deep Learning Models such as `YOLO (You Only Look Once)` and `SSD (Single Shot Detection)` out-performs Object Detection tasks at speed of `~20fps` without batch. This are really decent figures for Self-Driving Cars to work. Hence I attempted to do inference with pretrained model of `SSD (MobileNet v2)` trained on `MSCOCO` dataset.

Code for performing infernce with TensorFlow(TM) APIs is in `dnn.py`, which can be run with `--method 'ssd'` in the command argument of `main.py`


Couple of example frame of detection pipeline using `SSD MobileNetv2` model.

![alt text][image9]