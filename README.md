
# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/test1_out.jpg
[image4]: ./output_images/windows_combined.jpg
[image41]: ./output_images/windows_64_0.75.jpg
[image42]: ./output_images/windows_96_0.75.jpg
[image43]: ./output_images/windows_128_0.5.jpg
[image50]: ./output_images/test1_out.jpg
[image51]: ./output_images/test2_out.jpg
[image52]: ./output_images/test3_out.jpg
[image53]: ./output_images/test4_out.jpg
[image54]: ./output_images/test5_out.jpg
[image55]: ./output_images/test6_out.jpg
[image6]: ./examples/labels_map.png
[image7]: ./output_images/test6_out.jpg
[image8]: ./output_images/test4_lane_out.jpg
[image9]: ./output_images/test4_dnn_out.jpg
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

    $ python main.py --fname project_video.mp4 --color_space 'YUV' --orient 11 --pix_per_cell 16 --cell_per_block 2 --spatial_size 32 32 --hist_bins 32 --hog_channel 'ALL' --heat_threshold 3 --method 'search_windows' --detect_vehicles --save 1 --train

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines `16` through `53` of the file called `classifier.py` as function `features()`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and here I mainly focused on accuracy with speed. Hence, fall for following combination of HOG Parameters most suited for me.

    color_space = `YUV`
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL'

Apart from HOG Features I have also used `color features` information with following parameters

    spatial_size = (32, 32)
    hist_bins = 32

Although, I also observed that `color_space = 'YCrCb'` works fine too.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn.svm.LinearSVC` with default arguments. Code for this is in `classifier.py` written as function `train()`.

To train a classifier I have used HOG features along with color features (`spatial_feature` and `hist_feature`) containing feature vector of size `4356`.

Training SVM took `10.15 seconds` for training `17760` images (without augmented dataset) which gave test accuracy of `99.16%`.

    car images: 8792, notcar images: 8968
    Using: 11 orientations 16 pixels per cell 2 cells per block YUV color_space (32, 32) spatial_size 32 hist_bins and ALL hog_channel False augment for features extract
    17760 images 17760 labels
    Feature vector length: 4356
    10.15 Seconds to train SVC...
    Test Accuracy of SVC =  0.9916
    Predictions: [1. 1. 1. 1. 0. 1. 1. 0. 1. 1.]
        Labels: [1. 1. 1. 1. 0. 1. 1. 0. 1. 1.]

    0.00147 seconds to predict 10 labels with SVC.

##### Command Line

    $ python .\main.py --fname .\project_video.mp4 --color_space 'YUV' --orient 11 --pix_per_cell 16 --cell_per_block 2 --spatial_size 32 32 --hist_bins 32 --hog_channel 'ALL' --heat_threshold 3 --method 'search_windows' --detect_vehicles --train

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Now to localizing the detections I chose sliding window search keeping fixed region where posibilities of finding vehicles is maximum. By tweaking, I ended up setting following `[xstart, xstop]` and `[ystart, ystop]`.

    [xstart, xstop] = [None, None]
    [ystart, ystop] = [396, 598]
    xy_window = (96, 96)
    xy_overlap = (0.75, 0.75)

I used `slide_window` function from lesson (code at line `103` to `145` in `lesson_functions.py`) for generating list of windows with above mentioned configurations and I then applied `search_windows` function from lesson (code at line `204` to `234` in `lesson_functions.py`) for finding cars with in each window by extracting feature vector and predicting with SVM.

I have implemented these in my pipeline which is at line `72` to `80` in `detect.py` within function `detect_cars`.

Later, I have applied `search_windows` for following scales to improve accuracy of detections

| xstart | xstop | ystart | ystop | scale | step |
|:-------|:-----:|:-------|:-----:|:-----:|:----:|
|  380   | None  |  400   |  480  |  1.0  |   3  |
|  240   | None  |  396   |  510  |  1.5  |   3  |
|  120   | None  |  380   |  620  |  2.0  |   2  |

where,

    xy_window = (64 * scale)
    xy_overlap = (0.25 * step)

(Note that None means it's max/min limit of image.)

In other words I have chosen (64, 64), (96, 96) and (128, 128) window scales for this.

Here, I have assumed for the `project_video.mp4` and `test_video.mp4` that Car will be always in left most lane and hence all the ongoing traffic cars will come from right side only hence to improve speed of pipeline I have reduced X direction of each scale windows.

*Note that, `detect_cars()` performs various methods for obtaining car bounding boxes such as `find_cars`, `ssd` (Single Shot Detection) or `search_windows`. To run with any of the algorithm use `--method` argument of `main.py`.*

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:


![alt text][image50]
![alt text][image52]


Here, I have assumed for the `project_video.mp4` and `test_video.mp4` that Car will be always in left most lane and hence all the ongoing traffic cars will come from right side only hence to improve speed of pipeline I have reduced X direction of each scale windows.

Also, for reduction of the false positives from the video, I have used `decision_function` from the `LinearSVC()`, which gave `confidence` score for the prediction, and based on experiments I have thresholded all the prediction results with `0.35` (i.e. `35%`) to remove false positives. Code for this is at line `12`, `233-235` in `lesson_functions.py`.

---

### Video Implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_out.mkv)

Here's a [link to my Lane detection + Vehicle detection video result](./project_video_combined_out.mkv)

##### Command line

    $ python main.py --fname test_video.mp4 --color_space 'YUV' --orient 11 --pix_per_cell 16 --cell_per_block 2 --spatial_size 32 32 --hist_bins 32 --hog_channel 'ALL' --heat_threshold 3 --method 'search_windows' --detect_vehicles --save 1

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used `search_windows` method for generating project result video. Although `find_cars` works just fine with existing code.

For `search_windows` approach, following are the window properties and scale used

| xstart | xstop | ystart | ystop | scale | step |
|:-------|:-----:|:-------|:-----:|:-----:|:----:|
|  380   | None  |  400   |  480  |  1.0  |   3  |
|  240   | None  |  396   |  510  |  1.5  |   3  |
|  120   | None  |  380   |  620  |  2.0  |   2  |

where,

    xy_window = (64 * scale)
    xy_overlap = (0.25 * step)

For `find_cars` approach, following are the window properties and scale used

| xstart | xstop | ystart | ystop | scale | step |
|:-------|:-----:|:-------|:-----:|:-----:|:----:|
|  380   | 1100  |  400   |  457  |  0.7  |  2   |
|  240   | None  |  400   |  480  |  1.0  |  3   |
|  120   | None  |  410   |  520  |  1.5  |  3   |
|  None  | None  |  415   |  560  |  2.0  |  2   |
|  None  | None  |  430   |  620  |  2.5  |  2   |

(Note that None means it's max/min limit of image.)

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.

I have used `deque` for storing past `5` heat signatures for detected vehicles, which I average over frames to get vehicle detection stable.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image50]
![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]
![alt text][image55]

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

2. Tracking Pipeline: I have used averaging heatmap over past `5` frames which helped me in reducing false positives but at the same time my latency for detecting new object dropped, which is not accepted in mosts of the cases as it may lead to hazardus situation in self-driven cars. To improve this I would like to try out something like Kalman Prediction techniques which allows me to track the detected object and detection pipeline gets slighly isolated from the tracking pipeline. (i.e. detection happens and later it gets tracked only. No detection for already detected object for certain period)

3. Lane/Vehicle Detection: Lane detection pipeline seems to throttle down the performance. To improve this I would recommend to use existing approach to generate labeled masks for Deep Learning datasets and train Sematic Segmantation Model (Masked-RCNN) to produce mask for Lane region as well as Car and other objects.

4. Distance objects: We can not detect distant objects with existing pipeline even though we add small scale, as they tend to produce false positives for Simple SVM classifier. We need to have strong classifier for detecting small objects such as LeNet.

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

---

### Extras

#### 1. Provide combined result containing pipeline of *P4: Advance Lane Finding* Pipeline to work alongside with Vehicle Detection Pipeline.

Here, I have integrated Lane detection pipeline with Vehicle Detection pipeline to work alongside to have complete solution.

Here's a [link to my video result](./project_video_combined_out.mkv)

Example frame where Lane Lines and Vehicle Detection Pipeline working together.

![alt text][image8]

#### 2. Discuss deep learning approach. How deep learning out-performs SVM results with real-time processing?

In above section I have learned how object localization/detection can be done with `hog` + `svm` + `search_window` technique but this technique seems to have lot of fall backs in terms of speed and accuracy. As you can see in my result video, entire pipeline process at `~1-2fps` without any parallism which is far from the real-time, whereas self-driving car needs to detect object in real-time (`~15-20fps`) on my Intel i7.

Although with Deep Learning Models such as `YOLO (You Only Look Once)` and `SSD (Single Shot Detection)` out-performs Object Detection tasks at speed of `~20fps` without batch. This are really decent figures for Self-Driving Cars to work. Hence I attempted to do inference with pretrained model of `SSD (MobileNet v2)` trained on `MSCOCO` dataset.

Code for performing infernce with TensorFlow(TM) APIs is in `dnn.py`, which can be run with `--method 'ssd'` in the command argument of `main.py`

Here's a [link to my video result based on dnn model](./project_video_dnn_out.mkv)

Example frame of detection pipeline using `SSD MobileNetv2` model.

![alt text][image9]