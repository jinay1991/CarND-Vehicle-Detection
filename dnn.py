import os
import sys
import tarfile

import numpy as np
import tensorflow as tf

import cv2
from six.moves import urllib


def maybe_download_and_extract(data_url, clean_dir=False):
    """
    Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.

    Args:
        data_url: Web location of the tar file containing the pretrained model.

    Returns:
        dest_directory: Destination directory where files were extracted
    """
    dest_directory = os.path.join(os.path.abspath(os.path.curdir), "tmp")
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    if clean_dir:
        for item in os.listdir(dest_directory):
            if item.endswith(".pb") or item.endswith(".txt"):
                os.remove(os.path.join(dest_directory, item))

    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print("")
        statinfo = os.stat(filepath)
        print('Successfully downloaded %s %d bytes.' % (filename, statinfo.st_size))
    else:
        print('Not downloading files, model gzip already present in disk')

    print('Extracting file from %s' % filepath)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    return dest_directory


class TFDetect(object):
    def __init__(self):
        dstPath = maybe_download_and_extract(
            "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz")
        # load pretrained graph using TF api
        with tf.gfile.GFile(os.path.join(dstPath, "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb"), "rb") as fp:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fp.read())

        with tf.Graph().as_default() as self.graph:
            tf.import_graph_def(graph_def)

        tensor_names = [op.name for op in self.graph.get_operations()]
        print("first 5 layers:\n", tensor_names[:5], "\nlast 5 layers:\n", tensor_names[-5:])

        # Obtain Tensor object fetched from it's name in graph
        self.image_tensor = self.graph.get_tensor_by_name("import/image_tensor:0")  # input image placeholder
        self.detection_boxes = self.graph.get_tensor_by_name("import/detection_boxes:0")  # bounding boxes (ymin, xmin, ymax, xmax)
        self.detection_scores = self.graph.get_tensor_by_name("import/detection_scores:0")  # score (float)
        self.detection_classes = self.graph.get_tensor_by_name("import/detection_classes:0")  # classid (int)
        self.num_detections = self.graph.get_tensor_by_name("import/num_detections:0")  # number of detections

        self.session = tf.Session(graph=self.graph)

    def detect(self, img):
        """
        Performs object detection with SSD COCO MobileNet v2 using TensorFlow APIs
        """

        boxes, scores, classes, n = self.session.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                                                     feed_dict={self.image_tensor: np.expand_dims(img, axis=0)})
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        n = np.squeeze(n).astype(np.int32)

        # Draw boxes on image
        vis = img.copy()
        height, width = vis.shape[:2]
        bboxes = []
        for i in range(boxes.shape[0]):
            classid = classes[i] - 1

            if scores[i] > 0.3:
                ymin, xmin, ymax, xmax = boxes[i]

                x1 = int(xmin * width)
                y1 = int(ymin * height)
                x2 = int(xmax * width)
                y2 = int(ymax * height)

                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 100), 4)
                bboxes.append(((x1, y1), (x2, y2)))

        return bboxes, vis
