# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model.yolov3 import yolov3
from model.yolov3_tiny import yolov3_tiny

ckpt_name = 'yolov3_tiny_my.cpkt'

parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
parser.add_argument("--input_video", type=str, default="./videos/test_video.mp4",
                  help="The path of the input video.")
parser.add_argument("--anchor_path", type=str, default="./data/tiny_yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/COCO.name",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./checkpoint/yolov3_tiny_COCO/%s"%(ckpt_name),
                    help="The path of the weights to restore.")
parser.add_argument("--save_video", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to save the video detection results.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

#vid = cv2.VideoCapture(args.input_video)
vid = cv2.VideoCapture(0)

video_width = int(vid.get(3))
video_height = int(vid.get(4))


if args.save_video:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, 20.0, (video_width, video_height))

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3_tiny(args.num_class, args.anchors)
    with tf.variable_scope('yolov3_tiny'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.5, iou_thresh=0.5)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    while vid.isOpened():
        ret, img_ori = vid.read()

        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        start_time = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        end_time = time.time()

        # rescale the coordinates to the original image
        boxes_[:, 0] *= (width_ori/float(args.new_size[0]))
        boxes_[:, 2] *= (width_ori/float(args.new_size[0]))
        boxes_[:, 1] *= (height_ori/float(args.new_size[1]))
        boxes_[:, 3] *= (height_ori/float(args.new_size[1]))


        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
        cv2.putText(img_ori, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.imshow('image', img_ori)
        if args.save_video:
            videoWriter.write(img_ori)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
    if args.save_video:
        videoWriter.release()
