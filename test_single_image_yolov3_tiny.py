####

# edicted by Huangdebo
# test the model using ckpt file

# CMD:python test_single_image.py --input_image bird.jpg --class_name_path ./data/COCO.name --restore_path ./checkpoint/yolov3_tiny_COCO/model-step_30000_loss_0.075246_lr_0.0003061015
# ***

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model.yolov3 import yolov3
from model.yolov3_tiny import yolov3_tiny
# 相关参数
net_name = 'yolov3_tiny'
body_name = 'darknet19'
data_name = 'COCO'
ckpt_name = 'yolov3_tiny_my.cpkt'
img_path = "imgs/person3.jpg"
# 解析器
parser = argparse.ArgumentParser(description="%s test single image test procedure."%net_name)
parser.add_argument("--input_image", type=str, default=img_path,
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/tiny_yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/%s.name"%data_name,
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./checkpoint/yolov3_tiny_COCO/%s"%(ckpt_name),
                    help="The path of the weights to restore.")
args = parser.parse_args()
# 锚框
args.anchors = parse_anchors(args.anchor_path)
# 类别名
args.classes = read_class_names(args.class_name_path)
#类别数
args.num_class = len(args.classes)
#得到框的颜色种类
color_table = get_color_table(args.num_class)
# 读取图片
img_ori = cv2.imread(args.input_image)
# 得到图片大小(h*w)
height_ori, width_ori = img_ori.shape[:2]
# 缩放图片(416*416)
img = cv2.resize(img_ori, tuple(args.new_size))
# 转为rgb图片
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 转化为float型
img = np.asarray(img, np.float32)
# 归一化
img = img[np.newaxis, :] / 255.

with tf.Session() as sess:
	# 输入数据的占位符
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    # yolo_model = yolov3(args.num_class, args.anchors)
    # 得到训练模型 
    yolo_model = yolov3_tiny(args.num_class, args.anchors)
    with tf.variable_scope(net_name):
		# 得到多尺度框
        pred_feature_maps = yolo_model.forward(input_data, False)
    #得到预测值[边界框,置信度,类别]
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    
    
    # 预测的得分
    pred_scores = pred_confs * pred_probs
    # 用非极大值抑制,得到[边界框,置信度,类别] 
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.4, iou_thresh=0.5)
    # 重载模型
    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)
    # 得到结果
    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image将坐标缩放到原始图像
    boxes_[:, 0] *= (width_ori/float(args.new_size[0]))
    boxes_[:, 2] *= (width_ori/float(args.new_size[0]))
    boxes_[:, 1] *= (height_ori/float(args.new_size[1]))
    boxes_[:, 3] *= (height_ori/float(args.new_size[1]))
    
    
    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)
    # 得到所有边界框坐标
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i] 
        # 显示出来
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
    cv2.imshow('Detection result', img_ori)
    #cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(0)
