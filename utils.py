import numpy as np
import pandas as pd
import cv2
import glob
import tensorflow as tf


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def iou(rect_1, rect_2):
    # 0: left, 1: top, 2: right, 3: bottom
    x_overlap = tf.maximum(0.0, tf.minimum(rect_1[:, :, :, 2], rect_2[:, :, :, 2]) - tf.maximum(rect_1[:, :, :, 0], rect_2[:, :, :, 0]))
    y_overlap = tf.maximum(0.0, tf.minimum(rect_1[:, :, :, 3], rect_2[:, :, :, 3]) - tf.maximum(rect_1[:, :, :, 1], rect_2[:, :, :, 1]))
    area_1 = tf.abs(rect_1[:, :, :, 2] - rect_1[:, :, :, 0]) * tf.abs(rect_1[:, :, :, 1] - rect_1[:, :, :, 3])
    area_2 = tf.abs(rect_2[:, :, :, 2] - rect_2[:, :, :, 0]) * tf.abs(rect_2[:, :, :, 1] - rect_2[:, :, :, 3])
    intersection = x_overlap * y_overlap
    union = area_1 + area_2 - intersection
    return intersection / union


def show_bbox(img, label):
    for i in range(len(label)):
        for j in range(len(label[0])):
            item = label[i][j]
            item[0] = item[0]
            if item[0] < 0.2:
                continue
            item *= 256
            cv2.rectangle(img, (int(item[1] - item[3] / 2), int(item[2] - item[4] / 2)),
                          (int(item[1] + item[3] / 2), int(item[2] + item[4] / 2)), (255, 255, 0), 1)
            cv2.circle(img, (int((i + 0.5) * 256 / 32), int((j + 0.5) * 256 / 32)), 2, (0, 0, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TODO: DEBUG LOSS FUNCTION
def loss_function(y_, y_pred):
    # y_: [object/no object, x, y, w, h]
    # y_pred: [confidence, x, y, w, h]

    w_1 = 5
    w_2 = 1
    w_3 = 0.5

    tmp_1 = tf.stack([y_[:, :, :, 1] - y_[:, :, :, 3] / 2, y_[:, :, :, 2] - y_[:, :, :, 4] / 2,
                      y_[:, :, :, 1] + y_[:, :, :, 3] / 2, y_[:, :, :, 2] + y_[:, :, :, 4] / 2], axis=-1)
    tmp_2 = tf.stack([y_pred[:, :, :, 1] - y_pred[:, :, :, 3] / 2, y_pred[:, :, :, 2] - y_pred[:, :, :, 4] / 2,
                      y_pred[:, :, :, 1] + y_pred[:, :, :, 3] / 2, y_pred[:, :, :, 2] + y_pred[:, :, :, 4] / 2], axis=-1)
    iou_loss = iou(tmp_1, tmp_2)

    zeros = tf.zeros_like(iou_loss)
    zeros_five = tf.tile(tf.expand_dims(zeros, axis=-1), [1, 1, 1, 5])

    is_obj = tf.equal(y_[:, :, :, 0], 1)
    is_obj_iou = tf.where(is_obj, iou_loss, zeros)
    is_obj_y_pred = tf.where(tf.tile(tf.expand_dims(is_obj, axis=-1), [1, 1, 1, 5]), y_pred, zeros_five)
    is_obj_y_ = tf.where(tf.tile(tf.expand_dims(is_obj, axis=-1), [1, 1, 1, 5]), y_, zeros_five)
    is_obj_loss = tf.losses.mean_squared_error(is_obj_iou, is_obj_y_pred[:, :, :, 0])

    no_obj = tf.equal(y_[:, :, :, 0], 0)
    no_obj_iou = tf.where(no_obj, iou_loss, zeros)
    no_obj_pred = tf.where(no_obj, y_pred[:, :, :, 0], zeros)
    no_obj_loss = tf.losses.mean_squared_error(no_obj_iou, no_obj_pred)

    coord_loss = tf.losses.mean_squared_error(is_obj_y_[:, :, :, 1:3], is_obj_y_pred[:, :, :, 1:3])
    len_loss = tf.losses.mean_squared_error(tf.sqrt(tf.abs(is_obj_y_[:, :, :, 3:])), tf.sqrt(tf.abs(is_obj_y_pred[:, :, :, 3:])))

    loss = w_1 * coord_loss + w_1 * len_loss + w_2 * is_obj_loss + w_3 * no_obj_loss

    return loss, w_1 * coord_loss, w_1 * len_loss, w_2 * is_obj_loss, w_3 * no_obj_loss


def to_yolo_format(label):
    new_label = np.zeros([32, 32, 5], np.float32)
    box_size = 1 / 32

    for coords in label:
        p_1 = np.array([coords[0] - coords[2] / 2, coords[1] - coords[3] / 2])
        p_2 = np.array([coords[0] + coords[2] / 2, coords[1] + coords[3] / 2])

        center = np.mean([p_1, p_2], axis=0)
        center = (center / box_size).astype(np.int32)

        tmp = np.concatenate(([1], coords))
        new_label[center[0], center[1]] = tmp

    return new_label


def read_data(batch_num, n_batches):
    id_list = sorted([x[:-4] for x in glob.glob('new_dataset/*.png')])

    id_list = id_list[:]

    batch_size = len(id_list) // n_batches
    if (batch_num + 1) * batch_size > len(id_list):
        id_list = id_list[batch_num * batch_size:]
    else:
        id_list = id_list[batch_num * batch_size:(batch_num + 1) * batch_size]

    imgs = list()
    labels = list()

    for id in id_list:
        img = cv2.imread(id + '.png')
        label = pd.read_csv(id + '.txt').astype(np.float32).as_matrix()

        label /= 256
        label = to_yolo_format(label)

        imgs.append(img)
        labels.append(label)

    imgs = np.array(imgs)
    labels = np.array(labels)

    return [imgs, labels]


# a = tf.constant([[[[1, 3, 2, 5], [3, 5, 6, 7]], [[2, 4, 3, 7], [4, 4, 8, 8]]]], tf.float32)
# b = tf.constant([[[[1, 4, 3, 5], [3, 5, 5, 7]], [[2, 4, 3, 8], [4, 3, 9, 8]]]], tf.float32)
# p = iou(a, b)
#
# with tf.Session() as sess:
#     print(sess.run(p))