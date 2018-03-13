import numpy as np
import pandas as pd
import cv2
import glob


def read_data(batch_num, n_batches):
    id_list = sorted([x[:-4] for x in glob.glob('new_dataset/*.png')])
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

        label /= 255

        label = np.concatenate((np.ones([label.shape[0], 1]), label), axis=-1)

        # TODO: convert label to YOLO 16*16*5 format

        imgs.append(img)
        labels.append(label)

    imgs = np.array(imgs)
    labels = np.array(labels)

    labels = np.expand_dims(labels, axis=-1)

    return [imgs, labels]
