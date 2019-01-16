import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import queue
import threading

from tqdm import tqdm
import itertools
import pandas as pd 
import numpy as np
from scipy.misc import imresize, imread, imrotate
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf 

from stack_hourglass import net
from config import *


landmark_len = len(landmarks)
data_list = pd.read_csv('../data/test.csv').values

def feed_data():

    for data in data_list:
        path, image_category = data
        image_path = os.path.join('..', 'data', path)
        im = imread(image_path)


        for scale, rotate in aug_list:
            aug_im = imrotate(im, rotate)
            aug_im = imresize(aug_im, scale)        
            feed_msg.put(aug_im)

    feed_msg.put(None)


def result():

    with open('result.csv', 'w') as f:
        f.write('image_id,image_category,{}\n'.format(','.join(landmarks)))
        for data in tqdm(data_list):
            path, image_category = data
            image_path = os.path.join('..', 'data', path)
            im = imread(image_path)
            w, h, _ = im.shape
            pred_string = ['-1_-1_-1'] * len(landmarks)
            output = np.zeros([w, h, landmark_len])

            for scale, rotate in aug_list:

                raw_output = result_msg.get()
                for i in xrange(landmark_len):
                    channel = raw_output[0,:,:,i]
                    channel = imrotate(channel, -rotate)
                    output[:,:,i] += imresize(channel, [w, h])

            for i in xrange(landmark_len):
                # heatmap = imresize(output[0,:,:,i], 4.0, interp='bilinear')
                heatmap = output[:,:,i]
                y, x = np.mean(np.where(heatmap==np.max(heatmap)), axis=1).astype(np.int)            
                pred_string[i] = '{}_{}_1'.format(x, y)
            f.write('{},{},{}\n'.format(path, image_category, ','.join(pred_string)))


def test():    

    raw_image = tf.placeholder(tf.float32, [None, None, 3])
    image = tf.div(raw_image, 255.0)
    inputs = tf.expand_dims(image, 0)
    pred = net(inputs, landmark_len)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'models_2/hg_4')

        while True:
            
            aug_im = feed_msg.get()
            if aug_im is None:
                break

            raw_output = sess.run(pred, feed_dict={raw_image:aug_im})
            # print raw_output.shape
            result_msg.put(raw_output)


if __name__ == '__main__':

    scale_range = [0.8, 0.9, 1.0, 1.1, 1.2]
    rotate_range = [-5, 0, 5]
    aug_list = list(itertools.product(scale_range, rotate_range))

    feed_msg = queue.Queue(maxsize=10)
    result_msg = queue.Queue(maxsize=10)

    producer = threading.Thread(target=feed_data)
    consumer = threading.Thread(target=test)
    consumer2 = threading.Thread(target=result)

    producer.start()
    consumer.start()
    consumer2.start()
    # feed_msg.join()
    # result_msg.join()