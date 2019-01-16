import os
import time
import argparse
import queue
import threading

import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from scipy.misc import imresize

from stack_hourglass import net
from data_utils import read_data, gen_traindata
from config import *

def data_generator(batch_size):

    data = read_data(batch_size)
    iterator = data.make_initializable_iterator()
    batch_images, batch_outputs, batch_categorys = iterator.get_next() 
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(32000):
            images, label, category = sess.run([batch_images, batch_outputs, batch_categorys])
            im, hmp, c = gen_traindata(images, label, category)
            message.put([i, im, hmp, c])
        message.put(None)


def train(nstack):

    landmark_len = len(landmarks)
   
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    heatmap = tf.placeholder(tf.float32, shape=[None, None, None, landmark_len])
    calculable = tf.placeholder(tf.float32, shape=[None, landmark_len])
    pred = net(inputs, landmark_len, nStack=nstack)
    outputs = tf.get_collection('heatmaps')

    c_loss = tf.reduce_mean(tf.stack([tf.nn.l2_loss(calculable*(heatmap - o)) for o in outputs]))

    # loss = tf.reduce_sum(tf.stack([tf.nn.l2_loss(heatmap - o) for o in outputs]))

    optimizer = tf.train.AdamOptimizer(1e-3).minimize(c_loss)


    saver = tf.train.Saver()        
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, 'models/hg_{}'.format(nstack))
        st = time.time()
        
        # data = message.get()

        while True:
        # for i in xrange(300):
            data = message.get()
            if data is None:
                break
            step, im, mk ,hmp, c = data
            # print c 
            sess.run(optimizer, feed_dict={inputs:im, heatmap:hmp, calculable:c})
            if not step % 2000:
                c_e = sess.run(c_loss, feed_dict={inputs:im, heatmap:hmp, calculable:c})
                print (step, c_e, time.time() - st)
                st = time.time()

        
        saver.save(sess, 'tmp/hg_{}'.format(nstack))

        # p = sess.run(pred, feed_dict={inputs:im})
        # for i in xrange(landmark_len):
        
        #     plt.imshow(imresize(im[0,:,:,:], 1/4.0))
        #     plt.imshow(p[0,:,:,i], alpha=0.5)
        #     plt.show()            


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, choices=['0', '1', '2', '3'], default='0')
    parser.add_argument('-nstack', type=int, default=4)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    nstack = args.nstack

    batch_size = 1
    message = queue.Queue(50)

    data_generator(batch_size)
    train(nstack)

    producer.start()
    consumer.start()
    message.join()