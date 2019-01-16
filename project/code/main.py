import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
import pandas as pd 
import numpy as np
from scipy.misc import imresize, imread, imrotate
import random
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf 
import datetime
from stack_hourglass import net
from config import *
from tensorflow.python.framework import graph_util


def test():    
    result_floder = ("../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+".csv")
    landmark_len = len(landmarks)
    
    data_list = pd.read_csv('../data/test.csv').values
    data_list=data_list[0:5000]
    raw_image = tf.placeholder(tf.float32, [None, None, 3])
    image = tf.div(raw_image, 255.0)
    inputs = tf.expand_dims(image, 0)
    pred = net(inputs, landmark_len)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'tmp/hg_4')  
        with open(result_floder, 'w') as f:
            f.write('image_id,image_category,{}\n'.format(','.join(landmarks)))
            for data in tqdm(data_list):
                path, image_category = data
                image_path = os.path.join('../data/', path)
                im = imread(image_path)
                w, h, _ = im.shape
                pred_string = ['-1_-1_-1'] * len(landmarks)

                output = np.zeros([w, h, landmark_len])


                for scale in [0.8,0.9,1.0,1.1,1.2]:
                    random_rotate = random.choice([5,0,-5])
                    ro_im = imrotate(im, random_rotate)
                    scale_im = imresize(ro_im, scale)
                    raw_output = sess.run(pred, feed_dict={raw_image:scale_im})
                    # print raw_output.shape, output.shape
                    for i in range(landmark_len):
                        out_im = imresize(raw_output[0,:,:,i], [w, h])
                        out_im_inv_ro = imrotate(out_im, -random_rotate)
                        output[:,:,i] += out_im_inv_ro


                for i in range(landmark_len):
                    # heatmap = imresize(output[0,:,:,i], 4.0, interp='bilinear')
                    heatmap = output[:,:,i]
                    y, x = np.mean(np.where(heatmap==np.max(heatmap)), axis=1).astype(np.int)            
                    pred_string[i] = '{}_{}_1'.format(x, y)

                    # plt.title(landmarks[landmark_index])
                    # plt.imshow(im)
                    # plt.plot(x, y, 'r.')
                    # plt.imshow(heatmap, alpha=0.5)
                
                    # plt.show()
                f.write('{},{},{}\n'.format(path, image_category, ','.join(pred_string)))
                # # break


if __name__ == '__main__':
    # for style in [
    #               #'trousers', 
    #               #'skirt',
    #               'outwear',
    #               'blouse', 
    #               'dress'
    #               ]:
    test()