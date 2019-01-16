import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
import pandas as pd 
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import tensorflow as tf 

from stack_hourglass import net
from config import *


def test(nstack=4):    

    landmark_len = len(landmarks)

    # data_list = pd.read_csv('../test/test.csv').values
    data_list = pd.read_csv('E:/Python_Code/FashionAI/train/Annotations/test.csv').values[:,:2]

    filename = tf.placeholder(tf.string)
    file_contents = tf.read_file(filename)
    image_raw = tf.image.decode_jpeg(file_contents, channels=3)
    image = tf.cast(image_raw, tf.float32)
    # image = tf.image.per_image_standardization(image)
    image = tf.div(image, 255.0)
    inputs = tf.expand_dims(image, 0)
    pred = net(inputs, landmark_len, nStack=nstack)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'models/hg_4')
        with open('result.csv', 'w') as f:
            f.write('image_id,image_category,{}\n'.format(','.join(landmarks)))
            for data in tqdm(data_list):
    #             fname = data[style][index]
                path, image_category = data 
                image_path = os.path.join('E:/Python_Code/FashionAI/train', 'test', path)
                pred_string = []
                im, output = sess.run([image_raw, pred], feed_dict={filename:image_path})
                

                for i in range(landmark_len):
                    heatmap = imresize(output[0,:,:,i], 4.0, interp='bilinear')
                    y, x = np.mean(np.where(heatmap==np.max(heatmap)), axis=1).astype(np.int)
                    # landmark_index = dresses[style]['position'][i]                 
                    # pred_string[landmark_index] = '{}_{}_1'.format(x, y)
                    pred_string.append('{}_{}_1'.format(x, y))
                    # plt.title(landmarks[i])
                    # plt.imshow(im)
                    # plt.plot(x, y, 'r.')
                    # plt.imshow(heatmap, alpha=0.5)
                
                    # plt.show()
                f.write('{},{},{}\n'.format(path, image_category, ','.join(pred_string)))
                # break


if __name__ == '__main__':

    test()