import os
import random
import numpy as np
from scipy.misc import imresize, imrotate
from sklearn.utils import shuffle
import cv2
import pandas as pd 
import tensorflow as tf 

from config import *

def read_csv(csvfile):
    text = pd.read_csv(csvfile)

    csv_dict = {'image_path':[], 'labels':[], 'category':[]}
    for row in text.values:
        image_id, image_category = row[:2]

        image_path = os.path.join('../data/train', image_id)
        labels = np.array([label.split('_') for label in row[2:]], dtype=np.int).tolist()
        csv_dict['image_path'].append(image_path)
        csv_dict['labels'].append(labels)
        csv_dict['category'].append(image_category)

    return csv_dict


def parse_function(image_path, label, category):
    image_contents = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.cast(image, tf.float32)
    # image = tf.div(image, 255.0)
    # image = tf.image.per_image_standardization(image)

    return image, label, category

def read_data(batch_size=1):

    with tf.name_scope('input_pipeline'):

        data_dict = read_csv('../data/train/Annotations/train.csv')
        image_list = data_dict['image_path']
        labels_list = data_dict['labels'] 
        category_list = data_dict['category']

        image_list, labels_list, category_list = shuffle(image_list, labels_list, category_list)

        dataset = tf.data.Dataset.from_tensor_slices((image_list, labels_list, category_list))

        dataset = dataset.map(parse_function)
        dataset = dataset.shuffle(10).batch(batch_size).repeat()
        return dataset

def gen_traindata(images, label, categorys, scale=4.0, sigma=3.0):

    random_scale = random.uniform(0.75, 1.25)
    random_rotation = random.uniform(-10, 10)
    random_flip = random.choice([True, False])
    
    n, w, h, _ = images.shape

    ow = int(w/scale*random_scale)
    oh = int(h/scale*random_scale)

    landmark_len = len(landmarks)
    heatmaps = np.zeros([n, ow, oh, landmark_len])
    train_images = []
    train_calculable = []
    x = np.arange(0, h, 1, np.float32)
    y = np.arange(0, w, 1,np.float32)[:,np.newaxis]

    for batch in range(n):
        
        image = images[batch]
        image = imresize(image, random_scale)
        image = imrotate(image, random_rotation)
        image = cv2.flip(image, 1) if random_flip else image
        train_images.append(image/255.0)


        category = str(categorys[batch],'utf-8')
        train_calculable.append(calculable_dict[category])

        points = label[batch]
        for index in range(landmark_len):
            
            x0, y0, v = points[index]
            if x0 <= 0 or y0 <= 0:
                continue
            heatmap = np.exp(-((x - x0) **2 + 
                                (y - y0) ** 2) / 2.0 / sigma / sigma)

            heatmap = imresize(heatmap, random_scale/scale) 
            heatmap = imrotate(heatmap, random_rotation)
            heatmap = cv2.flip(heatmap, 1) if random_flip else heatmap
            heatmaps[batch,:,:,index] = heatmap
            
        if random_flip:
            for symmetry_point in symmetry:
                
                heatmaps[batch,:,:,symmetry_point] = heatmaps[batch,:,:,symmetry_point[::-1]]

    return np.array(train_images),heatmaps, np.array(train_calculable)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    

    data = read_data()

    iterator = data.make_initializable_iterator()
    images, outputs, categorys = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for _ in xrange(100):
            # print image.shape, heatmap.shape
            print (calculable)
            print (category)
            plt.subplot(121)
            plt.imshow((image[0]*255).astype(np.uint8)) 
            plt.subplot(122)
            plt.show()  
            # for i in xrange(len(landmarks)):
            #     plt.title(landmarks[i])
            #     plt.imshow((image[0,:,:,:]*255).astype(np.uint8))
            #     plt.imshow(imresize(heatmap[0,:,:,i], 4.0), alpha=0.5)
            #     plt.show()
            #     plt.clf()
            # break