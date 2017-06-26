import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


# Step 0: Load The Data

# Load pickled data
import pickle

training_file   = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file    = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
# X_test, y_test = test['features'], test['labels']

# Step 1: Dataset Summary & Exploration

# The pickled data is a dictionary with 4 key/value pairs:
# 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
# 'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
# 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the pandas shape method might be useful for calculating some of the summary results.

# 'features' : 4D配列  raw pixel data (num examples, width, height, channels).
# 'labels'   : 1D配列  label/class from signnames.csv
# 'sizes'    : リスト (width, height) original size
# 'coords'   : タプルのリスト [(x1, y1, x2, y2), ...]  bounding box の位置


# extra traininig dataset

from PIL import Image

print(X_train[5060].shape)
sample = X_train[5060].reshape(32, 32, 3)
print(sample.shape)

Image.fromarray(np.uint8(sample)).save('./test_sample_org.jpg')

cropped = tf.random_crop(sample, [32, 32, 3])
result = tf.image.per_image_standardization(cropped)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        img = sess.run(result)
        print(img.shape)
        Image.fromarray(np.uint8(img)).save('./test_cropped_img{}.jpg'.format(i))
        
exit(0)




print(images.shape)

cls = 16
fig = plt.figure(figsize=(26,19))
fig.suptitle('class%02d: training dataset' % cls, fontsize=64)
plt.subplots_adjust(left=0.005, right=0.995, top=0.945, bottom=0.005, hspace=0.0, wspace=0.0)
wtile = 24
htile = 15
limit = wtile * htile  # 2000
no = 0
for i in range(len(images)):
    if labels[i] == cls and no < limit:
        ax = plt.subplot(htile, wtile, no + 1)
        plt.title("%d" % no, fontsize=12)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(images[i].reshape(32, 32, 3), cmap=None)
        no += 1
# plt.savefig('fig/class%02d_images_training_extra.png' % cls)
plt.show()



exit(0)




### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results


# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_validation = len(y_valid)

# Number of testing examples.
# n_test = len(y_test)

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples    =", n_train)
print("Number of validation examples  =", n_validation)
# print("Number of testing examples     =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.


# Visualization




# Minimally, the image data should be normalized so that the data has mean zero and equal variance. 
# For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project.
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance.
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.






images = X_train
labels = y_train
title = 'X_train'

# clslist = (22, )
# clslist = (16, 21, 20, 25, 24, 0)
# clslist = (16, 21, 20, 24, 25, 0, 5, 27, 8, 29, 23, 22)

clslist = (16, 21, 40, 20, 24, 27)
for cls in clslist:
    fig = plt.figure(figsize=(26,19))
    fig.suptitle('class%02d: training dataset' % cls, fontsize=64)
    plt.subplots_adjust(left=0.005, right=0.995, top=0.945, bottom=0.005, hspace=0.0, wspace=0.0)
    wtile = 24
    htile = 15
    limit = wtile * htile  # 2000
    no = 0
    for i in range(len(images)):
        if labels[i] == cls and no < limit:
            ax = plt.subplot(htile, wtile, no + 1)
            plt.title("%d" % no, fontsize=12)
            plt.axis("off")
            plt.tick_params(labelbottom="off")
            plt.tick_params(labelleft="off")
            plt.imshow(images[i].reshape(32, 32, 3), cmap=None)
            no += 1
    # plt.savefig('fig/class%02d_images_training_extra.png' % cls)
    plt.show()

exit(0)
