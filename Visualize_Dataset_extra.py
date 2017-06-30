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


X_train = X_train.astype(np.float32)
num_train = len(X_train)
print(len(X_train))
print(X_train.shape)

# extra traininig dataset

from PIL import Image
import cv2

# class16
orgimages = [X_train[5010].reshape(32, 32, 3),
             X_train[5030].reshape(32, 32, 3),
             X_train[5112].reshape(32, 32, 3),
             X_train[5130].reshape(32, 32, 3),]

# low chroma on red
# for org in orgimages:

extra_num = 0
for ans, org in zip(y_train, X_train):

    if False and 16 == ans:
        # まずRGBの確認 0:R, 1:G, 2:B
        img = np.zeros((32, 32, 3)).astype(np.float32)
        img = org.astype(np.float32) / 255.0
        Vnoise = np.random.randn(32, 32) * 0.01
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        fig = plt.figure()  # (figsize=(26,19))
        fig.suptitle('class 16: augment sample', fontsize=24)
        plt.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005, hspace=0.2, wspace=0.1)

        ax = plt.subplot(1, 2, 1)
        plt.title("original", fontsize=12)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(img)

        # print('org min / max = ', np.min(org), np.max(org))
        # print('img min / max = ', np.min(img), np.max(img))
        # print('noise min / max = ', np.min(Vnoise), np.max(Vnoise))
        # print('  H min / max = ', np.min(hsv[:, :, 0]), np.max(hsv[:, :, 0]))
        # print('  S min / max = ', np.min(hsv[:, :, 1]), np.max(hsv[:, :, 1]))
        # print('  V min / max = ', np.min(hsv[:, :, 2]), np.max(hsv[:, :, 2]))

        hsv[:, :, 0] = hsv[:, :, 0] + 30
        hsv[:, :, 1] = hsv[:, :, 1] * 0.4
        hsv[:, :, 1] = hsv[:, :, 1].clip(.05, 0.95)
        hsv[:, :, 2] = hsv[:, :, 2] + Vnoise + 0.1
        hsv[:, :, 2] = hsv[:, :, 2].clip(.05, 0.95)

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        ax = plt.subplot(1, 2, 2)
        plt.title("augmented", fontsize=12)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(img)
        plt.show()

        X_train = np.append(X_train, img)
        y_train = np.append(y_train, ans)
        extra_num += 1

    elif 21 == ans:
    
        img = np.zeros((32, 32, 3)).astype(np.float32)
        img = org.astype(np.float32) / 255.0
        Vnoise = np.random.randn(32, 32) * 0.08
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        fig = plt.figure()  # (figsize=(26,19))
        fig.suptitle('class 21: augment sample', fontsize=24)
        plt.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005, hspace=0.2, wspace=0.1)

        ax = plt.subplot(1, 2, 1)
        plt.title("original", fontsize=12)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(img)

        # print('org min / max = ', np.min(org), np.max(org))
        # print('img min / max = ', np.min(img), np.max(img))
        # print('noise min / max = ', np.min(Vnoise), np.max(Vnoise))
        # print('  H min / max = ', np.min(hsv[:, :, 0]), np.max(hsv[:, :, 0]))
        # print('  S min / max = ', np.min(hsv[:, :, 1]), np.max(hsv[:, :, 1]))
        # print('  V min / max = ', np.min(hsv[:, :, 2]), np.max(hsv[:, :, 2]))

        # hsv[:, :, 0] = hsv[:, :, 0] + 30
        # hsv[:, :, 1] = hsv[:, :, 1] * 0.4
        # hsv[:, :, 1] = hsv[:, :, 1].clip(.05, 0.95)
        hsv[:, :, 2] = hsv[:, :, 2] + Vnoise
        hsv[:, :, 2] = hsv[:, :, 2].clip(.05, 0.95)

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        ax = plt.subplot(1, 2, 2)
        plt.title("augmented", fontsize=12)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(img)
        plt.show()

        X_train = np.append(X_train, img)
        y_train = np.append(y_train, ans)
        extra_num += 1

    elif False and (40 == ans or 24 == ans):
    
        img = np.zeros((32, 32, 3)).astype(np.float32)
        img = org.astype(np.float32) / 255.0
        Vnoise = np.random.randn(32, 32) * 0.08
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        fig = plt.figure()  # (figsize=(26,19))
        fig.suptitle('class {}: augment sample'.format(ans), fontsize=24)
        plt.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005, hspace=0.2, wspace=0.1)

        ax = plt.subplot(1, 2, 1)
        plt.title("original", fontsize=12)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(img)

        # print('org min / max = ', np.min(org), np.max(org))
        # print('img min / max = ', np.min(img), np.max(img))
        # print('noise min / max = ', np.min(Vnoise), np.max(Vnoise))
        # print('  H min / max = ', np.min(hsv[:, :, 0]), np.max(hsv[:, :, 0]))
        # print('  S min / max = ', np.min(hsv[:, :, 1]), np.max(hsv[:, :, 1]))
        # print('  V min / max = ', np.min(hsv[:, :, 2]), np.max(hsv[:, :, 2]))

        # hsv[:, :, 0] = hsv[:, :, 0] + 30
        # hsv[:, :, 1] = hsv[:, :, 1] * 0.4
        # hsv[:, :, 1] = hsv[:, :, 1].clip(.05, 0.95)
        hsv[:, :, 2] = hsv[:, :, 2] * 0.2
        hsv[:, :, 2] = hsv[:, :, 2].clip(.05, 0.95)

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        ax = plt.subplot(1, 2, 2)
        plt.title("augmented", fontsize=12)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(img)
        plt.show()

        X_train = np.append(X_train, img)
        y_train = np.append(y_train, ans)
        extra_num += 1


print('X_train augmented')
X_train = X_train.reshape(num_train + extra_num, 32, 32, 3)
print(len(X_train))
print(X_train.shape)
print("Number of training examples    =", n_train)











exit(0)


# following is trial to augment



# print(X_train[5060].shape)
sample = X_train[5060].reshape(32, 32, 3)
#print(sample.shape)
#sample = sample.clip(0, 255)
#sample = sample * 2
print('original: ')
isample = np.uint8(sample)
print(isample.shape)
sample = sample / 255.0
Image.fromarray(isample).save('./test_sample_org.jpg')

inputimg = tf.placeholder(tf.float32, shape=[32, 32, 3])
extra_cropped = tf.random_crop(inputimg, [30, 30, 3])
extra_brightness = tf.image.random_brightness(inputimg, max_delta=0.3)
extra_contrast = tf.image.random_contrast(inputimg, lower=0.1, upper=0.7)


funcs = [extra_brightness]
funcs = [extra_cropped]
funcs = [extra_contrast, extra_brightness, extra_cropped]
random_num = 1
extra_num = len(funcs) * random_num
print(extra_num)

print('sample')
print(sample)
dataset = sample
with tf.Session() as sess:
    for func in funcs:
        for i in range(random_num):
            nimg = sess.run(func, feed_dict={inputimg: sample})
            img = Image.fromarray(np.uint8(nimg * 255))
            img = img.resize((32, 32), Image.LANCZOS)
            img = np.array(img).astype(np.float32)
            img = -1.0 * img
            # plt.imshow(img)
            # plt.show()

            # print('img')
            # print(img)
            dataset = np.concatenate((dataset, img), axis=0)
            # dataset = np.concatenate((dataset, sample), axis=0)

            # dataset = np.append(dataset, img)
            # dataset = np.array((dataset, img))
            X_train = np.append(X_train, img)


print('X_train')
X_train = X_train.reshape(num_train + extra_num, 32, 32, 3)
print(len(X_train))
print(X_train.shape)

for i in range(extra_num):
    plt.imshow(X_train[num_train + i, :, :, :].reshape(32, 32, 3))
    plt.show()


print('dataset')
plt.imshow(dataset)
plt.show()

dataset = dataset.reshape((extra_num + 1, 32, 32, 3))
print(len(dataset))
print(dataset.shape)

for img in dataset:
    plt.imshow(img)
    plt.show()
    


exit(0)


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
