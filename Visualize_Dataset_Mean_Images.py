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

images = X_train
labels = y_train
title = 'X_train'

wtile = 9
htile = (43 // wtile) + 1
clslist = range(43)

fig = plt.figure(figsize=(26,14))
fig.suptitle('the first image of each classes', fontsize=36)
plt.subplots_adjust(left=0.005, right=0.995, top=0.900, bottom=0.01, hspace=0.2, wspace=0.2)
for cls in clslist:
    image = np.zeros((32, 32, 3)).astype(np.float32)
    for i in range(len(images)):
        if labels[i] == cls:
            image = images[i].reshape(32, 32, 3)
            break
    ax = plt.subplot(htile, wtile, cls + 1)
    plt.title("class %d" % cls, fontsize=18)
    plt.axis("off")
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    plt.imshow(image, cmap=None)
plt.savefig('fig/sampled_43_images_in_%s.png' % title)
# plt.show()



X_train_org = X_train
nsigma = 2.0
labels = y_train
title = 'X_train'

X_train = X_train_org.astype(np.float32)
mean = np.mean(X_train[:, :, :, :])
stdv = np.std(X_train[:, :, :, :])
for i in range(len(X_train)):
    mean = np.mean(X_train[i, :, :, :])
    stdv = np.std(X_train[i, :, :, :])
    for c in range(3):
        X_train[i, :, :, c] = X_train[i, :, :, c] / 255.0

images = X_train
fig = plt.figure(figsize=(26,14))
fig.suptitle('averaged images of each classes without normalization', fontsize=36)
plt.subplots_adjust(left=0.005, right=0.995, top=0.900, bottom=0.01, hspace=0.2, wspace=0.2)
for cls in clslist:
    no = 0
    image = np.zeros((32, 32, 3)).astype(np.float32)
    for i in range(len(images)):
        if labels[i] == cls:
            image = image + images[i].reshape(32, 32, 3)
            no += 1
    if 0 < no:
        image = image / no
        image = image.clip(0, 1)
        ax = plt.subplot(htile, wtile, cls + 1)
        plt.title("class %d" % cls, fontsize=18)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(image, cmap=None)
plt.savefig('fig/mean_images_in_%s_wo_normalization.png' % title)
# plt.show()



X_train = X_train_org.astype(np.float32)
mean = np.mean(X_train[:, :, :, :])
stdv = np.std(X_train[:, :, :, :])
for i in range(len(X_train)):
    for c in range(3):
        X_train[i, :, :, c] = X_train[i, :, :, c] - mean
        X_train[i, :, :, c] = X_train[i, :, :, c] / (stdv * nsigma)
        X_train[i, :, :, c] = X_train[i, :, :, c] + 1.0
        X_train[i, :, :, c] = X_train[i, :, :, c] / 2.0
        X_train[i, :, :, c] = X_train[i, :, :, c].clip(0, 1)

images = X_train
fig = plt.figure(figsize=(26,14))
fig.suptitle('averaged images of each classes with type 0 normalization', fontsize=36)
plt.subplots_adjust(left=0.005, right=0.995, top=0.900, bottom=0.01, hspace=0.2, wspace=0.2)
for cls in clslist:
    no = 0
    image = np.zeros((32, 32, 3)).astype(np.float32)
    for i in range(len(images)):
        if labels[i] == cls:
            image = image + images[i].reshape(32, 32, 3)
            no += 1
    if 0 < no:
        image = image / no
        image = image.clip(0, 1)
        ax = plt.subplot(htile, wtile, cls + 1)
        plt.title("class %d" % cls, fontsize=18)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(image, cmap=None)
plt.savefig('fig/mean_images_in_%s_type0_normalization.png' % title)
# plt.show()


X_train = X_train_org.astype(np.float32)
for i in range(len(X_train)):
    mean = np.mean(X_train[i, :, :, :])
    stdv = np.std(X_train[i, :, :, :])
    for c in range(3):
        X_train[i, :, :, c] = X_train[i, :, :, c] - mean
        X_train[i, :, :, c] = X_train[i, :, :, c] / (stdv * nsigma)
        X_train[i, :, :, c] = X_train[i, :, :, c] + 1.0
        X_train[i, :, :, c] = X_train[i, :, :, c] / 2.0
        X_train[i, :, :, c] = X_train[i, :, :, c].clip(0, 1)

images = X_train
fig = plt.figure(figsize=(26,14))
fig.suptitle('averaged images of each classes with type 1 normalization', fontsize=36)
plt.subplots_adjust(left=0.005, right=0.995, top=0.900, bottom=0.01, hspace=0.2, wspace=0.2)
for cls in clslist:
    no = 0
    image = np.zeros((32, 32, 3)).astype(np.float32)
    for i in range(len(images)):
        if labels[i] == cls:
            image = image + images[i].reshape(32, 32, 3)
            no += 1
    if 0 < no:
        image = image / no
        image = image.clip(0, 1)
        ax = plt.subplot(htile, wtile, cls + 1)
        plt.title("class %d" % cls, fontsize=18)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(image, cmap=None)
plt.savefig('fig/mean_images_in_%s_type1_normalization.png' % title)
# plt.show()


X_train = X_train_org.astype(np.float32)
for i in range(len(X_train)):
    for c in range(3):
        mean = np.mean(X_train[i, :, :, c])
        stdv = np.std(X_train[i, :, :, c])
        X_train[i, :, :, c] = X_train[i, :, :, c] - mean
        X_train[i, :, :, c] = X_train[i, :, :, c] / (stdv * nsigma)
        X_train[i, :, :, c] = X_train[i, :, :, c] + 1.0
        X_train[i, :, :, c] = X_train[i, :, :, c] / 2.0
        X_train[i, :, :, c] = X_train[i, :, :, c].clip(0, 1)

images = X_train
fig = plt.figure(figsize=(26,14))
fig.suptitle('averaged images of each classes with type 2 normalization', fontsize=36)
plt.subplots_adjust(left=0.005, right=0.995, top=0.900, bottom=0.01, hspace=0.2, wspace=0.2)
for cls in clslist:
    no = 0
    image = np.zeros((32, 32, 3)).astype(np.float32)
    for i in range(len(images)):
        if labels[i] == cls:
            image = image + images[i].reshape(32, 32, 3)
            no += 1
    if 0 < no:
        image = image / no
        image = image.clip(0, 1)
        ax = plt.subplot(htile, wtile, cls + 1)
        plt.title("class %d" % cls, fontsize=18)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(image, cmap=None)
plt.savefig('fig/mean_images_in_%s_type2_normalization.png' % title)
# plt.show()

