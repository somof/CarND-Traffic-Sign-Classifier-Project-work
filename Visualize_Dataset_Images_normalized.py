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
# with open(validation_file, mode='rb') as f:
#     valid = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
# X_valid, y_valid = valid['features'], valid['labels']
# X_test, y_test = test['features'], test['labels']

X_train_org = X_train

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

# How many unique classes/labels there are in the dataset.



### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.


X_train = X_train.astype(np.float32)
nsigma = 2.0

imagelist = (0, 1000, 10000)

for i in imagelist:
    # mean = np.mean(X_train[i, :, :, :])
    # stdv = np.std(X_train[i, :, :, :])
    for c in range(3):
        mean = np.mean(X_train[i, :, :, c])
        stdv = np.std(X_train[i, :, :, c])
        X_train[i, :, :, c] = X_train[i, :, :, c] - mean
        X_train[i, :, :, c] = X_train[i, :, :, c] / (stdv * nsigma)
        X_train[i, :, :, c] = X_train[i, :, :, c] + 1.0
        X_train[i, :, :, c] = X_train[i, :, :, c].clip(0, 1)



wtile = 2
htile = len(imagelist)
limit = wtile * htile
fig = plt.figure(figsize=(6,8))
plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05, hspace=0.2, wspace=0.2)
fig.suptitle('training data normalization samples')

no = 0
for i in imagelist:
    ax = plt.subplot(htile, wtile, no + 1)
    plt.title("No.%d original" % i, fontsize=6)
    # plt.axis("off")
    # plt.tick_params(labelbottom="off")
    # plt.tick_params(labelleft="off")
    plt.imshow(X_train_org[i].reshape(32, 32, 3), cmap=None)
    no = no + 1
    #
    ax = plt.subplot(htile, wtile, no + 1)
    plt.title("No.%d normalized" % i, fontsize=6)
    # plt.axis("off")
    # plt.tick_params(labelbottom="off")
    # plt.tick_params(labelleft="off")
    plt.imshow(X_train[i].reshape(32, 32, 3), cmap=None)
    no = no + 1

plt.savefig('fig/images_normalized_for_training.png')
plt.show()

exit(0)









# Visualization




# Minimally, the image data should be normalized so that the data has mean zero and equal variance. 
# For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project.
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance.
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


clslist = (16, 21, 20, 25, 24, 0)
clslist = (16, 21, 20, 24, 25, 0, 5, 27, 8, 29, 23, 22)
clslist = (22, )

# images = X_valid
# labels = y_valid
# title = 'X_valid'
# fig = plt.figure(figsize=(26,14))
# wtile = 16
# htile = 6
# limit = wtile * htile
# no = 0
# for i in range(len(images)):
#     if labels[i] in clslist and no < limit:
#         # print(no)
#         ax = plt.subplot(htile, wtile, no + 1)
#         plt.subplots_adjust(left=0.005, right=0.990, top=0.995, bottom=0.001, hspace=0.0, wspace=0.0)
#         plt.title("%d" % no, fontsize=6)
#         plt.axis("off")
#         plt.tick_params(labelbottom="off")
#         plt.tick_params(labelleft="off")
#         plt.imshow(images[i].reshape(32, 32, 3), cmap=None)
#         no += 1
# plt.savefig('fig/class%02d_images_valid.png' % clslist[0])
# plt.show()



images = X_train
labels = y_train
title = 'X_train'

clslist = range(43)
for cls in clslist:
    fig = plt.figure(figsize=(26,14))
    wtile = 40  # 30
    htile = 16  # 12
    limit = wtile * htile  # 2000
    no = 0
    for i in range(len(images)):
        if labels[i] == cls:
            if no % 2 == 0 and no < limit:
                ax = plt.subplot(htile, wtile, no//2 + 1)
                plt.subplots_adjust(left=0.005, right=0.990, top=0.995, bottom=0.001, hspace=0.0, wspace=0.0)
                plt.title("%d" % no, fontsize=6)
                plt.axis("off")
                plt.tick_params(labelbottom="off")
                plt.tick_params(labelleft="off")
                plt.imshow(images[i].reshape(32, 32, 3), cmap=None)
            no += 1
    plt.savefig('fig/class%02d_images_training.png' % cls)
    # plt.show()

exit(0)












# 34799 image ->

wtile = 24
htile = 9
start = 0
limit = 210
for i in range(0, wtile * htile):
    no = start + i
    if 0 <= no and no < limit and no < len(X_train):
        # print(no)
        ax = plt.subplot(htile, wtile, i + 1)
        plt.subplots_adjust(left=0.005, right=0.990, top=0.995, bottom=0.001, hspace=0.0, wspace=0.0)
        plt.title("%d" % no, fontsize=6)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(X_train[no].reshape(32, 32, 3), cmap=None)
plt.savefig('fig/TrainingImageValiationSample.png')  # bbox_inches="tight", pad_inches=0.0)
# exit(0)
# plt.show()



wtile = 48
htile = 25
unit = int(len(X_train) / (wtile * htile))
print('skip %d each image.' % unit)
for no in range(0, len(X_train), unit):
    i = int(no / unit)
    j = int(i / wtile)
    if j < htile:
        print(j, i % wtile, no)
        ax = plt.subplot(htile, wtile, i + 1)
        plt.subplots_adjust(left=0.005, right=0.990, top=0.995, bottom=0.001, hspace=0.0, wspace=0.0)
        # plt.title("%d" % no, fontsize=6)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(X_train[no].reshape(32, 32, 3), cmap=None)

plt.savefig('fig/AllTrainingImage_skip28.png')  # bbox_inches="tight", pad_inches=0.0)
# plt.show()



