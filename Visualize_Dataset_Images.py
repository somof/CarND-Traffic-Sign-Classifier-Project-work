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




# Minimally, the image data should be normalized so that the data has mean zero and equal variance. 
# For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project.
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance.
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.



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

from PIL import Image

images = X_train
labels = y_train
title = 'X_train'

imglist = (3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3250, 3251, 3253, 3257, 3260, 3261)

for no in imglist:
    pil_img = Image.fromarray(X_valid[no])
    pil_img.save('examples/vdata_%05d.png' % no)
    

|No|input image|Inferenced sign|
|:-:|:-:|:-|:-|
|3240|![vdata3240](examples/vdata_03240.png)|<img width=32 src="examples/class12_00002_00013.png"/>|12:Priority road 
|3241|![vdata3241](examples/vdata_03241.png)|<img width=32 src="examples/class26_00001_00015.png"/>|26:Traffic signals 
|3242|![vdata3242](examples/vdata_03242.png)|<img width=32 src="examples/class26_00001_00015.png"/>|26:Traffic signals 
|3243|![vdata3243](examples/vdata_03243.png)|<img width=32 src="examples/class29_00004_00019.png"/>|29:Bicycles crossing 
|3244|![vdata3244](examples/vdata_03244.png)|<img width=32 src="examples/class31_00004_00015.png"/>|31:Wild animals crossing 
|3245|![vdata3245](examples/vdata_03245.png)|<img width=32 src="examples/class23_00001_00015.png"/>|23:Slippery road 
|3246|![vdata3246](examples/vdata_03246.png)|<img width=32 src="examples/class29_00004_00019.png"/>|29:Bicycles crossing 
|3247|![vdata3247](examples/vdata_03247.png)|<img width=32 src="examples/class31_00004_00015.png"/>|31:Wild animals crossing 
|3248|![vdata3248](examples/vdata_03248.png)|<img width=32 src="examples/class31_00004_00015.png"/>|31:Wild animals crossing 
|3250|![vdata3250](examples/vdata_03250.png)|<img width=32 src="examples/class31_00004_00015.png"/>|31:Wild animals crossing 
|3251|![vdata3251](examples/vdata_03251.png)|<img width=32 src="examples/class23_00001_00015.png"/>|23:Slippery road 
|3253|![vdata3253](examples/vdata_03253.png)|<img width=32 src="examples/class31_00004_00015.png"/>|31:Wild animals crossing 
|3257|![vdata3257](examples/vdata_03257.png)|<img width=32 src="examples/class23_00001_00015.png"/>|23:Slippery road 
|3260|![vdata3260](examples/vdata_03260.png)|<img width=32 src="examples/class31_00004_00015.png"/>|31:Wild animals crossing 
|3261|![vdata3261](examples/vdata_03261.png)|<img width=32 src="examples/class31_00004_00015.png"/>|31:Wild animals crossing 



|No|image|Inferenced sign|
|:-:|:-:|:-|:-|
|3240|![vdata3240](examples/vdata_03240.png)|![class12](examples/class12_00002_00013.png)|12:Priority road 
|3241|![vdata3241](examples/vdata_03241.png)|![class26](examples/class26_00001_00015.png)|26:Traffic signals 
|3242|![vdata3242](examples/vdata_03242.png)|![class26](examples/class26_00001_00015.png)|26:Traffic signals 
|3243|![vdata3243](examples/vdata_03243.png)|![class29](examples/class29_00004_00019.png)|29:Bicycles crossing 
|3244|![vdata3244](examples/vdata_03244.png)|![class31](examples/class31_00004_00015.png)|31:Wild animals crossing 
|3245|![vdata3245](examples/vdata_03245.png)|![class23](examples/class23_00001_00015.png)|23:Slippery road 
|3246|![vdata3246](examples/vdata_03246.png)|![class29](examples/class29_00004_00019.png)|29:Bicycles crossing 
|3247|![vdata3247](examples/vdata_03247.png)|![class31](examples/class31_00004_00015.png)|31:Wild animals crossing 
|3248|![vdata3248](examples/vdata_03248.png)|![class31](examples/class31_00004_00015.png)|31:Wild animals crossing 
|3250|![vdata3250](examples/vdata_03250.png)|![class31](examples/class31_00004_00015.png)|31:Wild animals crossing 
|3251|![vdata3251](examples/vdata_03251.png)|![class23](examples/class23_00001_00015.png)|23:Slippery road 
|3253|![vdata3253](examples/vdata_03253.png)|![class31](examples/class31_00004_00015.png)|31:Wild animals crossing 
|3257|![vdata3257](examples/vdata_03257.png)|![class23](examples/class23_00001_00015.png)|23:Slippery road 
|3260|![vdata3260](examples/vdata_03260.png)|![class31](examples/class31_00004_00015.png)|31:Wild animals crossing 
|3261|![vdata3261](examples/vdata_03261.png)|![class31](examples/class31_00004_00015.png)|31:Wild animals crossing 



exit(0)



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
    plt.savefig('fig/class%02d_images_training.png' % cls)
    # plt.show()


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



