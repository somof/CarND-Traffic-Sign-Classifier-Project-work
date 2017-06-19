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
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

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
n_test = len(y_test)

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples    =", n_train)
print("Number of validation examples  =", n_validation)
print("Number of testing examples     =", n_test)
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

# 同じラベルの教師画像の平均値の分布
mean = np.zeros((len(images), 3)).astype(np.float)
stdv = np.zeros((len(images), 3)).astype(np.float)
for i in range(len(images)):
    for c in range(3):
        mean[i, c] = np.mean(images[i, :, :, c])
        stdv[i, c] = np.std(images[i, :, :, c])

cls_mean = np.zeros((43, 3)).astype(np.float)
cls_stdv = np.zeros((43, 3)).astype(np.float)

for cls in range(43):
    smean = mean[labels == cls, :]
    for c in range(3):
        cls_mean[cls, c] = np.mean(smean[:, c])
        cls_stdv[cls, c] = np.std(smean[:, c])

#

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
gr = plt.plot(range(43), cls_mean, '.')
plt.title('pixel mean vs label in ' + title)
plt.xlim(0, 43)
ax.set_xlabel("class(label)")
ax.set_ylabel("mean")
names = ['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
plt.xticks(range(43), names, rotation=-90, fontsize="small")
plt.legend(['ch0', 'ch1', 'ch2'], loc="best")
# plt.savefig('fig/pixel_mean_vs_label.png')
plt.show()

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
gr = plt.plot(range(43), cls_stdv, '.')
plt.title('pixel stdv vs label in ' + title)
plt.xlim(0, 43)
ax.set_xlabel("class(label)")
ax.set_ylabel("stdv")
plt.xticks(range(43), names, rotation=-90, fontsize="small")
plt.legend(['ch0', 'ch1', 'ch2'], loc="best")
plt.savefig('fig/pixel_stdv_vs_label.png')
plt.show()

exit(0)



#

w = 0.3
X0 = np.arange(43)
X1 = X0 + w
X2 = X1 + w

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
gr = plt.bar(X0, cls_stdv[:, 0], color='r', width=w, label='ch0', align="center")
gr = plt.bar(X1, cls_stdv[:, 1], color='g', width=w, label='ch1', align="center")
gr = plt.bar(X2, cls_stdv[:, 2], color='b', width=w, label='ch2', align="center")
names = ['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
plt.xticks(range(43), names, rotation=-90, fontsize="small")
plt.legend(['ch0', 'ch1', 'ch2'], loc="best")
plt.title('pixel stdv of mean vs label in ' + title)
plt.xlim(0, 43)
# plt.ylim(0, 120)
ax.set_xlabel("class(label)")
ax.set_ylabel("mean")
plt.savefig('fig/pixel_mean_stdv_vs_label.png')

for cls in range(43):
    sstdv = stdv[labels == cls, :]
    for c in range(3):
        cls_mean[cls, c] = np.mean(sstdv[:, c])
        cls_stdv[cls, c] = np.std(sstdv[:, c])

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
gr = plt.bar(X0, cls_stdv[:, 0], color='r', width=w, label='ch0', align="center")
gr = plt.bar(X1, cls_stdv[:, 1], color='g', width=w, label='ch1', align="center")
gr = plt.bar(X2, cls_stdv[:, 2], color='b', width=w, label='ch2', align="center")
names = ['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
plt.xticks(range(43), names, rotation=-90, fontsize="small")
plt.legend(['ch0', 'ch1', 'ch2'], loc="best")
plt.title('pixel stdv of stdv vs label in ' + title)
plt.xlim(0, 43)
# plt.ylim(0, 120)
ax.set_xlabel("class(label)")
ax.set_ylabel("stdv")
plt.savefig('fig/pixel_stdv_stdv_vs_label.png')

# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# gr = plt.plot(range(43), cls_mean, '.')
# plt.legend(['ch0', 'ch1', 'ch2'])
# plt.title('pixel mean vs label in ' + title)
# plt.xlim(0, 43)
# # plt.ylim(0, 120)
# ax.set_xlabel("class(label)")
# ax.set_ylabel("mean")
# plt.savefig('fig/pixel_mean_vs_label.png')
# # plt.show()

exit(0)

# 同じラベルの教師画像の表示


simages = images[labels == 21, :, :, :]

wtile = 8
htile = 4
displimit = wtile * htile
for no in range(displimit):
    if no < len(simages):
        ax = plt.subplot(htile, wtile, no + 1)
        plt.subplots_adjust(left=0.005, right=0.990, top=0.995, bottom=0.001, hspace=0.0, wspace=0.0)
        plt.title("%d" % no, fontsize=6)
        plt.axis("off")
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        plt.imshow(simages[no].reshape(32, 32, 3), cmap=None)
# plt.savefig('fig/XXX.png')
# exit(0)
# plt.show()


exit(0)



# ラベルの頻度分布
fig = plt.figure(figsize=(12,8))
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
plt.title('Histgram of Train/Valid/Test data')
names=['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
labels=['y_train', 'y_valid', 'y_test']
plt.hist([y_train, y_valid, y_test],
         range=(0, 43),
         rwidth=20,
         stacked=False,
         bins=43,
         label=labels
         )
plt.xticks(range(0,43), names, rotation=-90, fontsize="small")
plt.legend()
plt.savefig('fig/histgram_of_dataset_all.png')
# plt.show()

fig = plt.figure(figsize=(12,8))
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
plt.title('Histgram of Train/Valid data')
names=['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
labels=['y_train', 'y_valid', 'y_test']
plt.hist([y_train, y_valid],
         range=(0, 43),
         rwidth=20,
         stacked=False,
         bins=43,
         label=labels
         )
plt.xticks(range(0,43), names, rotation=-90, fontsize="small")
plt.legend()
plt.savefig('fig/histgram_of_dataset.png')
# plt.show()



