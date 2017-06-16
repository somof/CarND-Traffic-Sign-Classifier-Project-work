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

# 平均輝度の分布

print(X_train.shape)
print(X_train.dtype)
print(type(X_train))

mean = np.zeros((len(X_train), 3)).astype(np.float)
stdv = np.zeros((len(X_train), 3)).astype(np.float)

X_train = X_train.astype(np.float)

for i in range(len(X_train)):
    for c in range(3):
        mean[i, c] = np.mean(X_train[i, :, :, c])
        stdv[i, c] = np.std(X_train[i, :, :, c])

fig = plt.figure()
ax = fig.add_subplot(111)
gr = plt.plot(mean, stdv, '.')
plt.legend(['ch0', 'ch1', 'ch2'])
plt.title('pixel mean vs stdv in X_train')
plt.xlim(0, 255)
plt.ylim(0, 150)
ax.set_xlabel("mean")
ax.set_ylabel("stdv")
plt.savefig('fig/pixel_mean_vs_stdv_in_X_train.png')
# plt.show()
# exit(0)

for i in range(len(X_train)):
    for c in range(3):
        X_train[i, :, :, c] = X_train[i, :, :, c] - mean[i, c]
        X_train[i, :, :, c] = X_train[i, :, :, c] / (stdv[i, c] * 2.0)

for i in range(len(X_train)):
    for c in range(3):
        mean[i, c] = np.mean(X_train[i, :, :, c])
        stdv[i, c] = np.std(X_train[i, :, :, c])

fig = plt.figure()
ax = fig.add_subplot(111)
gr = plt.plot(mean, stdv, '.')
plt.legend(['ch0', 'ch1', 'ch2'])
plt.title('pixel mean vs stdv in X_train (normalized)')
plt.xlim(-1, 1)
plt.ylim(0, 1)
ax.set_xlabel("mean")
ax.set_ylabel("stdv")
plt.savefig('fig/pixel_mean_vs_stdv_in_X_train_normalized.png')
# plt.show()
# exit(0)


exit(0)
for i in range(len(X_train)):
    for c in range(3):
        # mean[i, c] = np.mean(X_train[i, :, :, c])
        # stdv[i, c] = np.std(X_train[i, :, :, c])
        print('ch', c)
        print()
        print(' mean ', mean[i, c])
        print(' stdv ', stdv[i, c])
        print()
        print(X_train[i, :, :, c])
        print()
        # X_train[i, :, :, c] = X_train[i, :, :, c] * 0.5
        X_train[i, :, :, c] = X_train[i, :, :, c] - mean[i, c]
        X_train[i, :, :, c] = X_train[i, :, :, c] / (stdv[i, c] * 2.0)
        print(X_train[i, :, :, c])
        print()

        exit(0)

exit(0)


