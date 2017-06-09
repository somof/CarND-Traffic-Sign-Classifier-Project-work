# Preprocess Data
from sklearn.utils import shuffle

# Setup TensorFlow
import tensorflow as tf
from tensorflow.contrib.layers import flatten

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

# print('\ntrain: ')
# print('features: ', end='')
# print(train['features'].shape)
# print('labels: ', end='')
# print(train['labels'].shape)
# print('sizes: ', end='')
# print(train['sizes'].shape)
# print(train['sizes'])
# print('coords: ', end='')
# print(train['coords'].shape)
# print(train['coords'])

# print('\ntest: ')
# print('features: ', end='')
# print(test['features'].shape)
# print('labels: ', end='')
# print(test['labels'].shape)
# print('sizes: ', end='')
# print(test['sizes'].shape)
# print(test['sizes'])
# print('coords: ', end='')
# print(test['coords'].shape)
# print(test['coords'])

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

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



# Step 2: Design and Test a Model Architecture

# Design and implement a deep learning model that learns to recognize traffic
# signs. Train and test your model on the German Traffic Sign Dataset.

# The LeNet-5 implementation shown in the classroom at the end of the CNN lesson
# is a solid starting point. You'll have to change the number of classes and
# possibly the preprocessing, but aside from that it's plug and play!

# With the LeNet-5 solution from the lecture, you should expect a validation set
# accuracy of about 0.89. To meet specifications, the validation set accuracy
# will need to be at least 0.93. It is possible to get an even higher accuracy,
# but 0.93 is the minimum for a successful project submission.

# There are various aspects to consider when thinking about this problem:
#   - Neural network architecture (is the network over or underfitting?)
#   - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
#   - Number of examples per label (some have more than others).
#   - Generate fake data.

# Here is an example of a published baseline model on this problem. It's not
# required to be familiar with the approach used in the paper but, it's good
# practice to try to read papers like these.



### Preprocess the data here. It is required to normalize the data.  Other
### preprocessing steps could include converting to grayscale, etc.
### Feel free to use as many code cells as needed.

X_train, y_train = shuffle(X_train, y_train)

# いやいや 最初から94%越えちゃったけど
# With the LeNet-5 solution from the lecture,
# you should expect a validation set accuracy of about 0.89.
# To meet specifications, the validation set accuracy will need to be at least 0.93.
# It is possible to get an even higher accuracy,
# but 0.93 is the minimum for a successful project submission.


### Define your architecture here.
### Feel free to use as many code cells as needed.

EPOCHS      = 400
BATCH_SIZE  = 100
FILTER1_NUM =   6  #  10  #   6
FILTER2_NUM =  12  #  20  #  16
FRC1_NUM    =  64  # 100  # 120
FRC2_NUM    =  32  #  60  #  84
CLASS_NUM   = 43
MU          = 0
SIGMA       = 0.1


# Arguments used for tf.truncated_normal, randomly defines variables for
# the weights and biases for each layer
weights = {
    'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, 3, FILTER1_NUM), mean=MU, stddev=SIGMA)),
    'wc2': tf.Variable(tf.truncated_normal(shape=(5, 5, FILTER1_NUM, FILTER2_NUM), mean=MU, stddev=SIGMA)),
    'wf1': tf.Variable(tf.truncated_normal(shape=(5 * 5 * FILTER2_NUM, FRC1_NUM), mean=MU, stddev=SIGMA)),
    'wf2': tf.Variable(tf.truncated_normal(shape=(FRC1_NUM, FRC2_NUM), mean=MU, stddev=SIGMA)),
    'wf3': tf.Variable(tf.truncated_normal(shape=(FRC2_NUM, CLASS_NUM), mean=MU, stddev=SIGMA))}

biases = {
    'bc1': tf.Variable(tf.truncated_normal(shape=(FILTER1_NUM,), mean=MU, stddev=SIGMA)),
    'bc2': tf.Variable(tf.truncated_normal(shape=(FILTER2_NUM,), mean=MU, stddev=SIGMA)),
    'bf1': tf.Variable(tf.truncated_normal(shape=(FRC1_NUM,), mean=MU, stddev=SIGMA)),
    'bf2': tf.Variable(tf.truncated_normal(shape=(FRC2_NUM,), mean=MU, stddev=SIGMA)),
    'bf3': tf.Variable(tf.truncated_normal(shape=(CLASS_NUM,), mean=MU, stddev=SIGMA))}


def LeNet(x):

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28xFILTER1_NUM.
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc1']
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28xFILTER1_NUM. Output = 14x14xFILTER1_NUM.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. put = 14x14xFILTER1_NUM. Output = 10x10xFILTER2_NUM.
    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='VALID') + biases['bc2']
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10xFILTER2_NUM. Output = 5x5xFILTER2_NUM.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5xFILTER2_NUM. Output = 5x5xFILTER2_NUM.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 5x5xFILTER2_NUM. Output = FRC1_NUM.
    fc1   = tf.matmul(fc0, weights['wf1']) + biases['bf1']
    fc1   = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = FRC1_NUM. Output = FRC2_NUM.
    fc2   = tf.matmul(fc1, weights['wf2']) + biases['bf2']
    fc2   = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = FRC2_NUM. Output = CLASS_NUM.
    logits = tf.matmul(fc2, weights['wf3']) + biases['bf3']

    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, CLASS_NUM)





### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


rate = 0.00001  # Very Slow to train
rate = 0.001  # @ pre learning
rate = 0.0005

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)





# ## Model Evaluation

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



# ## Evaluate the Model
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
#     test_accuracy = evaluate(X_test, y_test)
#     print("Test Accuracy = {:.3f}".format(test_accuracy))

# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
#     validation_accuracy = evaluate(X_valid, y_valid)
#     print("Validation Accuracy = {:.3f}".format(validation_accuracy))



# OK
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
#     print('wc1')
#     print(sess.run(weights['wc1']))  # 5 x 5 x 3 x 6
#     print()
#     print('bc1')
#     print(sess.run(biases['bc1']))  # 6
#     print()



#
# Visualization
#


# テスト画像の中で、予測に失敗した画像を洗い出す
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    # for i in range(0, len(X_test), 10):
    for i in range(0, 1000, 1):
        cls = sess.run(tf.argmax(logits, 1), feed_dict={x: [X_test[i]]})
        if y_test[i] !=  cls[0]:
            print(y_test[i], cls[0])

exit(0)


# 学習画像全体の中で、予測に失敗した画像を洗い出す
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    for i in range(0, len(X_train)):
        cls = sess.run(tf.argmax(logits, 1), feed_dict={x: [X_train[i]]})
        if y_train[i] !=  cls[0]:
            print(y_train[i], cls[0])
        


exit(0)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    weight = sess.run(weights['wc1'])  # 5 x 5 x 3 x 6
    weight = weight.reshape(5, 5, 18)
    for i in range(18):
        # cnv = sess.run(weights['wc1'][:, :, :, i])  
        cnv = weight[:, :, i]
        cnv = np.abs(cnv)
        cnv = cnv.clip(0, 1.0)
        print('cnv ', i)
        print(cnv)
        plt.subplot(3, 6, i + 1)
        plt.title("Cnv_%d" % i)
        plt.axis("off")
        plt.gray()
        plt.imshow(cnv, cmap='gray')
        # plt.imshow(cnv, cmap=None)
        # plt.imshow(cnv.transpose()[i].reshape(5, 5, 3), cmap=None)
    plt.show()



with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    weight = sess.run(weights['wc2'])  #  5x5 6x12
    weight = weight.reshape(5, 5, 72)
    for i in range(72):
        # cnv = sess.run(weights['wc1'][:, :, :, i])  # 5 x 5 x 3 x 6
        cnv = weight[:, :, i]
        # cnv += 0.6
        cnv = np.abs(cnv)
        cnv = cnv.clip(0, 1.0)
        print('cnv %d', i)
        print(cnv)
        plt.subplot(6, 12, i + 1)
        plt.title("Cnv_%d" % i)
        plt.axis("off")
        plt.gray()
        plt.imshow(cnv, cmap='gray')
        # plt.imshow(cnv, cmap=None)
        # plt.imshow(cnv.transpose()[i].reshape(5, 5, 3), cmap=None)
    plt.show()







exit(0)
