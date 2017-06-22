# Preprocess Data
from sklearn.utils import shuffle

# Setup TensorFlow
import tensorflow as tf
from tensorflow.contrib.layers import flatten

import numpy as np


# Step 0: Load The Data

# Load pickled data
import pickle

training_file   = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
# testing_file    = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
# with open(testing_file, mode='rb') as f:
#     test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
# X_test, y_test = test['features'], test['labels']

X_valid_org = X_valid

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

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

X_train, y_train = shuffle(X_train, y_train)

X_train = X_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
# X_test = X_test.astype(np.float32)
nsigma = 2.0

for i in range(len(X_train)):
    # mean = np.mean(X_train[i, :, :, :])
    # stdv = np.std(X_train[i, :, :, :])
    for c in range(3):
        mean = np.mean(X_train[i, :, :, c])
        stdv = np.std(X_train[i, :, :, c])
        X_train[i, :, :, c] = X_train[i, :, :, c] - mean
        X_train[i, :, :, c] = X_train[i, :, :, c] / (stdv * nsigma)

for i in range(len(X_valid)):
    # mean = np.mean(X_valid[i, :, :, :])
    # stdv = np.std(X_valid[i, :, :, :])
    for c in range(3):
        mean = np.mean(X_valid[i, :, :, c])
        stdv = np.std(X_valid[i, :, :, c])
        X_valid[i, :, :, c] = X_valid[i, :, :, c] - mean
        X_valid[i, :, :, c] = X_valid[i, :, :, c] / (stdv * nsigma)

# for i in range(len(X_test)):
#     # mean = np.mean(X_test[i, :, :, :])
#     # stdv = np.std(X_test[i, :, :, :])
#     for c in range(3):
#         mean = np.mean(X_test[i, :, :, c])
#         stdv = np.std(X_test[i, :, :, c])
#         X_test[i, :, :, c] = X_test[i, :, :, c] - mean
#         X_test[i, :, :, c] = X_test[i, :, :, c] / (stdv * nsigma)

# X_train = X_train.clip(-1.0, 1.0)
# X_valid = X_valid.clip(-1.0, 1.0)
# X_test = X_test.clip(-1.0, 1.0)
# No   limitation Validation Accuracy = 0.972 @ epoch 269
# With Limitation Validation Accuracy = 0.971 @ epoch 224
### x_train = X_train + 1.0
### X_valid = X_valid + 1.0
### X_test = X_test + 1.0



### Define your architecture here.
### Feel free to use as many code cells as needed.

# LeNet-Lesson Model
FILTER1_NUM =   6
FILTER2_NUM =  16
FRC1_NUM    = 120
FRC2_NUM    =  84
netdir = 'lenet-small'

# Middle-Size Model
FILTER1_NUM =  16
FILTER2_NUM =  48
FRC1_NUM    = 100
FRC2_NUM    = 100
netdir = 'lenet-middle'
last_validation_accuracy = 0.95125

# Large Model
# FILTER1_NUM =  64
# FILTER2_NUM =  84
# FRC1_NUM    = 240
# FRC2_NUM    = 240
# netdir = 'lenet-large'
# last_validation_accuracy = 0.98163

# netdir = 'dummy-to-renew'

CLASS_NUM   =  43
MU          =   0
SIGMA       = 0.1


def batch_normalization(x, decay=0.9, eps=1e-5):
    shape = x.get_shape().as_list()
    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

    return tf.nn.batch_normalization(x, batch_mean, batch_var, None, None, eps)


def LeNet(x):

    with tf.name_scope('conv1'):
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28xFILTER1_NUM.
        conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, FILTER1_NUM), mean=MU, stddev=SIGMA))
        # conv1_b = tf.Variable(tf.truncated_normal(shape=(FILTER1_NUM,), mean=MU, stddev=SIGMA))
        # conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        # batch Normalization
        conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID')
        conv1 = batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
        # Pooling. Input = 28x28xFILTER1_NUM. Output = 14x14xFILTER1_NUM.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Tensorboard
        conv1_w_hist = tf.summary.histogram("conv1_w", conv1_w)
        # conv1_b_hist = tf.summary.histogram("conv1_b", conv1_b)

    with tf.name_scope('conv2'):
        # Layer 2: Convolutional. put = 14x14xFILTER1_NUM. Output = 10x10xFILTER2_NUM.
        conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, FILTER1_NUM, FILTER2_NUM), mean=MU, stddev=SIGMA))
        conv2_b = tf.Variable(tf.truncated_normal(shape=(FILTER2_NUM,), mean=MU, stddev=SIGMA))
        conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)
        # Pooling. Input = 10x10xFILTER2_NUM. Output = 5x5xFILTER2_NUM.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Tensorboard
        conv2_w_hist = tf.summary.histogram("conv2_w", conv2_w)
        conv2_b_hist = tf.summary.histogram("conv2_b", conv2_b)

    # Flatten. Input = 5x5xFILTER2_NUM. Output = 5x5xFILTER2_NUM.
    fc0   = flatten(conv2)

    with tf.name_scope('fc1'):
        # Layer 3: Fully Connected. Input = 5x5xFILTER2_NUM. Output = FRC1_NUM.
        fc1_w = tf.Variable(tf.truncated_normal(shape=(5 * 5 * FILTER2_NUM, FRC1_NUM), mean=MU, stddev=SIGMA))
        fc1_b = tf.Variable(tf.truncated_normal(shape=(FRC1_NUM,), mean=MU, stddev=SIGMA))
        fc1   = tf.matmul(fc0, fc1_w) + fc1_b
        fc1   = tf.nn.relu(fc1)
        fc1   = tf.nn.dropout(fc1, 0.5)
        # Tensorboard
        fc1_w_hist = tf.summary.histogram("fc1_w", fc1_w)
        fc1_b_hist = tf.summary.histogram("fc1_b", fc1_b)

    with tf.name_scope('fc2'):
        # Layer 4: Fully Connected. Input = FRC1_NUM. Output = FRC2_NUM.
        fc2_w = tf.Variable(tf.truncated_normal(shape=(FRC1_NUM, FRC2_NUM), mean=MU, stddev=SIGMA))
        fc2_b = tf.Variable(tf.truncated_normal(shape=(FRC2_NUM,), mean=MU, stddev=SIGMA))
        fc2   = tf.matmul(fc1, fc2_w) + fc2_b
        fc2   = tf.nn.relu(fc2)
        fc2   = tf.nn.dropout(fc2, 0.5)
        # Tensorboard
        fc2_w_hist = tf.summary.histogram("fc2_w", fc2_w)
        fc2_b_hist = tf.summary.histogram("fc2_b", fc2_b)

    with tf.name_scope('output'):
        # Layer 5: Fully Connected. Input = FRC2_NUM. Output = CLASS_NUM.
        fc3_w  = tf.Variable(tf.truncated_normal(shape=(FRC2_NUM, CLASS_NUM), mean=MU, stddev=SIGMA))
        fc3_b  = tf.Variable(tf.truncated_normal(shape=(CLASS_NUM,), mean=MU, stddev=SIGMA))
        logits = tf.matmul(fc2, fc3_w) + fc3_b
        # Tensorboard
        fc3_w_hist = tf.summary.histogram("fc3_w", fc3_w)
        fc3_b_hist = tf.summary.histogram("fc3_b", fc3_b)

    return logits


# input & output unit
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))

with tf.name_scope('one_hot'):
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, CLASS_NUM)
logits = LeNet(x)

with tf.name_scope('loss'):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    with tf.name_scope('loss_operation'):
        loss_operation = tf.reduce_mean(cross_entropy)



### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

EPOCHS      = 100
BATCH_SIZE  = 100

rate = 0.0010  # good for pre learning
rate = 0.0002  # Slow to train
rate = 0.0005  # Good performance


with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    with tf.name_scope('accuracy'):
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Model Evaluation

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ## Train the Model

saver = tf.train.Saver()

with tf.Session() as sess:

    # Initialize & Train
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    # test code
    ckpt = tf.train.get_checkpoint_state(netdir)
    if ckpt:  # checkpointがある場合
        last_model = ckpt.model_checkpoint_path  # 最後に保存したmodelへのパス
        print("load " + last_model)
        saver.restore(sess, last_model)  # 変数データの読み込み

    # validation_accuracy = evaluate(X_valid, y_valid)
    # print("Validation Accuracy = {:.5f}".format(validation_accuracy))
    # print("Validation Accuracy = {:.5f}".format(sess.run(accuracy_operation, feed_dict={x: X_valid, y: y_valid})))
    # print("Training Accuracy   = {:.5f}".format(evaluate(X_train, y_train)))
    # print()
    print("Validation Accuracy = {:.5f}".format(sess.run(accuracy_operation, feed_dict={x: X_valid, y: y_valid})))
    correct_prediction = sess.run(correct_prediction, feed_dict={x: X_valid, y: y_valid})
    infer_failed = y_valid[np.logical_not(correct_prediction)]
    infer_failed10 = infer_failed
    for i in range(2, 10):
        infer_failed10 = np.r_[infer_failed10, infer_failed]

    # print(correct_prediction)
    # print(y_valid[correct_prediction])
    print(infer_failed)
    # print(infer_failed.dtype)
    # print(infer_failed10)

    infer_failed_hist = np.histogram(infer_failed, bins=42, range=(0, 42))
    print(infer_failed_hist)
    # array([14,  5,  7,  4,  3, 11,  3,  3,  9,  3,
    #         0,  2,  0,  1,  3,  1, 26,  5,  3,  0,
    #        19, 20,  7,  8, 18, 17,  2, 11,  0,  8,
    #         2,  3,  0,  5,  2,  0,  0,  6,  2,  0,
    #         7,  8]),
    #     array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
    #     11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,
    #     22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,  30.,  31.,  32.,
    #     33.,  34.,  35.,  36.,  37.,  38.,  39.,  40.,  41.,  42.]))
    # exit(0)


    import matplotlib.pyplot as plt

    # print(correct_prediction)
    # print(infer_failed)
    # print(infer_failed_hist)
    # exit(0)


    # 作業中 → やめる
    # 間違えられたクラスの頻度を見る

    # fig = plt.figure(figsize=(12,8))
    # plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
    # plt.title('Histgram of Failed samples and training data')
    # names=['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
    # labels=['inference failed data x 10', 'validation data']
    # plt.hist(infer_failed,
    #          range=(0, 43),
    #          rwidth=20,
    #          stacked=False,
    #          bins=43,
    #          label=labels
    #          )
    # plt.xticks(range(0,43), names, rotation=-90, fontsize="small")
    # plt.legend(loc="best")
    # plt.savefig('fig/histgram_misunderstood_labels.png')
    # plt.show()
    # exit(0)

    fig = plt.figure(figsize=(12,8))
    plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
    plt.title('Histgram of Failed samples and training data')
    names=['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
    labels=['inference failed data x 10', 'validation data']
    plt.hist(infer_failed,
             range=(0, 43),
             rwidth=20,
             stacked=False,
             bins=43,
             label=labels
             )
    plt.xticks(range(0,43), names, rotation=-90, fontsize="small")
    plt.legend(loc="best")
    plt.savefig('fig/histgram_failed_samples.png')
    # plt.show()

    fig = plt.figure(figsize=(12,8))
    plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
    plt.title('Histgram of Failed samples and training data')
    names=['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
    labels=['inference failed data x 10', 'validation data']
    plt.hist([infer_failed10, y_train],
             range=(0, 43),
             rwidth=20,
             stacked=False,
             bins=43,
             label=labels
             )
    plt.xticks(range(0,43), names, rotation=-90, fontsize="small")
    plt.legend(loc="best")
    plt.savefig('fig/histgram_failed_samples_and_traingdata.png')
    # plt.show()




    images = X_valid_org
    labels = y_valid
    title = 'X_valid'

    clslist = (16, 21, 20, 24, 25, 0, 5, 27, 8, 29, 23, 22)
    clslist = range(43)

    for cls in clslist:
        wtile = 16
        htile = 6
        limit = wtile * htile
        no = 0
        fig = plt.figure(figsize=(26,14))
        plt.subplots_adjust(left=0.005, right=0.990, top=0.910, bottom=0.001, hspace=0.0, wspace=0.0)
        fig.suptitle('class%02d: infered samples' % cls, fontsize=96)
        for i in range(len(images)):
            if labels[i] == cls and correct_prediction[i] and no < limit:
                ax = plt.subplot(htile, wtile, no + 1)
                plt.title("%d" % no, fontsize=6)
                plt.axis("off")
                plt.tick_params(labelbottom="off")
                plt.tick_params(labelleft="off")
                plt.imshow(images[i].reshape(32, 32, 3), cmap=None)
                no += 1
        plt.savefig('fig/class%02d_images_valid_infered.png' % cls)
        # plt.show()
        # 
        no = 0
        fig = plt.figure(figsize=(26,14))
        plt.subplots_adjust(left=0.005, right=0.990, top=0.910, bottom=0.001, hspace=0.0, wspace=0.0)
        fig.suptitle('class%02d: failed samples' % cls, fontsize=96)
        for i in range(len(images)):
            if labels[i] == cls and not correct_prediction[i] and no < limit:
                ax = plt.subplot(htile, wtile, no + 1)
                plt.title("%d" % no, fontsize=6)
                plt.axis("off")
                plt.tick_params(labelbottom="off")
                plt.tick_params(labelleft="off")
                plt.imshow(images[i].reshape(32, 32, 3), cmap=None)
                no += 1
        plt.savefig('fig/class%02d_images_valid_failed.png' % cls)
        # plt.show()

    # clslist = (16, 21, 20, 24, 25, 0, 5, 27, 8, 29, 23, 22)
    # clslist = (22, )

    # images = X_valid_org
    # labels = y_valid
    # title = 'X_valid'

    # fig = plt.figure(figsize=(26,14))
    # wtile = 16
    # htile = 6
    # limit = wtile * htile
    # no = 0
    # for i in range(len(images)):
    #     if labels[i] in clslist and not correct_prediction[i] and no < limit:
    #         ax = plt.subplot(htile, wtile, no + 1)
    #         plt.subplots_adjust(left=0.005, right=0.990, top=0.995, bottom=0.001, hspace=0.0, wspace=0.0)
    #         plt.title("%d" % no, fontsize=6)
    #         plt.axis("off")
    #         plt.tick_params(labelbottom="off")
    #         plt.tick_params(labelleft="off")
    #         plt.imshow(images[i].reshape(32, 32, 3), cmap=None)
    #         no += 1
    # plt.savefig('fig/class%02d_images_valid_failed.png' % clslist[0])
    # plt.show()

    # exit(0)
