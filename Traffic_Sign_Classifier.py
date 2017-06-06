# Preprocess Data
from sklearn.utils import shuffle

# Setup TensorFlow
import tensorflow as tf
from tensorflow.contrib.layers import flatten


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

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



# Test
# import numpy as np
# # Pad images with 0s
# X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
# X_valid = np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)), 'constant')
# X_test  = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
# print("Updated Image Shape: {}".format(X_train[0].shape))





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




### Define your architecture here.
### Feel free to use as many code cells as needed.

EPOCHS      = 200
BATCH_SIZE  = 100
FILTER1_NUM =  10 #   6
FILTER2_NUM =  20 #  16
FRC3_NUM    = 100 # 120
FRC4_NUM    =  60 #  84
CLASS_NUM   = 43


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for
    # the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28xFILTER1_NUM.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, FILTER1_NUM), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(FILTER1_NUM))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28xFILTER1_NUM. Output = 14x14xFILTER1_NUM.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. put = 14x14xFILTER1_NUM. Output = 10x10xFILTER2_NUM.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, FILTER1_NUM, FILTER2_NUM), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(FILTER2_NUM))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10xFILTER2_NUM. Output = 5x5xFILTER2_NUM.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Flatten. Input = 5x5xFILTER2_NUM. Output = 5x5xFILTER2_NUM.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 5x5xFILTER2_NUM. Output = FRC3_NUM.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(5 * 5 * FILTER2_NUM, FRC3_NUM), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(FRC3_NUM))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = FRC3_NUM. Output = FRC4_NUM.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(FRC3_NUM, FRC4_NUM), mean=mu, stddev=sigma))
    fc2_b  = tf.Variable(tf.zeros(FRC4_NUM))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = FRC4_NUM. Output = CLASS_NUM.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(FRC4_NUM, CLASS_NUM), mean=mu, stddev=sigma))
    fc3_b  = tf.Variable(tf.zeros(CLASS_NUM))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, CLASS_NUM)





### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


rate = 0.001  # @ pre learning
rate = 0.00001

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


# ## Train the Model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    # test code
    ckpt = tf.train.get_checkpoint_state('./')
    if ckpt:  # checkpointがある場合
        last_model = ckpt.model_checkpoint_path  # 最後に保存したmodelへのパス
        print("load " + last_model)
        saver.restore(sess, last_model)  # 変数データの読み込み
        # test code
    else:
        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluate(X_valid, y_valid)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        saver.save(sess, './lenet')
        print("Model saved")

    # print("Additional Training...")
    # print()
    # for i in range(EPOCHS):
    #     X_train, y_train = shuffle(X_train, y_train)
    #     for offset in range(0, num_examples, BATCH_SIZE):
    #         end = offset + BATCH_SIZE
    #         batch_x, batch_y = X_train[offset:end], y_train[offset:end]
    #         sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
    #     validation_accuracy = evaluate(X_valid, y_valid)
    #     print("EPOCH {} ...".format(i + 1))
    #     print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    #     print()
    # saver.save(sess, './lenet')
    # print("Model saved")

    # Visualization
    # _W = sess.run(logits.conv1_W)
    # for i in range(10):
    #     plt.subplot(2, 5, i+1)
    #     plt.title("W_%d" % i)
    #     plt.axis("off")
    #     plt.imshow(_W.transpose()[i].reshape(32, 32), cmap=None)
    # plt.show()


# ## Evaluate the Model
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    validation_accuracy = evaluate(X_valid, y_valid)
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
