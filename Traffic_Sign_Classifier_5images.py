# import os
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# Large Model
FILTER1_NUM =  64
FILTER2_NUM =  84
FRC1_NUM    = 240
FRC2_NUM    = 240
netdir = 'large_model_type1_RGB_tap5x5_NoBN'

CLASS_NUM   =  43
MU          =   0
SIGMA       = 0.1

def LeNet(x):

    with tf.name_scope('conv1'):
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28xFILTER1_NUM.
        conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, FILTER1_NUM), mean=MU, stddev=SIGMA))
        conv1_b = tf.Variable(tf.truncated_normal(shape=(FILTER1_NUM,), mean=MU, stddev=SIGMA))
        conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)
        # Pooling. Input = 28x28xFILTER1_NUM. Output = 14x14xFILTER1_NUM.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Tensorboard
        conv1_w_hist = tf.summary.histogram("conv1_w", conv1_w)
        conv1_b_hist = tf.summary.histogram("conv1_b", conv1_b)

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

# with tf.name_scope('loss'):
#     with tf.name_scope('cross_entropy'):
#         cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
#     with tf.name_scope('loss_operation'):
#         loss_operation = tf.reduce_mean(cross_entropy)





# input images
import csv
sign_name = [''] * 43
with open('signnames.csv', mode='r') as infile:
    reader = csv.reader(infile)
    next(reader, infile)
    sign_name = [rows[1] for rows in reader]

# imagefiles = ('inputimages/c03_speedlimit60.jpg',
#               'inputimages/c04_speedlimit70.jpg',
#               'inputimages/c11_right_of_way.jpg',
#               'inputimages/c13_yield_1.jpg',
#               'inputimages/c13_yield_2.jpg', 
#               'inputimages/c17_no_entry.jpg',
#               'inputimages/c17_no_entry_2.jpg',
#               'inputimages/c18_caution_1.jpg',
#               'inputimages/c18_caution_2.jpg',
#               'inputimages/c33_turn_right.jpg',
#               'inputimages/c25_road_work.jpg',
#               'inputimages/c40_roundabout.jpg')
# answer = (3, 4, 11, 13, 13, 17, 17, 18, 18, 33, 25, 40)

imagefiles = ('inputimages/c04_speedlimit70.jpg',
              'inputimages/c13_yield_2.jpg',
              'inputimages/c17_no_entry_2.jpg',
              'inputimages/c33_turn_right.jpg',
              'inputimages/c40_roundabout.jpg',
              'inputimages/c17_no_entry.jpg',
              'inputimages/c17_no_entry_2-3.jpg',)
answer = (4, 13, 17, 33, 40, 17, 17)

# for ans, file in zip(answer, imagefiles):
#     img = Image.open(file).resize((128, 128), Image.LANCZOS)
#     plt.imshow(img)
#     plt.show()
# exit(0)



saver = tf.train.Saver()

result = []
with tf.Session() as sess:
    # Initialize & Train
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(netdir))

    for no, ans, file in zip(range(7), answer, imagefiles):
        # load images
        img = Image.open(file).resize((32, 32), Image.LANCZOS)
        img = np.asarray(img)
        img = img.astype(np.float32) / 255.0

        # normalization
        mean = np.mean(img[:, :, :])
        stdv = np.std(img[:, :, :])
        for c in range(3):
            img[:, :, c] = img[:, :, c] - mean
            img[:, :, c] = img[:, :, c] / (stdv * 2.0)

        # inference
        cls = sess.run(tf.argmax(logits, 1), feed_dict={x: [img]})

        # output
        print('|{:d}'.format(no), end='')

        if cls[0] == ans:
            result.append(True)
            print('|O|', end='')
        else:
            result.append(False)
            print('|X|', end='')

        print('<img width=64 src="{:s}"/>|'.format(file), end='')
        print(ans, '|', cls[0], ':', sign_name[cls[0]])

result = np.array(result)
print(result)
print(result[result == True])
print('The accuracy fot the 5 images is ', 100 * len(result[result == True]) / len(result), '%')

top_k_op = tf.nn.top_k(tf.nn.softmax(logits), k=5)

with tf.Session() as sess:
    # Initialize & Train
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(netdir))

    for no, ans, file in zip(range(7), answer, imagefiles):
        # load images
        img = Image.open(file).resize((32, 32), Image.LANCZOS)
        img = np.asarray(img)
        img = img.astype(np.float32) / 255.0

        # normalization
        mean = np.mean(img[:, :, :])
        stdv = np.std(img[:, :, :])
        for c in range(3):
            img[:, :, c] = img[:, :, c] - mean
            img[:, :, c] = img[:, :, c] / (stdv * 2.0)

        # inference
        # cls = sess.run(tf.argmax(logits, 1), feed_dict={x: [img]})
        values, indices = sess.run(top_k_op, feed_dict={x: [img]})

        print('    No     : ', no)
        # print('    image  : ', file)
        print('    answer : ', ans)
        print('    inference:')
        for no, i, v in zip(range(7), indices[0], values[0]):
            print('      ', no, ': class {:2d}'.format(i), ':{:6.2f}% '.format(v * 100), sign_name[i])
        print('    ')

        # prob0 = sess.run(tf.nn.top_k(logits, k=5))
        # print(prob0)
        # prob.append(prob0)


