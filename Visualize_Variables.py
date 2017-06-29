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

    with tf.name_scope('conv1') as scope_conv1:
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


# print(logits)
# # with tf.name_scope(scope_conv1):
# with tf.name_scope('conv1'):
#     conv1 = tf.get_variable('conv1/conv1')
#     print(conv1_w)
# exit(0)


imagefiles = ('inputimages/c04_speedlimit70.jpg', 
              'inputimages/c13_yield_2.jpg', 
              'inputimages/c17_no_entry_2.jpg', 
              'inputimages/c33_turn_right.jpg', 
              'inputimages/c40_roundabout.jpg')
answer = (4, 13, 17, 33, 40)


saver = tf.train.Saver()


def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


with tf.Session() as sess:
    # Initialize & Train
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(netdir))

    # check inseide saver


    print(dir())
    print(locals())






    for no, ans, file in zip(range(5), answer, imagefiles):
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


        sess.run(outputFeatureMap([img], conv1))
        # outputFeatureMap([img], conv1)

    exit(0)

exit(0)

## (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
###1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

#6. Visualize the network's feature maps

# Step 4 (Optional): Visualize the Neural Network's State with Test Images
# This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

# Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the LeNet lab's feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

# For an example of what feature map outputs look like, check out NVIDIA's results in their paper End-to-End Deep Learning for Self-Driving Cars in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.



### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

# def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
#     # Here make sure to preprocess your image_input in a way your network expects
#     # with size, normalization, ect if needed
#     # image_input =
#     # Note: x should be the same name as your network's tensorflow data placeholder variable
#     # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
#     activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
#     featuremaps = activation.shape[3]
#     plt.figure(plt_num, figsize=(15,15))
#     for featuremap in range(featuremaps):
#         plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
#         plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
#         if activation_min != -1 & activation_max != -1:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
#         elif activation_max != -1:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
#         elif activation_min !=-1:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
#         else:
#             plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
