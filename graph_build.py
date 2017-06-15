# 失敗したサンプルの画像調査

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# clslist = (31, 23, 26, 29, 12)
# no = 0
# fig = plt.figure(figsize=(16, 9))
# fig.suptitle('class21 near images')
# plt.subplots_adjust(left=0.005, right=0.995, top=0.930, bottom=0.005, hspace=0.2, wspace=0.0)
# #
# for i in range(len(X_train_org)):
#     truth = y_train_org[i]
#     if truth in clslist:
#         ax = plt.subplot(15, 25, no + 1)
#         no += 1
#         ax.set_title("%d" % i, fontsize=6)
#         plt.axis("off")
#         plt.tick_params(labelbottom="off")
#         plt.tick_params(labelleft="off")
#         plt.imshow(X_train_org[i].reshape(32, 32, 3), cmap=None)
#         if 375 <= no:
#             print(no, 'images displayed')
#             break
# plt.savefig('fig/class21_mistake_images.png')
# plt.show()
# exit(0)

clslist = (21, )  # (21, 24, 20, 41, 27):
# if tf.train.get_checkpoint_state(netdir):
#     no = 0
#     fig = plt.figure(figsize=(16, 9))
#     fig.suptitle('class%d training images' % clslist[0])
#     plt.subplots_adjust(left=0.005, right=0.995, top=0.930, bottom=0.005, hspace=0.2, wspace=0.0)
#     #
#     for i in range(len(X_train_org)):
#         truth = y_train_org[i]
#         if truth in clslist:
#             ax = plt.subplot(10, 25, no + 1)
#             no += 1
#             # plt.title("%d" % i, fontsize=6)
#             ax.set_title("%d" % i, fontsize=6)
#             plt.axis("off")
#             plt.tick_params(labelbottom="off")
#             plt.tick_params(labelleft="off")
#             plt.imshow(X_train_org[i].reshape(32, 32, 3), cmap=None)
#             if 250 <= no:
#                 print(no, 'images displayed')
#                 break
# plt.savefig('fig/class%d_training_images.png' % clslist[0])  # bbox_inches="tight", pad_inches=0.0)
# plt.show()

if tf.train.get_checkpoint_state(netdir):
    no = 0
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('class%d validation images' % clslist[0])
    plt.subplots_adjust(left=0.005, right=0.995, top=0.970, bottom=0.005, hspace=0.2, wspace=0.0)
    #
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(netdir))

        for i in range(len(X_valid_org)):
            truth = y_valid_org[i]
            if truth in clslist:
                infcls = sess.run(tf.argmax(logits, 1), feed_dict={x: [X_valid_org[i]]})
                if truth !=  infcls[0]:
                    ax = plt.subplot(3, 6, no + 1)
                    no += 1
                    plt.title("%d" % i, fontsize=6)
                    plt.axis("off")
                    plt.tick_params(labelbottom="off")
                    plt.tick_params(labelleft="off")
                    plt.imshow(X_valid_org[i].reshape(32, 32, 3), cmap=None)
                    print(i, truth, infcls[0])
plt.savefig('fig/class%d_failed_valid_images.png' % clslist[0])  # bbox_inches="tight", pad_inches=0.0)
plt.show()
exit(0)



wtile = 14
htile = 8
fig = plt.figure(figsize=(26,14))
no = 0
ckpt = tf.train.get_checkpoint_state(netdir)
if ckpt:  # checkpointがある場合
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(netdir))

        # 検証画像の中で、予測に失敗した画像を洗い出す
        for i in range(len(X_valid_org)):
            # 21, 24, 1, 20, 25, 3, 8 ... 41, 27, 33
            truth = y_valid_org[i]
            if truth in (21, 24, 1, 20, 25, 3, 8):
                infcls = sess.run(tf.argmax(logits, 1), feed_dict={x: [X_valid_org[i]]})
                if truth !=  infcls[0]:
                    ax = plt.subplot(htile, wtile, no + 1)
                    no += 1
                    plt.subplots_adjust(left=0.005, right=0.990, top=0.995, bottom=0.001, hspace=0.0, wspace=0.0)
                    plt.title("%d" % i, fontsize=6)
                    plt.axis("off")
                    plt.tick_params(labelbottom="off")
                    plt.tick_params(labelleft="off")
                    plt.imshow(X_valid_org[i].reshape(32, 32, 3), cmap=None)
                    print(i, truth, infcls[0])
plt.savefig('fig/faled_valid_images.png')  # bbox_inches="tight", pad_inches=0.0)
plt.show()
exit(0)



# 失敗したサンプルの分布調査
gtruth_valid = []
y_fail_valid = []
gtruth_test = []
y_fail_test = []
fail_valid = []
fail_test = []

ckpt = tf.train.get_checkpoint_state(netdir)
if ckpt:  # checkpointがある場合
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(netdir))

        validation_accuracy = evaluate(X_valid, y_valid)
        test_accuracy = evaluate(X_test, y_test)
        train_accuracy = evaluate(X_train, y_train)

        print("Validation Accuracy = {:.5f}".format(validation_accuracy))
        print("Test Accuracy       = {:.5f}".format(test_accuracy))
        print("Training Accuracy   = {:.5f}".format(train_accuracy))

        # 検証画像の中で、予測に失敗した画像を洗い出す
        for i in range(len(X_valid)):
            infcls = sess.run(tf.argmax(logits, 1), feed_dict={x: [X_valid[i]]})
            truth = y_valid[i]
            if truth !=  infcls[0]:
                gtruth_valid.append(truth)
                y_fail_valid.append(infcls[0])
                fail_valid.append(truth)
                print(i, truth, infcls[0])

        # テスト画像の中で、予測に失敗した画像を洗い出す
        for i in range(len(X_test)):
            infcls = sess.run(tf.argmax(logits, 1), feed_dict={x: [X_test[i]]})
            truth = y_test[i]
            if truth !=  infcls[0]:
                gtruth_test.append(truth)
                y_fail_test.append(infcls[0])
                fail_test.append(truth)
                print(i, truth, infcls[0])


fig = plt.figure(figsize=(12,8))
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
plt.title('Histgram of Failed samples')
names=['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
labels=['validation data', 'test data']
plt.hist([fail_valid, fail_test],
         range=(0, 43),
         rwidth=20,
         stacked=False,
         bins=43,
         label=labels
         )
plt.xticks(range(0,43), names, rotation=-90, fontsize="small")
plt.legend()
plt.savefig('fig/fialed_samples.png')
# plt.show()

# gtruth = [25, 9, 2, 5, 2, 5, 7, 22, 21, 26, 7, 19, 18, 41, 27, 25, 38, 2, 24, 35, 25, 8, 18, 18, 18, 14, 40, 7, 30, 26, 8, 5, 24, 2, 7, 31, 21, 42, 21, 38, 14, 18, 7, 11, 6, 8, 1, 3, 0, 8, 9, 35]
# y_fail = [21, 3, 13, 26, 3, 2, 9, 20, 10, 25, 5, 23, 38, 15, 37, 11, 40, 5, 29, 15, 11, 5, 27, 4, 21, 17, 36, 9, 11, 25, 3, 2, 18, 1, 8, 21, 39, 11, 42, 34, 38, 26, 8, 26, 2, 7, 2, 33, 8, 3, 3, 36]
# 52 sample in Validation data failed
# print(len(gtruth))


fig = plt.figure(figsize=(12,8))
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
plt.title('Histgram of Failed Test samples')
names=['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
labels=['fail(truth)', 'fail(inference)']
plt.hist([gtruth_test, y_fail_test],
         range=(0, 43),
         rwidth=20,
         stacked=False,
         bins=43,
         label=labels
         )
plt.xticks(range(0,43), names, rotation=-90, fontsize="small")
plt.legend()
plt.savefig('fig/fialed_test_samples.png')

fig = plt.figure(figsize=(12,8))
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
plt.title('Histgram of Failed Validation samples')
names=['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
labels=['fail(truth)', 'fail(inference)']
plt.hist([gtruth_valid, y_fail_valid],
         range=(0, 43),
         rwidth=20,
         stacked=False,
         bins=43,
         label=labels
         )
plt.xticks(range(0,43), names, rotation=-90, fontsize="small")
plt.legend()
plt.savefig('fig/fialed_validation_samples_2.png')

plt.show()

# x512
for i in range(9):
    gtruth.extend(gtruth)
    y_fail.extend(y_fail)

fig = plt.figure(figsize=(12,8))
plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.45, hspace=0.0, wspace=0.0)
plt.title('Histgram of Train/Valid/Test data')
names=['0:Speed limit (20km/h)', '1:Speed limit (30km/h)', '2:Speed limit (50km/h)', '3:Speed limit (60km/h)', '4:Speed limit (70km/h)', '5:Speed limit (80km/h)', '6:End of speed limit (80km/h)', '7:Speed limit (100km/h)', '8:Speed limit (120km/h)', '9:No passing', '10:No passing for vehicles over 3.5 metric tons', '11:Right-of-way at the next intersection', '12:Priority road', '13:Yield', '14:Stop', '15:No vehicles', '16:Vehicles over 3.5 metric tons prohibited', '17:No entry', '18:General caution', '19:Dangerous curve to the left', '20:Dangerous curve to the right', '21:Double curve', '22:Bumpy road', '23:Slippery road', '24:Road narrows on the right', '25:Road work', '26:Traffic signals', '27:Pedestrians', '28:Children crossing', '29:Bicycles crossing', '30:Beware of ice/snow', '31:Wild animals crossing', '32:End of all speed and passing limits', '33:Turn right ahead', '34:Turn left ahead', '35:Ahead only', '36:Go straight or right', '37:Go straight or left', '38:Keep right', '39:Keep left', '40:Roundabout mandatory', '41:End of no passing', '42:End of no passing by vehicles over 3.5 metric tons']
labels=['y_train', 'y_valid', 'fail(truth)x512', 'fail(inference)x512']
plt.hist([y_train, y_valid, gtruth, y_fail],
         range=(0, 43),
         rwidth=20,
         stacked=False,
         bins=43,
         label=labels
         )
plt.xticks(range(0,43), names, rotation=-90, fontsize="small")
plt.legend()
plt.show()

exit(0)
