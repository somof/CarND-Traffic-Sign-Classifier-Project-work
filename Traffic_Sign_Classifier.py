
import matplotlib.pyplot as plt


# Step 0: Load The Data

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file= 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print(len(train['features']))

# Step 1: Dataset Summary & Exploration

# The pickled data is a dictionary with 4 key/value pairs:
# 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
# 'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
# 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the pandas shape method might be useful for calculating some of the summary results.

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = 100

# TODO: Number of validation examples
n_validation = 100

# TODO: Number of testing examples.
n_test = 100

# TODO: What's the shape of an traffic sign image?
image_shape = 10

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 10

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



### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.



### Define your architecture here.
### Feel free to use as many code cells as needed.



### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.



