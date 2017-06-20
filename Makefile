
# PYTHONC = ../miniconda3/envs/IntroToTensorFlow/bin/python
PYTHONC = ../miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../src/miniconda3/envs/IntroToTensorFlow/bin/python
PYTHONC = ../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../../../src/miniconda3/envs/carnd-term1/bin/python
endif

all: vis

vis:
	$(PYTHONC) Visualize_Dataset_distribution_labels.py
	#$(PYTHONC) Visualize_Dataset_Images.py

weight:
	$(PYTHONC) Visualize_Weights.py

tb:
	$(PYTHONG) Traffic_Sign_Classifier_TensorBoard.py

train:
	$(PYTHONG) Traffic_Sign_Classifier.py

train7:
	$(PYTHONG) Traffic_Sign_Classifier_7.py

data:
	#$(PYTHONC) Visualize_Dataset_Inferenced.py
	$(PYTHONC) Visualize_Dataset_Images.py
