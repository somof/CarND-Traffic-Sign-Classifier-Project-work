
# PYTHONC = ../miniconda3/envs/IntroToTensorFlow/bin/python
PYTHONC = ../miniconda3/envs/carnd-term1/bin/python
PYTHONG = ../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHONC = ../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHONC = ../../../src/miniconda3/envs/IntroToTensorFlow/bin/python
endif

all: vis

vis:
	$(PYTHONG) Traffic_Sign_Classifier_TensorBoard.py
	#$(PYTHONC) Visualize_Weights.py

train:
	$(PYTHONG) Traffic_Sign_Classifier.py

data:
	$(PYTHONC) Visualize_Dataset.py
