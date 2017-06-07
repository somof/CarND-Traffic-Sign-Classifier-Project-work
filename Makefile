
PYTHON = ../miniconda3/envs/carnd-term1/bin/python
PYTHON = ../miniconda3/envs/IntroToTensorFlow/bin/python
PYTHON = ../miniconda3/envs/carnd-term1-gpu/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHON = ../../../src/miniconda3/envs/carnd-term1/bin/python
PYTHON = ../../../src/miniconda3/envs/IntroToTensorFlow/bin/python
endif

all: vis

vis:
	$(PYTHON) Visualize_Weights.py

train:
	$(PYTHON) Traffic_Sign_Classifier.py

data:
	$(PYTHON) Visualize_Dataset.py
