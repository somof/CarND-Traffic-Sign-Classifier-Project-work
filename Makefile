
PYTHON = ../miniconda3/envs/carnd-term1/bin/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHON = ../../../src/miniconda3/envs/carnd-term1/bin/python
endif

all:
	$(PYTHON) Traffic_Sign_Classifier.py
