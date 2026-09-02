WORKING_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
SCRIPTS_DIR := $(WORKING_DIR)scripts
DATA_DIR := $(WORKING_DIR)data
RESULTS_DIR := $(WORKING_DIR)results

preprocess:
	python $(SCRIPTS_DIR)/run_preprocess.py -I $(DATA_DIR)/input/ -O $(RESULTS_DIR)/ 

format:
	black -t py314 .
	isort .

flake:
	flake8
