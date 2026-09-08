WORKING_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
SCRIPTS_DIR := $(WORKING_DIR)scripts
DATA_DIR := $(WORKING_DIR)data
RESULTS_DIR := $(WORKING_DIR)results


analysis:
	run-pfas-analysis -I $(DATA_DIR)/input/ -O $(RESULTS_DIR)/ 

analysis_alternative:
	python $(SCRIPTS_DIR)/run_analysis.py -I $(DATA_DIR)/input/ -O $(RESULTS_DIR)/ 

format:
	black -t py314 .
	isort .

flake:
	flake8

clean:
	python $(SCRIPTS_DIR)/clean.py

# Needs 'cffconvert' to be installed via pip
check_cff:
	cffconvert --validate 
