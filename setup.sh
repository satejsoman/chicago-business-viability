#!/bin/bash

# Make and load environment
echo "Making environment"
module load python/3.6.1+intel-16.0
python3 -mvenv env
source env/bin/activate

# Install requirements
echo "Installing requirements"
module load python/3.6.1+intel-16.0
pip3 install --upgrade pip
pip3 install -r code/requirements.txt

# Assemble data
echo "Assembling data"
gunzip data/Business_Licenses.csv.gz
gunzip data/merged_business_govtdata.csv.gz
python3 code/data-assembly/glue.py

deactivate
