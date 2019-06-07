#!/bin/bash

source env/bin/activate

# Install requirements
module load python/3.6.1+intel-16.0
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Assemble data
gunzip data/Business_Licenses.csv.gz
gunzip data/merged_business_govtdata.csv.gz
python3 code/data-assembly/glue.py

deactivate
