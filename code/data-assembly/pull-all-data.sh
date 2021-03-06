

# detail table pulls: B01001_001, B19013_001
python pull-acs.py --survey acs5 --outfile '../../data/Cook_tract_detail_2013-17.csv' 
python query-census-api.py --survey acs5 --outfile '../../data/Cook_tract_detail_2010-14.csv' --year 2014
python query-census-api.py --survey acs5 --outfile '../../data/Cook_tract_detail_2005-09.csv' --year 2009

# subject table for age and educational attainment: 
# S0101_C01_001-S0101_C01_015 (2013-17), S0101_C01_001-S0101_C01_015 (2010-14), S1501_C_02_015
python query-census-api.py --survey acs5 --outfile '../../data/Cook_tract_subj_2013-17.csv' --tabletype subject
python query-census-api.py --survey acs5 --outfile '../../data/Cook_tract_subj_2010-14.csv' --tabletype subject --year 2014

python extract-0509-acs.py

# unemployment
# python pull-unemployment.py --outfile "../../data/Cook_annual_unemployment.csv"

# output
# python pull-gross-output.py --outfile "../../data/chicago_rgdp_2001-17.csv"