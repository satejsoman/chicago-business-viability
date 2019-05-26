### Government data sources

We use demographic and economic data from three government sources. These indicators are available at the Census tract, the county, and the metropolitan statistical area level. We assign businesses to Census tracts based on their latitude and longitude, then assign data values based on that tract assignment.

#### Census
- at the tract level, we have 2005-2009, 2010-2014, 2013-2017 from ACS 5 year
    * what to do about overlapping years: can either take 2013 from 2010-14 and 2014 from 2013-17 or average for those two years
- we use 2000 decennial for the earliest years: still need to do this
- specific variables
    * median household income - B19013_001
    * population density - B01001_001 / land area
    * share with a BA+ among adults aged 25+ - table s1501, need to get 2005-09 data
    * share aged 35-64 [^1] - S0101? 9-14/1, need to get 2005-09 data

#### BLS
- unemployment at the county level, 2001-2018

#### BEA
- real output (2009 dollars) at the MSA level, 2001-2017

#### TO DO:
- pull 2005-09 data for educational attainment and share aged 35-64
- pull 2000 census data
- geocode businesses to tracts
- think about how to merge things in



[^1]: This group has higher expenditures: https://www.bls.gov/opub/btn/volume-4/consumer-expenditures-vary-by-age.htm