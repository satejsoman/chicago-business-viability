cd $(git rev-parse --show-toplevel)
cd data

wget 'https://www2.census.gov/geo/tiger/TIGER2017/TRACT/tl_2017_17_tract.zip'
# wget 'https://www2.census.gov/geo/tiger/TIGER2009/17_ILLINOIS/17031_Cook_County/tl_2009_17031_tract00.zip'
unzip tl_2017_17_tract.zip
# unzip tl_2009_17031_tract00.zip

ogr2ogr -f GeoJSON Cook_tract_2010.geojson tl_2017_17_tract.shp -sql "select * from tl_2017_17_tract where COUNTYFP = '031'" -overwrite
# ogr2ogr -f GeoJSON Cook_tract_2000.geojson tl_2009_17031_tract00.shp

rm *.shx *.shp *.dbf *.prj *.xml *.zip *.cpg

cd ..
