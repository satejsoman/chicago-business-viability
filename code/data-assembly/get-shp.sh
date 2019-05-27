cd $(git rev-parse --show-toplevel)
cd data

wget 'https://www2.census.gov/geo/tiger/TIGER2017/BG/tl_2017_17_bg.zip'
wget 'https://www2.census.gov/geo/tiger/TIGER2009/17_ILLINOIS/17031_Cook_County/tl_2009_17031_bg00.zip'
unzip tl_2017_17_bg.zip
unzip tl_2009_17031_bg00.zip

ogr2ogr -f GeoJSON Cook_bg_2010.geojson tl_2017_17_bg.shp -sql "select * from tl_2017_17_bg where COUNTYFP = '031'" -overwrite
ogr2ogr -f GeoJSON Cook_bg_2000.geojson tl_2009_17031_bg00.shp

rm *.shx *.shp *.dbf *.prj *.xml *.zip *.cpg

cd ..
