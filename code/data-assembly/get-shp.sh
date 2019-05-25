cd $(git rev-parse --show-toplevel)
cd data

wget 'https://www2.census.gov/geo/tiger/TIGER2017/BG/tl_2017_17_bg.zip'
unzip tl_2017_17_bg.zip

ogr2ogr -f GeoJSON Cook_bg.geojson tl_2017_17_bg.shp -sql "select * from tl_2017_17_bg where COUNTYFP = '031'" -overwrite

rm *.shx *.shp *.dbf *.prj *.xml *.zip *.cpg

cd ..
