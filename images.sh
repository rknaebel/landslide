INNAME=$1
OUTNAME=$2

gdal_translate -b 3 -b 2 -b 1 ${INNAME} ${OUTNAME}_rgb.tif
gdalwarp -co photometric=RGB -co tfw=yes -t_srs EPSG:3857 ${OUTNAME}_rgb.tif ${OUTNAME}_rgb_proj.tif
convert -sigmoidal-contrast 30x15 -depth 8 ${OUTNAME}_rgb_proj.tif ${OUTNAME}.tif

