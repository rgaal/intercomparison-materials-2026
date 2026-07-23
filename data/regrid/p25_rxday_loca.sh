#!/bin/sh

path="/bkirtman/rgaal/LOCA_s_east/pr/wetszn"
gridfile="mygrid4loca.txt"

cd $path

for i in $path/*_rx*.nc; do
    base=$(basename "$i")
    echo "$base"

    ncatted -a units,lon,o,c,"degrees_east" ${i}
    ncatted -a units,lat,o,c,"degrees_north" ${i}
	cdo -gencon,$gridfile -setgrid,landmask_loca_25km_preserved.nc ${i} weights_masked.nc
	cdo -remap,$gridfile,weights_masked.nc ${i} placehold/p25_$base

done

echo "DONE"