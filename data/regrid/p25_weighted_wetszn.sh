#!/bin/sh

path="/bkirtman/rgaal/prism/wetszn"
gridfile="mygrid4prism.txt"

cd $path

for i in $path/*wetszn.nc; do
    base=$(basename "$i")
    echo "$base"

    ncatted -a units,lon,o,c,"degrees_east" ${i}
    ncatted -a units,lat,o,c,"degrees_north" ${i}
	cdo -gencon,$gridfile -setgrid,landmask_25km_preserved.nc ${i} weights_masked.nc
	cdo -remap,$gridfile,weights_masked.nc ${i} p25_final_$base

done

echo "DONE"
