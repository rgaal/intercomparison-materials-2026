import xarray as xr
import cartopy as ctpy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import metpy.calc as mpcalc
import scipy.stats as stat
import nctoolkit as nc
import pandas as pd
import dask.array as da
import xcdat
import importlib
import glob
from numpy import *
from metpy.calc import heat_index
from metpy.units import units
from netCDF4 import Dataset
from scipy import stats
from statsmodels.stats.multitest import multipletests
import xclim.indices as xclims
import xskillscore as xss

def annual_climatology(data,label_choice,lat_loc=None,lon_loc=None,region_loc=None,latlon_flag=None):
    ds=data.bounds.add_missing_bounds()
    annual_climo = ds.temporal.climatology(label_choice, freq="month", weighted=True)
    if (region_loc == True):
        if (latlon_flag==True):
            region_data=annual_climo.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))#,method='nearest')
        else:
            region_data=annual_climo[label_choice][:,lon_loc[0]:lon_loc[1],lat_loc[0]:lat_loc[1]]
            
        return region_data
    else:
        return annual_climo

def rmse_calc_heatmap(model_data,truth_data,label_choice,lat_loc,lon_loc,season=False,season_abbrev=None):
    #assumed to be spatial average calculations
    #truth data assumed to be any of the verification datasets
    model_small=model_data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
    truth_small=truth_data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
    
    if (model_small.coords['time'].shape != truth_small.coords['time'].shape):
        print('time dimensions do not match, exiting..')
        return nan
    elif (model_small.coords['lat'].shape != truth_small.coords['lat'].shape) | (model_small.coords['lon'].shape != truth_small.coords['lon'].shape):
        print('spatial dimensions do not match, exiting...')
        return nan
    else:
        rmse_numerator=zeros((model_small[label_choice].shape))
        rmse_denominator=model_small[label_choice].shape[0]*model_small[label_choice].shape[1]*model_small[label_choice].shape[2]
        print('starting rmse calc')
        for i,time in enumerate(model_small.coords['time'].values):
            model_x=model_small[label_choice][i,:,:]
            truth_x=truth_small[label_choice][i,:,:]
            rmse_numerator[i,:,:]=(model_x - truth_x)**2
        print('time iteration complete for rmse calc')
        rmse_i=nansum(rmse_numerator)/rmse_denominator#over all axes
        rmse_f=sqrt(rmse_i)
        print(rmse_f) #should be 1/single number
        return rmse_f

def mae_calc_heatmap(model_data,truth_data,label_choice,lat_loc,lon_loc,season=False,season_abbrev=None):
    #assumed to be spatial average calculations
    #truth data assumed to be any of the verification datasets
    
    model_small=model_data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
    truth_small=truth_data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
    
    if (model_small.coords['time'].shape != truth_small.coords['time'].shape):
        print('time dimensions do not match, exiting..')
        return nan
    elif (model_small.coords['lat'].shape != truth_small.coords['lat'].shape) | (model_small.coords['lon'].shape != truth_small.coords['lon'].shape):
        print('spatial dimensions do not match, exiting...')
        return nan
    else:
        model_small=model_small.assign_coords(time=truth_small['time'].values)
        mae=xss.mae(truth_small[label_choice].load(),model_small[label_choice].load(), dim='time',skipna=True)

    return nanmean(mae)

def mean_bias_calc_singles(model_data,truth_data,label_choice,lat_loc,lon_loc,season=False,season_abbrev=None):
    #assumed to be spatial average calculations
    #truth data assumed to be any of the verification datasets
    if (season==True):
        model_data=model_data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
        truth_data=truth_data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
        
        model_szn=model_data.groupby('time.season')
        model_small=model_szn[season_abbrev]
        truth_szn=truth_data.groupby('time.season')
        truth_small=truth_szn[season_abbrev]
    else:
        model_small=model_data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
        truth_small=truth_data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
        
    if (model_small.coords['time'].shape != truth_small.coords['time'].shape):
        print('time dimensions do not match, exiting..')
        return nan
    elif (model_small.coords['lat'].shape != truth_small.coords['lat'].shape) | (model_small.coords['lon'].shape != truth_small.coords['lon'].shape):
        print('spatial dimensions do not match, exiting...')
        return nan
    else:
        numerator=zeros((model_small[label_choice].shape))
        print('starting rmse calc')
        for i,time in enumerate(model_small.coords['time'].values):
            model_x=model_small[label_choice][i,:,:]
            truth_x=truth_small[label_choice][i,:,:]
            numerator[i,:,:]=(model_x - truth_x)
        print('time iteration complete for rmse calc')
        mb=ravel(numerator)
        print(mb.shape) #should be lat,lon
        return mb

def etccdi_prcptot(data,label_choice,lat_loc,lon_loc,wrf_flag=None,unit_mm=True,subset=None):
    #assumed to be spatial average calculations
    #truth data assumed to be any of the verification datasets
    if (wrf_flag==True):
        try:
            data_small=data.rename({'Times': 'time'})
        except ValueError:
            data_small=data
    else:
        if (subset==True):
            data_small=data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
        else:
            data_small=data

    if (unit_mm==True):
        data_calc=data_small[label_choice].where(data_small[label_choice]>1., nan)
    else:
        data_calc=data_small[label_choice].where(data_small[label_choice]>0.0393701)
        
    return data_calc.groupby('time.year').sum()


def etccdi_sdii(data,label_choice,lat_loc,lon_loc,count_only=None,subset=None,wrf_flag=None,unit_mm=True):
    #for r1mm
    #assumed to be spatial average calculations
    #truth data assumed to be any of the verification datasets
    if (wrf_flag==True):
        try:
            data_small=data.rename({'Times': 'time'})
        except ValueError:
            data_small=data
    else:
        if (subset==True):
            data_small=data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
        else:
            data_small=data

    if (unit_mm==True):
        data_calc=data_small[label_choice].where(data_small[label_choice]>1., nan)
    else:
        data_calc=data_small[label_choice].where(data_small[label_choice]>0.0393701, nan)
    
    datas=data_calc.groupby('time.year').sum()
    datas_nan=datas.where(datas!=0., nan)
    
    pr_mm_count = xr.full_like(datas,1)
    for i,key in enumerate(data_calc.groupby('time.year').groups.keys()):
        # print(i)
        pr_mm_count[i,:,:]=sum(~isnan(data_calc.groupby('time.year')[key]),axis=0)
    pr_mm_valid=pr_mm_count.where(pr_mm_count!=0., nan)

    sdii=datas_nan/pr_mm_valid
    if (count_only==True):
        return pr_mm_valid
    else:
        return sdii.values


def etccdi_rxday(data,label_choice,lat_loc,lon_loc,window=None,season=None,subset=None,wrf_flag=None,unit_mm=True):
    window=int(window)
    #default 1 day window
    months_to_keep = [5,6,7,8,9,10]
    #assumed to be spatial average calculations
    #truth data assumed to be any of the verification datasets
    if (subset==True):
        data_small=data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
    else:
        data_small=data

    if (season==True):
        data_small=data_small.sel(time=data_small.time.dt.month.isin(months_to_keep))
        # keep = isin(data_small.time.dt.month, list(months_to_keep))

    data_small.pr_mm.attrs['units']='mm/day'
    years = data_small.time.dt.year.values
    unique_years = unique(years)

    nlat, nlon = data_small[label_choice].shape[1], data_small[label_choice].shape[2]
    rxday_max = full((unique_years.size, nlat, nlon), nan)
    mask=data_small[label_choice][0,:,:].values
    mask=where(isnan(mask)==True,mask,1.)
    
    for k, yr in enumerate(unique_years):
        print('year:',yr)
        t=data_small.sel(time=str(yr))
        test=xclims.max_n_day_precipitation_amount(t[label_choice],window=window)
        rxday_max[k,:,:]=test*mask
        
    return rxday_max


def etccdi_cwds(data,label_choice,lat_loc,lon_loc,season=None,subset=None,wrf_flag=None,unit_mm=True):
    #for r1mm
    threshold=1.
    months_to_keep = [5,6,7,8,9,10]
    #assumed to be spatial average calculations
    #truth data assumed to be any of the verification datasets
    if (subset==True):
        data_small=data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
    else:
        data_small=data

    if (season==True):
        data_small=data_small.sel(time=data_small.time.dt.month.isin(months_to_keep))
        # keep = isin(data_small.time.dt.month, list(months_to_keep))

    data_small.pr_mm.attrs['units']='mm/day'
    years = data_small.time.dt.year.values
    unique_years = unique(years)

    nlat, nlon = data_small[label_choice].shape[1], data_small[label_choice].shape[2]
    cwd_max = full((unique_years.size, nlat, nlon), nan)
    mask=data_small[label_choice][0,:,:].values
    mask=where(isnan(mask)==True,mask,1.)
    
    for k, yr in enumerate(unique_years):
        print('year:',yr)
        t=data_small.sel(time=str(yr))
        test=xclims.maximum_consecutive_wet_days(t[label_choice])
        cwd_max[k,:,:]=test*mask
        
    return cwd_max

def etccdi_cdds(data,label_choice,lat_loc,lon_loc,season=None,subset=None,wrf_flag=None,unit_mm=True):
    #for r1mm
    threshold=1.
    months_to_keep = [5,6,7,8,9,10]
    #assumed to be spatial average calculations
    #truth data assumed to be any of the verification datasets
    if (subset==True):
        data_small=data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
    else:
        data_small=data

    if (season==True):
        data_small=data_small.sel(time=data_small.time.dt.month.isin(months_to_keep))
        # keep = isin(data_small.time.dt.month, list(months_to_keep))

    data_small.pr_mm.attrs['units']='mm/day'
    years = data_small.time.dt.year.values
    unique_years = unique(years)

    nlat, nlon = data_small[label_choice].shape[1], data_small[label_choice].shape[2]
    cdd_max = full((unique_years.size, nlat, nlon), nan)
    mask=data_small[label_choice][0,:,:].values
    mask=where(isnan(mask)==True,mask,1.)
    
    for k, yr in enumerate(unique_years):
        print('year:',yr)
        t=data_small.sel(time=str(yr))
        test=xclims.maximum_consecutive_dry_days(t[label_choice])
        cdd_max[k,:,:]=test*mask
        
    return cdd_max

def cumulative_precip_anom_curve(data,label_choice,lat_loc,lon_loc,time_ave=None):
    #assumed to be spatial average calculations
    #truth data assumed to be any of the verification datasets
    try:
        ds=data.bounds.add_missing_bounds()
    except ZeroDivisionError:
        ds=data
    pmean_k=ds.temporal.climatology(label_choice, freq="day", weighted=True)
    pmean_k=pmean_k.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
    pmean_k_val=nanmean(pmean_k[label_choice])

    data_small=data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
    # data_small1d=nanmean(data_small,axis=(1,2))
    pr_vals = full((len(data_small.groupby('time.year').groups.keys()),
                    366,data_small[label_choice].shape[1],
                    data_small[label_choice].shape[2]),nan)

    for count,time in enumerate(data_small.groupby('time.year').groups.keys()):
        data_iter=data_small.groupby('time.year')[time]
        print(time)
        for i,j in enumerate(data_iter.coords['time']):
            data_ins=data_iter.sel(time=j)
            pr_vals[count,i,:,:]=data_ins[label_choice]-pmean_k_val

    if (time_ave==False):
        return pr_vals
    else:
        pr_point=nanmean(pr_vals,axis=0)
        pr_ave=nanmean(pr_point,axis=(1,2))
        #yearly average of every point, then average spatially^^
        return pr_ave

def ttest_bimodial(obs,data,lat_arr,lon_arr):
    #this can be used to evaluate ensemble mean t-test
    #or individual t-test for custom approach function
    
    differences=data-obs
    # print(differences.shape)
    p_values = zeros((lat_arr.shape[0], lon_arr.shape[0]))
    
    for i in range(lat_arr.shape[0]):
        for j in range(lon_arr.shape[0]):
            diff = differences[:, i, j]
            # Paired t-test: test if mean difference != 0
            # t_stat, p_val = stat.ttest_rel(model_final[:, i, j], prism_final[:, i, j])
            # t_stat, p_val = stat.ttest_ind(model_final[:, i, j], prism_final[:, i, j])
            t_stat,p_val = stat.ttest_1samp(diff, 0)
            # Or equivalently: stats.ttest_1samp(diff, 0)
            p_values[i, j] = p_val
    
    #Apply FDR correction across all grid points
    # mask nan
    mask = ~isnan(p_values)
    pvals_cleaned = p_values[mask]
    rejected, p_corrected, _, _ = multipletests(pvals_cleaned, alpha=0.1, method='fdr_bh')
    # 'fdr_bh' = Benjamini-Hochberg procedure
    final_pvals = full(p_values.shape, nan)
    final_pvals[mask] = p_corrected

    diff_arr=sign(nanmean(differences,axis=0))
    # print('returning')

    return final_pvals,diff_arr,differences

def bias_calc_heatmap(model_data,truth_data,label_choice,lat_loc,lon_loc):
    #assumed to be spatial average calculations
    #truth data assumed to be any of the verification datasets
    model_small=model_data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
    truth_small=truth_data.sel(lat=slice(lat_loc[0],lat_loc[1]),lon=slice(lon_loc[0],lon_loc[1]))
        
    if (model_small.coords['time'].shape != truth_small.coords['time'].shape):
        print('time dimensions do not match, exiting..')
        return nan
    elif (model_small.coords['lat'].shape != truth_small.coords['lat'].shape) | (model_small.coords['lon'].shape != truth_small.coords['lon'].shape):
        print('spatial dimensions do not match, exiting...')
        return nan
    else:

        model_small=model_small.assign_coords(time=truth_small['time'].values)
        bias_vals=model_small[label_choice]-truth_small[label_choice]
        
        bias=nanmean(bias_vals)#over all axes

        print(bias) #should be 1/single number
        return bias