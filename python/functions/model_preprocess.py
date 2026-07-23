import xcdat as xcdat
import pandas as pd
import xarray as xr
import datetime
import dask.array as da
import metpy.calc as mpcalc
from numpy import *
import nctoolkit as nc
import metpy.calc as mpcalc
from metpy.units import units
import filepaths as find_files
import re
import importlib
import sys

import filepaths as filepaths
import calculations as dr_g_calcs
import dictionary as dicts
importlib.reload(filepaths)
importlib.reload(dr_g_calcs)
importlib.reload(dicts)

def highresmips_open_file(filepath_4_data,varname,yr_i,yr_f):
    #varname = pr
    data=xr.open_mfdataset(filepath_4_data)
    data=data.assign(pr_mm=lambda x: x.pr*86400)
    data.pr_mm.attrs=data.pr.attrs
    data.pr_mm.attrs['units']='mm/day'
    
    data=data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    if (data.time.dtype == 'O'):
        dates_cftime=data.coords['time'].values
        dates_datetime = [datetime.datetime(d.year, d.month, d.day) for d in dates_cftime]
        data.coords['time'] = dates_datetime
    data=data.drop('time_bnds')
    data=data.sel(time=slice(str(yr_i),str(yr_f)))

    return data

def loca_monthly_open_file(filepath_4_data,varname,yr_i,yr_f,monthly_scale=None):
    #varname - pr
    data=xr.open_mfdataset(filepath_4_data)
    data=data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    data=data.sel(time=slice(str(yr_i),str(yr_f)))
    try:
        data=data.assign(pr_mm=lambda x: x.pr_tavg*86400)
    except AttributeError:
        data=data.assign(pr_mm=lambda x: x.pr*86400)
    data.pr_mm.attrs['units']='mm/day'

    if (monthly_scale==True):
        data=data.resample(time='MS').mean()
        data.attrs['resample']='monthly mean'

    return data

def era5_open_file(filepath_4_data,varname,yr_i,yr_f,masked=None,monthly_scale=None,stat_opt=None):
    #varname - pr
    data=xr.open_mfdataset(filepath_4_data)
    data=data.rename({'valid_time':'time'})
    
    data=data.sel(time=slice(str(yr_i),str(yr_f)))
    data=data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    data=data.sortby("lat")
    data=data.sel(lon=slice(-90.,-75.),lat=slice(24.,40.))
   
    data=data.assign(pr_mm=lambda x: x.tp*1000)
    data.pr_mm.attrs['units']='mm/day'
    
    if (monthly_scale==True):
        data=data.resample(time='MS').mean()
        data.attrs['resample']='monthly mean'
        
    if (masked==True):
        mask=xr.open_dataset('/bkirtman/rgaal/0.25deg_landmask.nc')
        mask=mask.assign_coords(lon=(((mask.lon + 180) % 360) - 180))
        mask=mask.sel(lon=slice(-90.,-75.),lat=slice(24.,40.))
        # mask['lsm'][:,:,:]=where(mask['lsm']==0.,nan,1.)
        return data*mask['hurs'][0,:,:].values
    else:
        return data
    
def gddp_open_file(filepath_4_data,varname,yr_i,yr_f,monthly_scale=None):
    #varname - pr
    data=xr.open_mfdataset(filepath_4_data)
    data=data.assign(pr_mm=lambda x: x.pr*86400)
    data.pr_mm.attrs=data.pr.attrs
    data.pr_mm.attrs['units']='mm/day'
            
    if (data.time.dtype == 'O'):
        dates_cftime=data.coords['time'].values
        dates_datetime = [datetime.datetime(d.year, d.month, d.day) for d in dates_cftime]
        data.coords['time'] = dates_datetime
        
    if (monthly_scale==True):
        data=data.resample(time='MS').mean()
        
    data=data.sel(time=slice(str(yr_i),str(yr_f)))

    return data

def mesaclip_ihesp_open_file(datapath,varname,yr_i,yr_f,ens_number,timeframe,monthly_scale=None,masked=True):
    mask=xr.open_dataset('/bkirtman/rgaal/0.25deg_landmask.nc')
    mask=mask.assign_coords(lon=(((mask.lon + 180) % 360) - 180))
    full_time_index = pd.date_range(start=str(yr_i)+'-01-01', end=str(yr_f)+'-12-31', freq='D')
    
    print('retrieving ensemble member:,'+ens_number)

    data=xr.open_mfdataset(datapath)
    data=data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    data=data.sortby("lon")
    data=data.sel(lon=slice(-90.,-75.),lat=slice(24.,40.))
    # print(data)
    if (data.time.dtype == 'O'):
        dates_cftime=data.coords['time'].values
        dates_datetime = [datetime.datetime(d.year, d.month, d.day) for d in dates_cftime]
        data.coords['time'] = dates_datetime
        
    data=data.drop_duplicates('time')
    data=data.reindex(time=full_time_index, fill_value=nan)

    data=data.sel(time=slice(str(yr_i),str(yr_f)))
    data.attrs['ensemble_member_number']=ens_number
    data=data.assign(pr_mm=lambda x: x.PRECT*8.64e+7) #m/s to mm/day
    data.pr_mm.attrs=data.PRECT.attrs
    data.pr_mm.attrs['units']='mm/day'
        
    data=data.sel(lat=slice(mask['lat'][0],mask['lat'][-1]),lon=slice(mask['lon'][0],mask['lon'][-1]))
    
    if (monthly_scale==True):
        data=data.resample(time='MS').mean()
    else:
        pass
    
    if (masked==True):
        return data*mask['hurs'][0,:,:]
    else:
        return data

def chc_products_open_file(filepath_4_data,varname,yr_i,yr_f,masked=None,monthly_scale=None):
    #precip units already in mm/day
    data=xr.open_mfdataset(filepath_4_data)
    data=data.sel(time=slice(str(yr_i),str(yr_f)))
    data=data.rename({'PRECT': 'pr_mm'})
    if (monthly_scale==True):
        data=data.resample(time='MS').mean()
    
    if (masked==True):
        mask=xr.open_dataset('/bkirtman/rgaal/0.25deg_landmask.nc')
        mask=mask.assign_coords(lon=(((mask.lon + 180) % 360) - 180))
        datas=data.sel(lon=slice(mask['lon'][0],mask['lon'][-1]),lat=slice(mask['lat'][0],mask['lat'][-1]))
        datas=datas.drop(['lat_bnds','lon_bnds','gw','area'])
        return datas*mask['hurs'][0,:,:].values
        # return data,mask
    else:
        return data


def bccaq_open_file(filepath_4_data,varname,yr_i,yr_f,monthly_scale=None):
    #precip units already in mm/day
    data=xr.open_mfdataset(filepath_4_data)
    data=data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    data=data.sel(time=slice(str(yr_i),str(yr_f)))
    data=data.rename({'pr': 'pr_mm'})
    if (monthly_scale==True):
        data=data.resample(time='MS').mean()
    
    return data


def prism_original(filepath_4_data,yr_i=None,yr_f=None,timeskip=None,monthly_scale=None):
    #precip units already in mm/day
    if (timeskip==False):
        if (yr_f==None) & (yr_i!=None):
            data=xr.open_mfdataset(filepath_4_data)
            data=data.sel(time=str(yr_i))
        elif (yr_i==None):
            data=xr.open_mfdataset(filepath_4_data)
        else:
            data=xr.open_mfdataset(filepath_4_data)
            data=data.sel(time=slice(str(yr_i),str(yr_f)))
    else:
        data=xr.open_dataset(filepath_4_data)
        
    try:
        data=data.rename({'Band1': 'pr_mm'})
    except ValueError:
        pass
    
    if (monthly_scale==True):
        data=data.resample(time='MS').mean()

    data=data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
    
    return data