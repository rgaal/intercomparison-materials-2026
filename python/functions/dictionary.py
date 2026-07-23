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
import re
import importlib

import filepaths as filepaths
import model_preprocess as model_prep
import calculations as dr_g_calcs

importlib.reload(filepaths)
importlib.reload(model_prep)
importlib.reload(dr_g_calcs)

cmip6_model_array = ['CAM-MPAS-HR','CMCC-CM2-VHR4','FGOALS-f3-H','ECMWF-IFS-HR','HiRAM-SIT-HR','MRI-AGCM3-2-S']
cmip6_informal_array=['cam-mpas','cmcc','fgoals','ecmwf','hiram','mri']

downscale_model_array = ['CESM2-LENS','CNRM-CM6-1','HadGEM3-GC31-MM','HadGEM3-GC31-LL','IPSL-CM6A-LR','MIROC6','MPI-ESM1-2-HR']
downscale_informal_array = ['cesm2','cnrm','hadgem3-mm','hadgem3-ll','ipsl','miroc6','mpi']

timeframe_array=['Annual','DJF','MAM','JJA','SON']
season=['DJF','MAM','JJA','SON']

so_fl_ll=[25.,-82.]
so_fl_ur=[27.,-80.]
locations=[so_fl_ll,so_fl_ur]
lat_arr=array(locations)[:,0]
lon_arr=array(locations)[:,1]

def agg_dicts_scatter(original_dict,column_names):
    
    agg_dict=pd.DataFrame(index=original_dict.index,columns=['season','obs_data','model_data'])
   
    for i,j in enumerate(column_names):
        print(j)
        if (i==0):
            agg_dict.loc[:,'season']=original_dict['season']
            agg_dict.loc[:,'obs_data']=original_dict['obs_data']
            agg_dict.loc[:,'model_data']=original_dict[j]
        else:
            df_to_add=pd.DataFrame(index=original_dict.index,columns=['season','obs_data','model_data'])
            df_to_add.loc[:,'season']=original_dict['season']
            df_to_add.loc[:,'obs_data']=original_dict['obs_data']
            df_to_add.loc[:,'model_data']=original_dict[j]
            agg_dict=pd.concat([agg_dict, df_to_add],ignore_index=True)
            
    return agg_dict

def initial_dict_scatter(data,var_choice,label_choice,timeframe,yr_i,yr_f):
    
    counts=ravel((data.groupby('time.month')[1][label_choice].values)).shape[0]
    mydict=pd.DataFrame(index=arange(0,counts*12,1),columns=['month','obs_data'])
    
    samples=arange(0,(counts*12)+counts,counts)
    for i,j in enumerate(data.groupby('time.month')):
        mydict.loc[samples[i]:samples[i+1]-1,'obs_data']=ravel(j[1][label_choice].values)
        mydict.loc[samples[i]:samples[i+1]-1,'month']=j[0]
        print('inserted data for group/month: ',j[0])

    mydict['season']='hold'
    for i,j in enumerate(mydict['month'].values):
        if (j<=4) or (j>=11):
            mydict.loc[i,'season']='Dry Season'
        else:
            mydict.loc[i,'season']='Wet Season'

    mydict['month'] = mydict['month'].astype(str)

    return mydict,samples
    
def coupled_highresmip_dict_scatter(lists,dict_name,samples,var_choice,label_choice,timeframe,yr_i,yr_f,model_name):
    idx = cmip6_informal_array.index(model_name)
    model_label=cmip6_model_array[idx]
    lists.append(model_label+' - Coupled')

    filepath_4_data=find_files.highresmip_coupled_filepath(model=model_name,varname=var_choice,masked=True,timeframe=timeframe,monthly_scale=True)
    data=model_prep.highresmips_open_file(filepath_4_data,varname=var_choice,yr_i=yr_i,yr_f=yr_f)
    data=data.sel(lat=slice(lat_arr[0],lat_arr[1]),lon=slice(lon_arr[0],lon_arr[1]))

    dict_name[model_label+' - Coupled']=nan
    print('created column: ',model_name)
    for i,j in enumerate(data.groupby('time.month')):
        dict_name.loc[samples[i]:samples[i+1]-1,model_label+' - Coupled']=ravel(j[1][label_choice].values)
        print('inserted data for group/month: ',j[0])
        
    print('done with dictionary key: ',model_name)

def uncoupled_highresmip_dict_scatter(lists,dict_name,samples,var_choice,label_choice,timeframe,yr_i,yr_f,model_name):
    idx = cmip6_informal_array.index(model_name)
    model_label=cmip6_model_array[idx]
    lists.append(model_label+' - Uncoupled')

    filepath_4_data=find_files.highresmip_uncoupled_filepath(model=model_name,varname=var_choice,masked=True,timeframe=timeframe,monthly_scale=True)
    data=model_prep.highresmips_open_file(filepath_4_data,varname=var_choice,yr_i=yr_i,yr_f=yr_f)
    data=data.sel(lat=slice(lat_arr[0],lat_arr[1]),lon=slice(lon_arr[0],lon_arr[1]))

    dict_name[model_label+' - Uncoupled']=nan
    print('created column: ',model_name)
    for i,j in enumerate(data.groupby('time.month')):
        dict_name.loc[samples[i]:samples[i+1]-1,model_label+' - Uncoupled']=ravel(j[1][label_choice].values)
        print('inserted data for group/month: ',j[0])
        
    print('done with dictionary key: ',model_name)

def mesaclip_dict_scatter(lists,dict_name,samples,var_choice,label_choice,timeframe,yr_i,yr_f):

    for k,ens in enumerate(arange(1,11,1)):
        if (ens==9) & (timeframe!='historical'):
            continue
        print('created key in ensemble 0',str(ens))
        lists.append('mesaclip '+str(ens))
        datapath=find_files.mesaclip_ihesp_filepath('0'+str(ens),varname=var_choice,timeframe=timeframe,monthly_scale=True)
        data=model_prep.mesaclip_ihesp_open_file(datapath,varname=var_choice,yr_i=yr_i,yr_f=yr_f,ens_number='0'+str(ens),timeframe=timeframe,monthly_scale=True,masked=True)
        data=data.sel(lat=slice(lat_arr[0],lat_arr[1]),lon=slice(lon_arr[0],lon_arr[1]))
        dict_name['mesaclip '+str(ens)]=nan
        print('created column: ',str(ens))
        for i,j in enumerate(data.groupby('time.month')):
            print(ravel(j[1][label_choice].values).shape)
            dict_name.loc[samples[i]:samples[i+1]-1,'mesaclip '+str(ens)]=ravel(j[1][label_choice].values)
            print('inserted data for group/month: ',j[0])
    
        print('done with dictionary key: ',str(ens))

def loca_dict_scatter(lists,dict_name,samples,var_choice,label_choice,timeframe,yr_i,yr_f,model_name):
    idx = downscale_informal_array.index(model_name)
    model_label=downscale_model_array[idx]
    # print(model_label)

    try:
        filepath_4_data=find_files.loca_filepath(model_name,var_choice,highres=False,timeframe=timeframe)
        # print(filepath_4_data)
        data=model_prep.loca_monthly_open_file(filepath_4_data,var_choice,yr_i,yr_f,monthly_scale=True)
        # print(data)
        data=data.sel(lat=slice(lat_arr[0],lat_arr[1]),lon=slice(lon_arr[0],lon_arr[1]))
        lists.append(model_label+' - LOCA')
    except ValueError:
        return nan
        
    dict_name[model_label+' - LOCA']=nan
    print('created column: ',model_name)
    for i,j in enumerate(data.groupby('time.month')):
        dict_name.loc[samples[i]:samples[i+1]-1,model_label+' - LOCA']=ravel(j[1][label_choice].values)
        print('inserted data for group/month: ',j[0])
        
    print('done with dictionary key: ',model_name)

def gddp_dict_scatter(lists,dict_name,samples,var_choice,label_choice,timeframe,yr_i,yr_f,model_name):
    idx = downscale_informal_array.index(model_name)
    model_label=downscale_model_array[idx]

    try:
        filepath_4_data=find_files.gddp_filepath(model_name,var_choice)
        data=model_prep.gddp_open_file(filepath_4_data,var_choice,yr_i,yr_f,monthly_scale=True)
        data=data.sel(lat=slice(lat_arr[0],lat_arr[1]),lon=slice(lon_arr[0],lon_arr[1]))
        lists.append(model_label+' - BCSD')
    except ValueError:
        return nan
        
    dict_name[model_label+' - BCSD']=nan
    print('created column: ',model_name)
    for i,j in enumerate(data.groupby('time.month')):
        dict_name.loc[samples[i]:samples[i+1]-1,model_label+' - BCSD']=ravel(j[1][label_choice].values)
        print('inserted data for group/month: ',j[0])

        
    print('done with dictionary key: ',model_name)

def bccaq_dict_scatter(lists,dict_name,samples,var_choice,label_choice,timeframe,yr_i,yr_f,model_name):
    idx = downscale_informal_array.index(model_name)
    model_label=downscale_model_array[idx]

    try:
        filepath_4_data=find_files.bccaq_filepath(model_name,var_choice)
        data=model_prep.bccaq_open_file(filepath_4_data,var_choice,yr_i,yr_f,monthly_scale=True)
        data=data.sel(lat=slice(lat_arr[0],lat_arr[1]),lon=slice(lon_arr[0],lon_arr[1]))
        lists.append(model_label+' - BCCAQ')
    except ValueError:
        return nan
    dict_name[model_label+' - BCCAQ']=nan
    print('created column: ',model_name)
    for i,j in enumerate(data.groupby('time.month')):
        dict_name.loc[samples[i]:samples[i+1]-1,model_label+' - BCCAQ']=ravel(j[1][label_choice].values)
        print('inserted data for group/month: ',j[0])

        
    print('done with dictionary key: ',model_name)
    
####################################################

def coupled_highresmip_dict_rmse(var_choice,label_choice,timeframe,yr_i,yr_f,model_name,truth,calc):
    idx = cmip6_informal_array.index(model_name)
    model_label=cmip6_model_array[idx]
    
    print('created key in dictionary: ',model_name)

    try:
        filepath_4_data=find_files.highresmip_coupled_filepath(model=model_name,varname=var_choice,masked=True,timeframe=timeframe)
        data=model_prep.highresmips_open_file(filepath_4_data,varname=var_choice,yr_i=yr_i,yr_f=yr_f)
        months_to_keep = [5,6,7,8,9,10]
        data=data.sel(time=data.time.dt.month.isin(months_to_keep))
    except (ValueError,OSError) as e:
        return nan
        
    if (calc=='rmse'):
        data_ac=dr_g_calcs.rmse_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='mae'):
        data_ac=dr_g_calcs.mae_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='bias'):
        data_ac=dr_g_calcs.bias_calc_heatmap(data,truth,label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='std_dev'):
        data_ac = nanstd(data['pr_mm'].values)/nanstd(truth['pr_mm'].values)

    if (calc=='corr'):
        # data_ac=dr_g_calcs.taylor_skill_score(data,truth,label_choice,lat_loc,lon_loc)
        data_ac=nan
        
    print('done with dictionary key: ',model_name)
    return data_ac

def uncoupled_highresmip_dict_rmse(var_choice,label_choice,timeframe,yr_i,yr_f,model_name,truth,calc):
    idx = cmip6_informal_array.index(model_name)
    model_label=cmip6_model_array[idx]
    
    print('created key in dictionary: ',model_name)

    try:
        filepath_4_data=find_files.highresmip_uncoupled_filepath(model=model_name,varname=var_choice,masked=True,timeframe=timeframe)
        data=model_prep.highresmips_open_file(filepath_4_data,varname=var_choice,yr_i=yr_i,yr_f=yr_f)
        months_to_keep = [5,6,7,8,9,10]
        data=data.sel(time=data.time.dt.month.isin(months_to_keep))
    except (ValueError,OSError) as e:
        return nan
    
    if (calc=='rmse'):
        data_ac=dr_g_calcs.rmse_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='mae'):
        data_ac=dr_g_calcs.mae_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='bias'):
        data_ac=dr_g_calcs.bias_calc_heatmap(data,truth,label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='std_dev'):
        data_ac = nanstd(data['pr_mm'].values)/nanstd(truth['pr_mm'].values)

    if (calc=='corr'):
        # data_ac=dr_g_calcs.taylor_skill_score(data,truth,label_choice,lat_loc,lon_loc)
        data_ac=nan
        
    print('done with dictionary key: ',model_name)
    return data_ac

def mesaclip_dict_rmse(var_choice,label_choice,timeframe,yr_i,yr_f,truth,calc):
    # idx = cmip6_informal_array.index(model_name)
    # model_label=cmip6_model_array[idx]
    ens_arr=zeros(10,)
    for k,ens in enumerate(arange(1,11,1)):
        print('created key in ensemble 0',str(ens))
        
        datapath=find_files.mesaclip_ihesp_filepath('0'+str(ens),var_choice,timeframe=timeframe)
        data=model_prep.mesaclip_ihesp_open_file(datapath,varname=var_choice,yr_i=yr_i,yr_f=yr_f,ens_number='0'+str(ens),timeframe=timeframe,masked=True)
        months_to_keep = [5,6,7,8,9,10]
        data=data.sel(time=data.time.dt.month.isin(months_to_keep))
        if (calc=='rmse'):
            data_ac=dr_g_calcs.rmse_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

        if (calc=='mae'):
            data_ac=dr_g_calcs.mae_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)
    
        if (calc=='bias'):
            data_ac=dr_g_calcs.bias_calc_heatmap(data,truth,label_choice,lat_loc=lat_arr,lon_loc=lon_arr)
    
        if (calc=='std_dev'):
            data_ac = nanstd(data['pr_mm'].values)/nanstd(truth['pr_mm'].values)
    
        if (calc=='corr'):
            # data_ac=dr_g_calcs.taylor_skill_score(data,truth,label_choice,lat_loc,lon_loc)
            data_ac=nan
            
        ens_arr[k]=data_ac
        
        print('done with dictionary key: ',str(ens))
    
    return nanmean(ens_arr)


def loca_dict_rmse(var_choice,label_choice,timeframe,yr_i,yr_f,model_name,truth,calc):
    idx = downscale_informal_array.index(model_name)
    model_label=downscale_model_array[idx]
    
    print('created key in dictionary: ',model_name)
    try:
        filepath_4_data=find_files.loca_filepath(model_name,var_choice,highres=False,timeframe=timeframe)
        # print(filepath_4_data)
        data=model_prep.loca_monthly_open_file(filepath_4_data,var_choice,yr_i,yr_f)
        # print(data)
        months_to_keep = [5,6,7,8,9,10]
        data=data.sel(time=data.time.dt.month.isin(months_to_keep))
    except (ValueError,OSError) as e:
        return nan
        
    if (calc=='rmse'):
        data_ac=dr_g_calcs.rmse_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='mae'):
        data_ac=dr_g_calcs.mae_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='bias'):
        data_ac=dr_g_calcs.bias_calc_heatmap(data,truth,label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='std_dev'):
        data_ac = nanstd(data['pr_mm'].values)/nanstd(truth['pr_mm'].values)

    if (calc=='corr'):
        # data_ac=dr_g_calcs.taylor_skill_score(data,truth,label_choice,lat_loc,lon_loc)
        data_ac=nan
       
    print('done with dictionary key: ',model_name)
    return data_ac

def gddp_dict_rmse(var_choice,label_choice,timeframe,yr_i,yr_f,model_name,truth,calc):
    idx = downscale_informal_array.index(model_name)
    model_label=downscale_model_array[idx]
    
    print('created key in dictionary: ',model_name)
    
    filepath_4_data=find_files.gddp_filepath(model_name,var_choice)
    data=model_prep.gddp_open_file(filepath_4_data,var_choice,yr_i,yr_f)
    months_to_keep = [5,6,7,8,9,10]
    data=data.sel(time=data.time.dt.month.isin(months_to_keep))
    
    if (calc=='rmse'):
        data_ac=dr_g_calcs.rmse_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='mae'):
        data_ac=dr_g_calcs.mae_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='bias'):
        data_ac=dr_g_calcs.bias_calc_heatmap(data,truth,label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='std_dev'):
        data_ac = nanstd(data['pr_mm'].values)/nanstd(truth['pr_mm'].values)

    if (calc=='corr'):
        # data_ac=dr_g_calcs.taylor_skill_score(data,truth,label_choice,lat_loc,lon_loc)
        data_ac=nan
        
    print('done with dictionary key: ',model_name)
    return data_ac

def bccaq_dict_rmse(var_choice,label_choice,timeframe,yr_i,yr_f,model_name,truth,calc):
    idx = downscale_informal_array.index(model_name)
    model_label=downscale_model_array[idx]
    
    print('created key in dictionary: ',model_name)

    try:
        filepath_4_data=find_files.bccaq_filepath(model_name,var_choice,masked=True)
        data=model_prep.bccaq_open_file(filepath_4_data,var_choice,yr_i,yr_f)
        months_to_keep = [5,6,7,8,9,10]
        data=data.sel(time=data.time.dt.month.isin(months_to_keep))
        
    except ValueError:
        return nan
    
    if (calc=='rmse'):
        data_ac=dr_g_calcs.ub_rmse_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='mae'):
        data_ac=dr_g_calcs.mae_calc_heatmap(data,truth,label_choice=label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='bias'):
        data_ac=dr_g_calcs.bias_calc_heatmap(data,truth,label_choice,lat_loc=lat_arr,lon_loc=lon_arr)

    if (calc=='std_dev'):
        data_ac = nanstd(data['pr_mm'].values)/nanstd(truth['pr_mm'].values)

    if (calc=='corr'):
        # data_ac=dr_g_calcs.taylor_skill_score(data,truth,label_choice,lat_loc,lon_loc)
        data_ac=nan
        
    print('done with dictionary key: ',model_name)
    return data_ac


#the following functions (*_dict_anoms) are used to create pickle (.pkl) and numerical (.npy) files for cumulative anomaly plots (figs 2-3)

def couple_highresmip_dict_anoms(dict_name,model_name,labels,var_choice,label_choice,timeframe,yr_i,yr_f):
    idx = cmip6_informal_array.index(model_name)
    model_label=cmip6_model_array[idx]
    
    filepath_4_data=filepaths.highresmip_coupled_filepath(model=model_name,varname=var_choice,masked=True,timeframe=timeframe,monthly_scale=False)
    print(filepath_4_data)
    data=model_prep.highresmips_open_file(filepath_4_data,varname=var_choice,yr_i=yr_i,yr_f=yr_f)
    data_ac=dr_g_calcs.cumulative_precip_anom_curve(data,label_choice,lat_arr,lon_arr)
    dict_name.append(data_ac)
    labels.append(model_label)
    print('created key in dictionary: ',model_name)

def mesaclip_dict_anoms(dict_name,labels,var_choice,label_choice,timeframe,yr_i,yr_f):
    for i,j in enumerate(arange(1,11,1)):
        print('ensemble 0',str(j))
        datapath=filepaths.mesaclip_ihesp_filepath('0'+str(j),varname=var_choice,timeframe=timeframe,monthly_scale=False)
        print(datapath)
        data=model_prep.mesaclip_ihesp_open_file(datapath,varname=var_choice,yr_i=yr_i,yr_f=yr_f,ens_number='0'+str(j),timeframe=timeframe,monthly_scale=False,masked=True)
        data_ac=dr_g_calcs.cumulative_precip_anom_curve(data,label_choice,lat_arr,lon_arr)
        dict_name.append(data_ac)
        labels.append('ensemble 0'+str(j))

def uncouple_highresmip_dict_anoms(dict_name,model_name,labels,var_choice,label_choice,timeframe,yr_i,yr_f):
    idx = cmip6_informal_array.index(model_name)
    model_label=cmip6_model_array[idx]
    
    filepath_4_data=filepaths.highresmip_uncoupled_filepath(model=model_name,varname=var_choice,masked=True,timeframe=timeframe,monthly_scale=False)
    print(filepath_4_data)
    data=model_prep.highresmips_open_file(filepath_4_data,varname=var_choice,yr_i=yr_i,yr_f=yr_f)
    data_ac=dr_g_calcs.cumulative_precip_anom_curve(data,label_choice,lat_arr,lon_arr)
    dict_name.append(data_ac)
    labels.append(model_label)
    print('created key in dictionary: ',model_name)

def downscale_dict_anoms(dict_name,data_name,labels,var_choice,label_choice,timeframe,yr_i,yr_f):

    model_names=['cesm2','cnrm','hadgem3-mm','hadgem3-ll','ipsl','miroc6','mpi']
    print('creating dictionary for dataset: ',data_name)
    for k,l in enumerate(model_names):
        idx = downscale_informal_array.index(l)
        model_label=downscale_model_array[idx]
        try:
            if (data_name=='loca'):
                filepath_4_data=filepaths.loca_filepath(model=l,varname=var_choice,monthly_scale=False,timeframe=timeframe)
                print(filepath_4_data)
                data=model_prep.loca_monthly_open_file(filepath_4_data,varname=var_choice,yr_i=yr_i,yr_f=yr_f)
                data_ac=dr_g_calcs.cumulative_precip_anom_curve(data,label_choice,lat_arr,lon_arr)
                dict_name.append(data_ac)
                labels.append(model_label)
                print('created key in dictionary: ',l)
            elif (data_name=='bccaq'):
                filepath_4_data=filepaths.bccaq_filepath(model=l,varname=var_choice,masked=True)
                data=model_prep.bccaq_open_file(filepath_4_data,varname=var_choice,yr_i=yr_i,yr_f=yr_f,monthly_scale=False)
                data_ac=dr_g_calcs.cumulative_precip_anom_curve(data,label_choice,lat_arr,lon_arr)
                dict_name.append(data_ac)
                labels.append(model_label)
                print('created key in dictionary: ',l)
            else:
                filepath_4_data=filepaths.gddp_filepath(model=l,varname=var_choice)
                data=model_prep.gddp_open_file(filepath_4_data,varname=var_choice,yr_i=yr_i,yr_f=yr_f,monthly_scale=False)
                data_ac=dr_g_calcs.cumulative_precip_anom_curve(data,label_choice,lat_arr,lon_arr)
                dict_name.append(data_ac)
                labels.append(model_label)
                print('created key in dictionary: ',l)
        except (OSError, ValueError) as e:
            print('no files')
            pass            