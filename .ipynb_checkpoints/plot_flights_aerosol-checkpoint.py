#=========================================================
#=========================================================

#--------------------------------
# Imports
#--------------------------------
import numpy as np
import matplotlib.pyplot as plt
import glob
import xarray
import datetime
import calendar
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import matplotlib
import pickle
import pandas as pd
import os
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator as nn
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.ticker as ticker
from scipy import stats
#from PseudoNetCDF.icarttfiles import ffi1001
import PseudoNetCDF as pnc
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import scipy.io
#--------------------------------------------
#--------------------------------------------

def toTimestamp(d):
    return calendar.timegm(d.timetuple())

def toDatetime(d):
    return datetime.datetime.utcfromtimestamp(d)


    


path = '/mnt/raid/mwstanfo/camp2ex/aerosol/'
aero_files = glob.glob(path+'*.ict')
aero_files = sorted(aero_files)
num_flights = len(aero_files)

elev_path = '/mnt/raid/mwstanfo/camp2ex/elevation/'
elev_files = glob.glob(elev_path+'*.ict')
elev_files = sorted(elev_files)

fcdp_path = '/mnt/raid/mwstanfo/camp2ex/cloud/'
fcdp_files = glob.glob(fcdp_path+'*.ict')
fcdp_files = sorted(fcdp_files)

nav_path = '/mnt/raid/mwstanfo/camp2ex/nav/modified/'
nav_files = glob.glob(nav_path+'*.ict')
nav_files = sorted(nav_files)
# eliminate first 6 flights and last 4 flights since
# no aerosol data available
nav_files = nav_files[6:]
nav_files = nav_files[:-4]

#nav_10Hz_path = '/mnt/raid/mwstanfo/camp2ex/nav/orig/10Hz/'
#nav_10Hz_files = glob.glob(nav_10Hz_path+'*.ict')
#nav_10Hz_files = sorted(nav_10Hz_files)
# eliminate first 6 flights and last 4 flights since
# no aerosol data available
#nav_10Hz_files = nav_10Hz_files[4:]
#nav_10Hz_files = nav_10Hz_files[:-2]

# get dates of elevation files
elev_dates_dt = []
for elev_file in elev_files:
    # get date
    tmp_str = elev_file.split('/')
    tmp_str = tmp_str[-1]
    tmp_str = tmp_str.split('_')
    tmp_str = tmp_str[-2]
    tmp_year = int(tmp_str[0:4])
    tmp_month = int(tmp_str[4:6])
    tmp_day = int(tmp_str[6:])
    elev_date_dt = datetime.datetime(tmp_year,tmp_month,tmp_day)
    elev_dates_dt.append(elev_date_dt) 
    
# eliminate first two flights since no aerosol data was available
elev_files = elev_files[2:]
elev_dates_dt = elev_dates_dt[2:]

# cloud flag
path = '/mnt/raid/mwstanfo/camp2ex/'
file = path+'CloudFlag.mat'
mat = scipy.io.loadmat(file)
mat_cloud_flag = np.squeeze(mat['CloudFlag'])
num_flights = 19
cloud_flag = []
cloud_flag_time = []

for ii in range(num_flights):
    tmp_cloud_flag_time = mat_cloud_flag[ii][0][0][0][0] # time
    tmp_cloud_flag = mat_cloud_flag[ii][0][0][1][0] # cloud flag
    cloud_flag.append(tmp_cloud_flag)
    cloud_flag_time.append(tmp_cloud_flag_time)

# convert cloud flag time to UTC
cloud_flag_times_dt = []
for ii in range(num_flights):
    tmp_cloud_flag_time = cloud_flag_time[ii]
    # tmp_cloud_flag_time is in hour.fraction_of_hour from the start time
    start_time = tmp_cloud_flag_time[0]
    start_time_hour = int(np.floor(start_time))
    start_time_frac_of_hour = start_time - start_time_hour
    start_time_min_frac_of_min = start_time_frac_of_hour*60.
    start_time_min = int(np.floor(start_time_min_frac_of_min))
    start_time_frac_of_min = start_time_min_frac_of_min - start_time_min
    start_time_sec_frac_of_sec = start_time_frac_of_min*60.
    start_time_sec = np.round(start_time_sec_frac_of_sec)
    start_time_sec = start_time_sec.astype(int)

    #start_time_sec = int(np.floor(start_time_sec_frac_of_sec))
    #start_time_frac_of_sec = start_time_sec_frac_of_sec - start_time_sec
    #start_time_millisec = start_time_frac_of_sec*1000.
    #start_time_millisec = start_time_millisec.astype(int)
    
    start_time_dt = datetime.datetime(elev_dates_dt[ii].year,\
                                      elev_dates_dt[ii].month,\
                                      elev_dates_dt[ii].day,\
                                      start_time_hour,\
                                      start_time_min,\
                                      start_time_sec)
    
    # need to convert this to seconds since it is seconds past the start time
    #tmp_secs_past = cloud_flag_time[ii]*60.*60.
    tmp_millisecs_past = cloud_flag_time[ii]*60.*60.*1000.
    millisecs_past = tmp_millisecs_past - tmp_millisecs_past[0]
    millisecs_past = millisecs_past.astype(int)
    secs_past = np.round(millisecs_past/1000)
    #secs_past = secs_past.astype(int)
    # round to nearest second

    #dum = np.diff(secs_past)
    #np.unique(dum)
    #print(aaaa)
    tmp_time = np.array([start_time_dt + datetime.timedelta(seconds=int(secs_past[jj])) for jj in range(len(secs_past))])
    cloud_flag_times_dt.append(tmp_time)
    #print(aaaaa)
    
    


aero_dates_dt = []
fcdp_dates_dt = []
nav_dates_dt = []
nav_10Hz_dates_dt = []
#N_D_all_flights = {}
#N_tot_all_flights = {}

# loop through flights
for ii in range(num_flights):
    print(ii)
    print(str((ii+1)/num_flights*100.)+'% done')
    
        
    # get aerosol dates
    tmp_str = aero_files[ii].split('/')
    tmp_str = tmp_str[-1]
    tmp_str = tmp_str.split('_')
    tmp_str = tmp_str[-2]
    tmp_year = int(tmp_str[0:4])
    tmp_month = int(tmp_str[4:6])
    tmp_day = int(tmp_str[6:])
    aero_date_dt = datetime.datetime(tmp_year,tmp_month,tmp_day)
    aero_dates_dt.append(aero_date_dt)
    
    # nav dates
    tmp_str = nav_files[ii].split('/')
    tmp_str = tmp_str[-1]
    tmp_str = tmp_str.split('_')
    tmp_str = tmp_str[-2]
    tmp_year = int(tmp_str[0:4])
    tmp_month = int(tmp_str[4:6])
    tmp_day = int(tmp_str[6:])
    nav_date_dt = datetime.datetime(tmp_year,tmp_month,tmp_day)
    nav_dates_dt.append(nav_date_dt) 
    
    # nav 10Hz dates
    #tmp_str = nav_10Hz_files[ii].split('/')
    #tmp_str = tmp_str[-1]
    #tmp_str = tmp_str.split('_')
    #tmp_str = tmp_str[-2]
    #tmp_year = int(tmp_str[0:4])
    #tmp_month = int(tmp_str[4:6])
    #tmp_day = int(tmp_str[6:])
    #nav_10Hz_date_dt = datetime.datetime(tmp_year,tmp_month,tmp_day)
    #nav_10Hz_dates_dt.append(nav_10Hz_date_dt)        

    # get fcdp dates
    tmp_str = fcdp_files[ii].split('/')
    tmp_str = tmp_str[-1]
    tmp_str = tmp_str.split('_')
    tmp_str = tmp_str[-2]
    tmp_year = int(tmp_str[0:4])
    tmp_month = int(tmp_str[4:6])
    tmp_day = int(tmp_str[6:])
    fcdp_date_dt = datetime.datetime(tmp_year,tmp_month,tmp_day)
    fcdp_dates_dt.append(fcdp_date_dt)     
    
    #print(aaaa)
    
    # aerosol file
    aero_infile = pnc.pncopen(aero_files[ii], format = 'ffi1001')
    N_D = []
    for key in aero_infile.variables:
        var = np.array(aero_infile.variables[key].copy())
        #print(key,'max:',np.max(var),'min:',np.min(var))
        if 'Dp' in key:
            N_D.append(var)

        # seconds since 00Z
        if key == 'Time_Start':
            aero_time_start = var
    N_D = np.array(N_D)

    aero_times_dt = aero_date_dt + aero_time_start*datetime.timedelta(seconds=1)
    aero_infile.close()
    
    # cloud flag
    cloud_flag_times_dt_single_flight = cloud_flag_times_dt[ii]
    cloud_flag_single_flight = cloud_flag[ii]

    # elevation file
    elev_infile = pnc.pncopen(elev_files[ii],format='ffi1001')
    elev_date_dt = elev_dates_dt[ii]
    elevation_time_start = np.array(elev_infile.variables['Time_Start'].copy())
    alt_agl = np.array(elev_infile.variables['Altitude_AGL_m'].copy())
    ground_elevation = np.array(elev_infile.variables['Ground_Elevation_m'].copy())
    gps_alt = np.array(elev_infile.variables['GPS_Altitude_m'].copy())
    elev_times_dt = elev_date_dt + elevation_time_start*datetime.timedelta(seconds=1)  
    aircraft_alt = gps_alt-ground_elevation
    elev_infile.close()
    
    # fcdp file
    fcdp_infile = pnc.pncopen(fcdp_files[ii],format='ffi1001')
    fcdp_con = np.array(fcdp_infile.variables['conc'].copy()) #/L
    fcdp_con = fcdp_con*1.e-3 #/cm^3
    fcdp_time_start = np.array(fcdp_infile.variables['Time_Start'].copy())
    fcdp_infile.close()
    fcdp_date_dt = fcdp_dates_dt[ii]
    fcdp_times_dt = fcdp_date_dt + fcdp_time_start*datetime.timedelta(seconds=1)  
    

        
    # nav 10Hz file
    #nav_10Hz_infile = pnc.pncopen(nav_10Hz_files[ii],format='ffi1001') 
    
    # nav file
    nav_infile = pnc.pncopen(nav_files[ii],format='ffi1001')
    #for key in nav_infile.variables:
    #    var = np.array(nav_infile.variables[key].copy())
    #    print(key,np.shape(var))
    temp = np.array(nav_infile.variables['Total_Air_Temp'].copy())
    rh = np.array(nav_infile.variables['Relative_Humidity'].copy())
    w = np.array(nav_infile.variables['Vertical_Speed'].copy())
    nav_time_start = np.array(nav_infile.variables['Time_Start'].copy())
    nav_date_dt = nav_dates_dt[ii]
    nav_times_dt = nav_date_dt + nav_time_start*datetime.timedelta(seconds=1)  
    nav_infile.close()
    
    #if ii < 16:
    #    continue
        
    dlogD = 0.061 # nm
    tmpid = np.where(N_D < 0.)
    N_D[tmpid] = np.nan
    N_tot = np.nansum(N_D,axis=0)
    #N_D_all_flights[aero_date_dt] = N_D
    dN_dlogD = np.array([N_D[jj,:]/dlogD for jj in range(len(N_D[:,0]))])
    #N_tot_all_flights[aero_date_dt] = N_tot


    #Geometric mean particle diameters (in nm) of size bins 1-30 are 10.0000, 11.5164, 13.2627, 15.2738, 17.5899, 20.2571, 23.3289, 26.8664, 30.9403, 35.6320, 41.0351, 47.2575, 54.4235, 62.6761, 72.1802, 83.1253, 95.7302, 110.2464, 126.9639, 146.2163, 168.3880, 193.9219, 223.3276, 257.1923, 296.1921, 341.1057, 392.8299, 452.3974, 520.9976, 600.0000. Size distribution dlogD = 0.061 = "-Geometric mean particle diameters (in nm) of size bins 1-30 are 10.0000, 11.5164, 13.2627, 15.2738, 17.5899, 20.2571, 23.3289, 26.8664, 30.9403, 35.6320, 41.0351, 47.2575, 54.4235, 62.6761, 72.1802, 83.1253, 95.7302, 110.2464, 126.9639, 146.2163, 168.3880, 193.9219, 223.3276, 257.1923, 296.1921, 341.1057, 392.8299, 452.3974, 520.9976, 600.0000. Size distribution dlogD = 0.061" 
    
    diam_mid = np.array([10.0000, 11.5164, 13.2627, 15.2738, 17.5899, 20.2571, 23.3289, 26.8664, 30.9403, 35.6320, 41.0351, 47.2575, 54.4235, 62.6761, 72.1802, 83.1253, 95.7302, 110.2464, 126.9639, 146.2163, 168.3880, 193.9219, 223.3276, 257.1923, 296.1921, 341.1057, 392.8299, 452.3974, 520.9976, 600.0000])
    
    # calculate log of geometric standard deviation as a function of time
    

    log_geo_std_dev = []
    geo_mean_D = []
    vol_weight_mean_D = []
    tmp = []
    for jj in range(len(aero_times_dt)):
        if (np.max(N_D[:,jj]) == 0.) or (np.all(np.isnan(N_D[:,jj]))):
            geo_mean_D.append(np.nan)
            vol_weight_mean_D.append(np.nan)
            log_geo_std_dev.append(np.nan)
        else:
            
            tmp_geo_mean_D = np.nanprod(diam_mid**(N_D[:,jj]/N_tot[jj]) )
            tmp_log_geo_std_dev = ( np.nansum(np.diff(np.log10(diam_mid) - np.log10(tmp_geo_mean_D))**2.) / (np.nansum(N_D[:,jj])-1) )**0.5
        
            tmp_vol_weight_mean_D =  np.nansum(N_D[:,jj] *diam_mid**4.) / np.nansum(N_D[:,jj] * diam_mid**3.)
            log_geo_std_dev.append(tmp_log_geo_std_dev)
            geo_mean_D.append(tmp_geo_mean_D)
            vol_weight_mean_D.append(tmp_vol_weight_mean_D)
            
    log_geo_std_dev = np.array(log_geo_std_dev)
    vol_weight_mean_D = np.array(vol_weight_mean_D)
    geo_std_dev = log_geo_std_dev.copy()
    geo_std_dev[~np.isnan(geo_std_dev)] = 10.**geo_std_dev[~np.isnan(geo_std_dev)]
    geo_mean_D = np.array(geo_mean_D)
    tmpid = np.where(N_tot == 0.)
    geo_std_dev[tmpid] = np.nan
    log_geo_std_dev[tmpid] = np.nan
    geo_mean_D[tmpid] = np.nan
    
    
#    print(aaaaa)
    
    # limit times to when we have aerosol data, fcdp data, and elevation data
    # since these are all needed to perform proper calculations

    min_aero_times_dt = np.min(aero_times_dt)
    max_aero_times_dt = np.max(aero_times_dt)   
    min_fcdp_times_dt = np.min(fcdp_times_dt)
    max_fcdp_times_dt = np.max(fcdp_times_dt)     
    min_nav_times_dt = np.min(nav_times_dt)
    max_nav_times_dt = np.max(nav_times_dt)
    
    lower_time_limit = np.max([min_aero_times_dt,min_fcdp_times_dt,min_nav_times_dt])
    upper_time_limit = np.min([max_aero_times_dt,max_fcdp_times_dt,max_nav_times_dt])
    
    tmpid = np.where( (aero_times_dt >= lower_time_limit) & (aero_times_dt <= upper_time_limit) )
    aero_times_dt = aero_times_dt[tmpid]
    N_tot = N_tot[tmpid]
    geo_std_dev = geo_std_dev[tmpid]
    geo_mean_D = geo_mean_D[tmpid]
    vol_weight_mean_D = vol_weight_mean_D[tmpid]
    
    tmpid = np.where( (fcdp_times_dt >= lower_time_limit) & (fcdp_times_dt <= upper_time_limit) )
    fcdp_times_dt = fcdp_times_dt[tmpid]    
    fcdp_con = fcdp_con[tmpid]
    
    tmpid = np.where( (nav_times_dt >= lower_time_limit) & (nav_times_dt <= upper_time_limit) )
    nav_times_dt = nav_times_dt[tmpid]    
    temp = temp[tmpid]    
    rh = rh[tmpid]    

    
    tmpid = np.where( (elev_times_dt >= lower_time_limit) & (elev_times_dt <= upper_time_limit) )
    elev_times_dt = elev_times_dt[tmpid]    
    aircraft_alt = aircraft_alt[tmpid]      
    
    tmpid = np.where( (cloud_flag_times_dt_single_flight >= lower_time_limit) & (cloud_flag_times_dt_single_flight <= upper_time_limit) )
    cloud_flag_times_dt_single_flight = cloud_flag_times_dt_single_flight[tmpid]
    cloud_flag_single_flight = cloud_flag_single_flight[tmpid]
    
    

    
    # create cloud mask
    
    
    # create cloud mask
    cloud_mask = np.zeros(np.shape(cloud_flag_single_flight))
    tmpid = np.where(cloud_flag_single_flight == 1.)
    cloud_mask[tmpid] = 1
    
    
    
    #cloud_mask = np.zeros(np.shape(fcdp_con))
    #tmpid = np.where(fcdp_con > 0.03)
    #cloud_mask[tmpid] = 1
    
    # calculate "clear sky" aerosol
    N_D_out_of_cloud = np.zeros(np.shape(N_D))
    dN_dlogD_out_of_cloud = np.zeros(np.shape(N_D))
    tmpid = np.where(cloud_mask == 1.)
    N_D_out_of_cloud[:,tmpid] = np.nan
    dN_dlogD_out_of_cloud[:,tmpid] = np.nan    
    tmpid = np.where(cloud_mask == 0.)
    N_D_out_of_cloud[:,tmpid] = N_D[:,tmpid]
    dN_dlogD_out_of_cloud[:,tmpid] = dN_dlogD[:,tmpid]     
    comp_dist_out_of_cloud = np.nanmean(N_D_out_of_cloud,axis=1)
    comp_dist_dN_dlogD_out_of_cloud = np.nanmean(dN_dlogD_out_of_cloud,axis=1)
    
    
    N_tot_out_of_cloud = N_tot[cloud_mask == 0.]
    geo_std_dev_out_of_cloud = geo_std_dev[cloud_mask == 0.]
    vol_weight_mean_D_out_of_cloud = vol_weight_mean_D[cloud_mask == 0.]
    geo_mean_D_out_of_cloud = geo_mean_D[cloud_mask == 0.]
    alt_out_of_cloud = aircraft_alt[cloud_mask == 0.]

    alt_bins = np.arange(0,11,1)*1.e3
    N_tot_median_profile = []
    geo_std_dev_median_profile = []
    vol_weight_mean_D_median_profile = []
    geo_mean_D_median_profile = []
    for kk in range(len(alt_bins)-1):
        tmpid = np.where((alt_out_of_cloud > alt_bins[kk] ) & (alt_out_of_cloud <= alt_bins[kk+1]) )
        if np.size(tmpid) > 0.:
            tmpid = np.squeeze(tmpid)
            if np.size(tmpid) > 10.:
                tmp_N_tot = N_tot_out_of_cloud[tmpid]
                tmp_geo_std_dev = geo_std_dev_out_of_cloud[tmpid]
                tmp_geo_mean_D = geo_mean_D_out_of_cloud[tmpid]
                tmp_vol_weight_mean_D = vol_weight_mean_D_out_of_cloud[tmpid]
                tmpid2 = np.where(tmp_N_tot > 0.)
                tmp_N_tot = tmp_N_tot[tmpid2]
                tmp_geo_std_dev = tmp_geo_std_dev[tmpid2]
                tmp_vol_weight_mean_D = tmp_vol_weight_mean_D[tmpid2]
                tmp_med_N_tot = np.median(tmp_N_tot)
                tmp_med_geo_std_dev = np.median(tmp_geo_std_dev)
                tmp_med_geo_mean_D = np.nanmedian(tmp_geo_mean_D)
                tmp_med_vol_weight_mean_D = np.median(tmp_vol_weight_mean_D)
                N_tot_median_profile.append(tmp_med_N_tot)
                geo_std_dev_median_profile.append(tmp_med_geo_std_dev)
                geo_mean_D_median_profile.append(tmp_med_geo_mean_D)
                vol_weight_mean_D_median_profile.append(tmp_med_vol_weight_mean_D)
            else:
                N_tot_median_profile.append(np.nan)
                geo_std_dev_median_profile.append(np.nan)
                vol_weight_mean_D_median_profile.append(np.nan)
                geo_mean_D_median_profile.append(np.nan)
        else:
            N_tot_median_profile.append(np.nan)
            geo_std_dev_median_profile.append(np.nan)
            vol_weight_mean_D_median_profile.append(np.nan)
            geo_mean_D_median_profile.append(np.nan)
    N_tot_median_profile = np.array(N_tot_median_profile)
    geo_std_dev_median_profile = np.array(geo_std_dev_median_profile)
    vol_weight_mean_D_median_profile = np.array(vol_weight_mean_D_median_profile)
    geo_mean_D_median_profile = np.array(geo_mean_D_median_profile)

    #if aero_date_dt != datetime.datetime(2019,9,21):
    #    continue
    

    # plot 
    #if False:
    if True:
        dfmt = mdates.DateFormatter('%d-%H:%M')
        Fontsize=16
        fig = plt.figure(figsize=(20,20))
        gs=GridSpec(5,6) # 6 rows, 6 columns

        ax1 = fig.add_subplot(gs[0,0:4])
        ax2 = fig.add_subplot(gs[1,0:4])
        ax3 = fig.add_subplot(gs[2,0:4])
        ax4 = fig.add_subplot(gs[3,0:4])
        ax5 = fig.add_subplot(gs[0,4:6])
        ax6 = fig.add_subplot(gs[1,4:6])
        ax7 = fig.add_subplot(gs[2,4:6])
        ax8 = fig.add_subplot(gs[3,4:6])
        ax9 = fig.add_subplot(gs[4,0:4])
        #ax10 = fig.add_subplot(gs[4,4:6])
        
        axlist = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
        for ax in axlist:
            ax.grid(which='both')
            ax.tick_params(labelsize=Fontsize)
        axlist2 = [ax1,ax2,ax3,ax4,ax9]
        for ax in axlist2:
            ax.set_xlabel('Time UTC [Day-HH:MM]',fontsize=Fontsize)

        ax1.set_ylabel('$N_{a,tot}$ [cm$^{-3}$]',fontsize=Fontsize)
        ax2.set_ylabel('D$_{g}$ [nm]',fontsize=Fontsize)
        ax3.set_ylabel('$\\sigma_{G}$',fontsize=Fontsize)
        ax4.set_ylabel('Altitude AGL [km]',fontsize=Fontsize)
        

        
        # N_tot
        tmp_N_tot = N_tot.copy()
        tmp_N_tot[tmp_N_tot == 0.] = np.nan
        ax1.plot(aero_times_dt,tmp_N_tot,lw=1,c='k',label='1-s resolution')
        # calculate running mean
        num_seconds = aero_time_start.copy()
        tmp_N_tot[np.isnan(tmp_N_tot)] = 0.
        N = int(len(num_seconds)/10.) # 10-second moving average
        tmp_N_tot_run_mean = np.convolve(tmp_N_tot, np.ones(N)/N, mode='same')
        tmp_N_tot_run_mean[tmp_N_tot == 0.] = np.nan
        
        # plot running mean
        ax1.plot(aero_times_dt,tmp_N_tot_run_mean,lw=3,c='red',\
                 label='10-s running mean')
        lgnd1 = ax1.legend(loc='upper left',fontsize=Fontsize,\
                   bbox_to_anchor=(-0.025,1.35),framealpha=1)
        
        
        
        # geometric mean diameter
        ax2.plot(aero_times_dt,geo_mean_D,lw=1,c='k',label='1s resolution')
        

        # calculate running mean
        tmp_geo_mean_D = geo_mean_D.copy()
        num_seconds = aero_time_start.copy()
        tmp_geo_mean_D[np.isnan(tmp_geo_mean_D)] = 0.
        N = int(len(num_seconds)/10.) # 10-second moving average
        tmp_geo_mean_D_run_mean = np.convolve(tmp_geo_mean_D, np.ones(N)/N, mode='same')
        tmp_geo_mean_D_run_mean[tmp_geo_mean_D == 0.] = np.nan
        
        # plot running mean
        ax2.plot(aero_times_dt,tmp_geo_mean_D_run_mean,lw=3,c='red',\
                 label='10-s running mean')
        lgnd2 = ax2.legend(loc='upper left',fontsize=Fontsize,\
                   bbox_to_anchor=(-0.025,1.35),framealpha=1)
                    
        
        # geometric standard deviation
        #ax2.set_ylabel('$N_{a}(D)$ [cm$^{-3}$]',fontsize=Fontsize)
        ax3.set_xlabel('Aerosol Particle Diameter [nm]',fontsize=Fontsize)
        ax3.plot(aero_times_dt,geo_std_dev,lw=1,c='k')
        #ax3.set_title('$\\sigma_{G}$',fontsize=Fontsize*2.)
        ax3.set_ylabel('$\\sigma_{G}$',fontsize=Fontsize)
        
        
        # calcualte the mean 0 deg C layer height
        tmpid = np.where( (temp > -0.1) & (temp < 0.1) )
        if np.size(tmpid) > 0.:
            tmpid = np.squeeze(tmpid)
            mean_0degC_alt = np.nanmean(aircraft_alt[tmpid])
            axlist4 = [ax6,ax7,ax8]
            for ax in axlist4:
                ax.axhline(mean_0degC_alt*1.e-3,lw=3,ls='dashed',\
                           color='magenta',label='0 $^{\\circ}$C')

        # altitude, temp, and RH
        ax4.plot(elev_times_dt,aircraft_alt*1.e-3,lw=2,c='k')
        ax4a = ax4.twinx()
        tmp_temp = temp.copy()
        tmp_temp[tmp_temp < -999.] = np.nan
        tmp_rh = rh.copy()
        tmp_rh[tmp_rh < -999.] = np.nan
        
        ax4a.plot(nav_times_dt,tmp_temp,lw=2,c='red')
        ax4a.spines['right'].set_color('red')
        ax4a.xaxis.label.set_color('red')
        ax4a.tick_params(axis='y', labelsize=Fontsize, colors='red')
        ax4a.spines['right'].set_visible(False) 
        ax4a.set_ylabel('Temperature [$^{\\circ}$C]',fontsize=Fontsize,c='red')
        ax4a.axhline(0.,lw=3,ls='dashed',\
                    color='magenta',label='0 $^{\\circ}$C')
        
        #ax4a.invert_yaxis()
        ax4b = ax4.twinx()
        ax4b.set_ylabel('RH [%]',fontsize=Fontsize)
        ax4b.yaxis.set_label_position("right")
        ax4b.yaxis.set_ticks_position("right")
        ax4b.spines['right'].set_position(('axes', 1.075))
        ax4b.yaxis.label.set_color('blue')
        ax4b.tick_params(axis='y',labelsize=Fontsize,colors='blue')
        ax4b.spines['right'].set_color('blue')    
        ax4b.plot(nav_times_dt,tmp_rh,lw=2,c='blue')
        ax4b.set_ylim(0,105.)
        
        ax4a.legend(fontsize=Fontsize,loc='upper left',bbox_to_anchor=(0,1.25))
        
            
        # composite distribution
        ax5.plot(diam_mid,comp_dist_dN_dlogD_out_of_cloud,lw=2,c='k')
        ax5.set_title('Out-of-Cloud\nComposite Size Distribution',fontsize=Fontsize)
        ax5.set_ylabel('$\\frac{dN_{a}}{dlogD}$ [cm$^{-3}$]',\
                       fontsize=Fontsize)
        ax5.set_xlabel('Aerosol Particle Diameter [nm]',fontsize=Fontsize)

        
        # fcdp conc
        tmp_fcdp_con = fcdp_con.copy()
        tmp_fcdp_con[tmp_fcdp_con == 0.] = np.nan
        tmp_fcdp_con[~np.isnan(tmp_fcdp_con)] = tmp_fcdp_con[~np.isnan(tmp_fcdp_con)]
        ax9.plot(fcdp_times_dt,tmp_fcdp_con,lw=1,c='k')
        ax9.set_ylabel('Cloud Droplet\nNumber Concenration\n($N_{c}$) [cm$^{-3}$]',fontsize=Fontsize)
        



        # scatter plot of out-of-cloud aerosol concentration
        # versus altitude
        tmp_cloud_mask = cloud_mask.copy()
        tmp_aircraft_alt = aircraft_alt.copy()
        tmpid = np.where((N_tot > 0.) & (tmp_cloud_mask == 0.))
        if np.size(tmpid) > 0.:
            tmpid = np.squeeze(tmpid)
            tmp_N_tot = N_tot[tmpid]
            tmp_aircraft_alt = tmp_aircraft_alt[tmpid]
            tmp_cloud_mask = tmp_cloud_mask[tmpid]
            ax6.scatter(tmp_N_tot,tmp_aircraft_alt*1.e-3,s=1,c='k',marker='o')
            
        # profile of median out-of-cloud N_tot
        mid_alt_bins = np.array([(alt_bins[kk]+alt_bins[kk+1])/2. for kk in range(len(alt_bins)-1)])
        ax6.plot(N_tot_median_profile,mid_alt_bins*1.e-3,lw=3,c='red',label='Median')
        ax6.legend(fontsize=Fontsize)
        ax6.set_xscale('log')
        ax6.set_xlabel('Out-of-Cloud N$_{a,tot}$ [cm$^{-3}$]',fontsize=Fontsize)
        ax6.set_ylabel('Altitude AGL [m]',fontsize=Fontsize)
        ax6.axvline(1.e3,lw=3,ls='dashed',c='green')
        ax6.axvline(1.e4,lw=3,ls='dashed',c='blue')
        ax6.axvline(1.e5,lw=3,ls='dashed',c='darkorange')
        ax6.text(1.e3,np.max(aircraft_alt)*1.e-3+0.85,\
                 '10$^{3}$',ha='center',\
                 c='green',fontsize=Fontsize)#transform=ax9.transAxes)
        ax6.text(1.e4,np.max(aircraft_alt)*1.e-3+0.85,\
                 '10$^{4}$',ha='center',\
                 c='blue',fontsize=Fontsize)#transform=ax9.transAxes)      
        ax6.text(1.e5,np.max(aircraft_alt)*1.e-3+0.85,\
                 '10$^{5}$',ha='center',\
                 c='darkorange',fontsize=Fontsize)#transform=ax9.transAxes)  
        
        
        # scatter plot of out-of-cloud aerosol geometric standard deviation
        # versus altitude
        tmp_cloud_mask = cloud_mask.copy()
        tmp_aircraft_alt = aircraft_alt.copy()
        tmpid = np.where((geo_std_dev > 0.) & (tmp_cloud_mask == 0.))
        if np.size(tmpid) > 0.:
            tmpid = np.squeeze(tmpid)
            tmp_geo_std_dev = geo_std_dev[tmpid]
            tmp_aircraft_alt = tmp_aircraft_alt[tmpid]
            tmp_cloud_mask = tmp_cloud_mask[tmpid]
            ax7.scatter(tmp_geo_std_dev,tmp_aircraft_alt*1.e-3,s=1,c='k',marker='o')
            
        # profile of median out-of-cloud geometric standard deviation
        mid_alt_bins = np.array([(alt_bins[kk]+alt_bins[kk+1])/2. for kk in range(len(alt_bins)-1)])
        ax7.plot(geo_std_dev_median_profile,mid_alt_bins*1.e-3,lw=3,c='red',label='Median')
        ax7.legend(fontsize=Fontsize)
        ax7.set_xlabel('Out-of-Cloud $\\sigma_{G}$',fontsize=Fontsize)
        ax7.set_ylabel('Altitude AGL [m]',fontsize=Fontsize)
                                       
        # scatter plot of out-of-cloud aerosol geometric standard deviation
        # versus altitude
        tmp_cloud_mask = cloud_mask.copy()
        tmp_aircraft_alt = aircraft_alt.copy()
        tmpid = np.where((vol_weight_mean_D > 0.) & (tmp_cloud_mask == 0.))
        if np.size(tmpid) > 0.:
            tmpid = np.squeeze(tmpid)
            tmp_geo_mean_D = geo_mean_D[tmpid]
            tmp_aircraft_alt = tmp_aircraft_alt[tmpid]
            tmp_cloud_mask = tmp_cloud_mask[tmpid]
            ax8.scatter(tmp_geo_mean_D,tmp_aircraft_alt*1.e-3,s=1,c='k',marker='o')
            
        # profile of median out-of-cloud geometric standard deviation
        mid_alt_bins = np.array([(alt_bins[kk]+alt_bins[kk+1])/2. for kk in range(len(alt_bins)-1)])
        ax8.plot(geo_mean_D_median_profile,mid_alt_bins*1.e-3,lw=3,c='red',label='Median')
        ax8.legend(fontsize=Fontsize)
        ax8.set_xscale('log')
        ax8.set_xlabel('Out-of-Cloud $D_{G}$ [nm]',fontsize=Fontsize)
        ax8.set_ylabel('Altitude AGL [km]',fontsize=Fontsize)
        ax8.axvline(20,lw=3,ls='dashed',c='green')
        ax8.text(20,np.max(aircraft_alt)*1.e-3+0.85,\
                 '20',ha='center',\
                 c='green',fontsize=Fontsize)#transform=ax9.transAxes)
        ax8.axvline(50,lw=3,ls='dashed',c='blue')
        ax8.text(50,np.max(aircraft_alt)*1.e-3+0.85,\
                 '50',ha='center',\
                 c='blue',fontsize=Fontsize)#transform=ax9.transAxes)
        ax8.axvline(100,lw=3,ls='dashed',c='darkorange')
        ax8.text(100,np.max(aircraft_alt)*1.e-3+0.85,\
                 '100',ha='center',\
                 c='darkorange',fontsize=Fontsize)#transform=ax9.transAxes)
        
        
        ax6.set_xlim(1.e2,1.e6)
        ax7.set_xlim(1.,1.05)
        ax8.set_xlim(1.e1,150)
        ax9.set_ylim(1.e-2,5.e2)
        ax3.set_ylim(1,1.1)
        ax1.set_ylim(5.e1,5.e6)
        ax2.set_ylim(5,300)

        
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax5.set_yscale('log')
        ax5.set_xscale('log')  
        ax9.set_yscale('log')  
        
        # create cloud mask
        #tmp_cloud_mask = np.zeros(np.shape(fcdp_con))
        tmp_cloud_mask = np.zeros(np.shape(cloud_flag_single_flight))
        #tmpid = np.where(fcdp_con > 0.03)
        tmpid = np.where(cloud_flag_single_flight == 1.)
        tmp_cloud_mask[tmpid] = 1
        cloud_objects,num_cloud_objects = ndimage.label(tmp_cloud_mask)
        
        axlist3 = [ax1,ax2,ax3,ax4,ax9]

        for ax in axlist3:
            for jj in range(num_cloud_objects):
                tmpid = np.where(cloud_objects == jj+1)
                if np.size(tmpid) > 1:
                    tmpid = np.squeeze(tmpid)
                    #dum = ax.axvspan(fcdp_times_dt[tmpid[0]],fcdp_times_dt[tmpid[-1]],\
                    #                        alpha=0.5,color='deepskyblue',ec=None)
                    dum = ax.axvspan(cloud_flag_times_dt_single_flight[tmpid[0]],cloud_flag_times_dt_single_flight[tmpid[-1]],\
                                            alpha=0.5,color='deepskyblue',ec=None)
        
        blue_patch = mpatches.Patch(color='deepskyblue',\
                                label='In-Cloud',alpha=0.5)
                                #label='In-Cloud, $N_{c}$ > 0.03 cm$^{-3}$',alpha=0.5)

        lgnd2 = ax1.legend(handles=[blue_patch],fontsize=Fontsize,\
                              bbox_to_anchor=(0.5,1.3),\
                              ncol=1,loc='upper center')
        ax9.axhline(0.03,lw=3,ls='dashed',c='blue')
        ax9.text(fcdp_times_dt[-1]+datetime.timedelta(minutes=5),0.03,\
                 '$N_{c}$ = 0.03 cm$^{-3}$',\
                 c='blue',fontsize=Fontsize*1.25)#transform=ax9.transAxes)
        ax9.axhline(1,lw=3,ls='dashed',c='darkorange')
        ax9.text(fcdp_times_dt[-1]+datetime.timedelta(minutes=5),1.,\
                 '$N_{c}$ = 1 cm$^{-3}$',\
                 c='darkorange',fontsize=Fontsize*1.25)#transform=ax9.transAxes)
        
        
        ax1.axhline(1.e3,lw=3,ls='dashed',c='green')
        ax1.text(fcdp_times_dt[-1]+datetime.timedelta(minutes=5),1.e3,\
                 '10$^{3}$',\
                 c='green',fontsize=Fontsize*1.25)#transform=ax9.transAxes)        
        ax1.axhline(1.e4,lw=3,ls='dashed',c='blue')
        ax1.text(fcdp_times_dt[-1]+datetime.timedelta(minutes=5),1.e4,\
                 '10$^{4}$',\
                 c='blue',fontsize=Fontsize*1.25)#transform=ax9.transAxes)
        ax1.axhline(1.e5,lw=3,ls='dashed',c='darkorange')
        ax1.text(fcdp_times_dt[-1]+datetime.timedelta(minutes=5),1.e5,\
                 '10$^{5}$',\
                 c='darkorange',fontsize=Fontsize*1.25)#transform=ax9.transAxes)
        
        
        ax2.axhline(20,lw=3,ls='dashed',c='green')
        ax2.text(fcdp_times_dt[-1]+datetime.timedelta(minutes=5),20,\
                 '20',\
                 c='green',fontsize=Fontsize*1.25)#transform=ax9.transAxes)        
        ax2.axhline(50,lw=3,ls='dashed',c='blue')
        ax2.text(fcdp_times_dt[-1]+datetime.timedelta(minutes=5),50,\
                 '50',\
                 c='blue',fontsize=Fontsize*1.25)#transform=ax9.transAxes)
        ax2.axhline(100,lw=3,ls='dashed',c='darkorange')
        ax2.text(fcdp_times_dt[-1]+datetime.timedelta(minutes=5),100,\
                 '100',\
                 c='darkorange',fontsize=Fontsize*1.25)#transform=ax9.transAxes)        
        
        ax1.add_artist(lgnd1)

        x_min = aero_times_dt[0]
        x_max = aero_times_dt[-1]
        
        for ax in axlist2:
            ax.xaxis.set_major_formatter(dfmt)
            ax.set_xlim(x_min,x_max)
        

        
        # Plot Date
        tmp_time = aero_date_dt.strftime("%m/%d/%Y")
        plt.figtext(0.5,0.95,'Flight Start Date: '+tmp_time+'\nFlight #{}'.format(str(ii+1)),fontsize=Fontsize*2,ha='center')
        plt.subplots_adjust(hspace=0.55,top=0.91,wspace=3.5)
        
        
        tmp_time = aero_date_dt.strftime("%m_%d_%Y")
        fig_path = '/home/mwstanfo/camp2ex/figures/'
        outfile = 'flight_aerosol_{}_v3.png'.format(tmp_time)
        plt.savefig(fig_path+outfile,dpi=200,bbox_inches='tight')
        plt.close()  

        

print(aaaaaaa)  


keys = [list(N_D_all_flights.keys())[ii] for ii in range(len(N_D_all_flights.keys()))]

print(aaaaa)



