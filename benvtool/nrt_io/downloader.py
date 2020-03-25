"""
Created on Tue Nov 05 13:44:00 2019

A python module which retrieves datasets from various near_realtime_sources
for which no Web-Services are available
@author: nixdorf
"""
#%% import some packages

#packages dealing with time
from datetime import datetime as _datetime
from datetime import timedelta as _timedelta
import time
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrule,HOURLY,DAILY,MONTHLY
#packages for html/ftp communcation, reading
from bs4 import BeautifulSoup
from ftplib import FTP
import urllib.request
import requests
from IPython.display import display, clear_output, HTML
#packages for geoprocessing/projection
import rasterio as _rasterio
from rasterio import warp as _warp
from rasterio.io import MemoryFile
from pyproj import Proj, Transformer,CRS
import geopandas as gpd
from shapely.geometry import Polygon
import wradlib as wrl
from osgeo import gdal

#packages for unzipping things
import tarfile
import bz2
import gzip

#packages for math/db power
import numpy as np
import pandas as pd
from scipy import ndimage
import xarray as xr
from io import BytesIO

#others from standard librarie
from itertools import product
import re
import sys
from PIL import Image
import os

from nrt_io.downmodis import downmodis_mod as _downmodis


#%%
def connect_ftp(server = 'opendata.dwd.de',connected=False):
    while not connected:
        try:
            ftp = FTP(server)
            ftp.login()
            connected = True
    
        except:
            time.sleep(5)
            print('Reconnect to Server')
            pass
    return ftp

# generator to create list of dates
def daterange(start, end, delta):
    """
    generator to create list of dates
    """
    current = start
    while current <= end:
        yield current
        current += delta
# a function which clips through raster provides as numpy array, too        
def buffered_raster_clipping(raster_inpt,
                             shape_inpt=None,
                             raster_transfrm=None,
                             raster_proj=None,
                             buffrcllsz=2,
                             na_value=-1):
    """
    This function clipped a raster dataset with shapefile, even if rasterdata
    is numpy array only
    raster_inpt could be either a georeferenced raster or a numpy array 
    (requires raster_transfrm and raster_crs)
    shape_inpt (currently either string or geodataframe)
    Output:
        r_clip_data: clipped_numpy array
        r_clip_transform: New transform in gdal transform style
        colums: Positions in original dataarray for begin/end clip (column)
        rows: Positions in original dataarray for begin/end clip (row)
    """
    # read the rasterfile
    if isinstance(raster_inpt, str):
        rds = gdal.Open(raster_inpt)
        r_proj = rds.GetProjection()
        r_transform = rds.GetGeoTransform()
        r_data = np.array(rds.GetRasterBand(1).ReadAsArray())
        r_na = rds.GetRasterBand(1).GetNoDataValue()
    else:
        r_data = raster_inpt
        r_proj = raster_proj
        r_transform = raster_transfrm
        r_na = na_value  #
    if isinstance(shape_inpt, str):
        gdfbnd = gpd.read_file(shape_inpt)
    else:
        try:
            gdfbnd = shape_inpt
        except Exception:
            # if no shape is given no clipping will be done:
            print('No readable clipping file defined')
            exit()
    #Unify both CRS systems
    crs_dest=_rasterio.crs.CRS.from_user_input(r_proj)
    crs_shp=_rasterio.crs.CRS.from_user_input(gdfbnd.crs)
    if crs_dest.to_string() != crs_shp.to_string():
        try:
            # if not we simply use projection str
            gdfbnd = gdfbnd.to_crs(crs=crs_dest)
            print('Boundary GIS vectorfile was reprojected to Raster projection')
        except Exception as e:
            print(e)
            print('projection is not provided as crs object')
    else:
        print('projection system is similar')

    #gdfbnd.to_file('test_example.shp')
    # Find the clipping window
    cellsize = abs(min((abs(r_transform[1]), abs(r_transform[5]))))
    BoundsClipDs = np.add(gdfbnd.geometry.total_bounds,
                          np.array([-1, -1, 1, 1]) * buffrcllsz * cellsize)
    # find the positions of the new boundary
    colums = ((
        BoundsClipDs[[0, 2]] - r_transform[0]) / r_transform[1]).astype(int)
    rows = ((
        BoundsClipDs[[1, 3]] - r_transform[3]) / r_transform[5]).astype(int)
    # get the new dataarray (first row than columns)
    r_clip_data = r_data[rows[1]:rows[0], colums[0]:colums[1]]
    #write a new transform
    r_clip_transform = (r_transform[0] + colums[0] * r_transform[1],
                        r_transform[1], 0,
                        r_transform[3] + rows[1] * r_transform[5], 0,
                        r_transform[5])

    return r_clip_data, r_clip_transform, colums, rows

#The Main Downloader Class

class downloader():    
    def __init__(self,start_time='2019-11-01:1500', end_time='2019-11-03:0300',roi='.\\roi_examples\\einzugsgebiet.shp',roi_buffer=2000,**kwargs):
        #get path of this file
        self.path=os.path.dirname(os.path.abspath(__file__))
        self.start_time=_datetime.strptime(start_time,"%Y-%m-%d:%H%M")
        self.end_time=_datetime.strptime(end_time,"%Y-%m-%d:%H%M")
        self.roi=gpd.GeoDataFrame.from_file(roi)
        self.xr=None
        self.roi_bounds=np.add(self.roi.geometry.total_bounds,
                                  np.array([-1, -1, 1, 1]) * roi_buffer).reshape(2,2)
        #get path of this file
        self.path=os.path.dirname(os.path.abspath(__file__))
        #create a rectangle
        sc_line=np.array((self.roi_bounds[0,0],self.roi_bounds[1,1]))
        trd_line=np.array((self.roi_bounds[1,0],self.roi_bounds[0,1]))
        self.roi_bounds_rect=np.concatenate((self.roi_bounds[0,:],sc_line,self.roi_bounds[1,:],trd_line,self.roi_bounds[0,:])).reshape(5,2)
    
        
    # define a reprojection of the whatever data format the string has to epsg 4326
    def reproject(self,init_crs={"init": "EPSG:4326"},dest_crs={"init": "EPSG:4326"}):
        """ 
        a general reprojection for xr datasets to geographic coordinates
        """
        
        # Compute the lon/lat coordinates with rasterio.warp.transform
        ny, nx = len(self.xr["y"]), len(self.xr["x"])
        x, y = np.meshgrid(self.xr["x"], self.xr["y"])
        
        # Rasterio works with 1D arrays
        lon, lat = _warp.transform(init_crs, dest_crs, x.flatten(), y.flatten())
        lon = np.asarray(lon).reshape((ny, nx))
        lat = np.asarray(lat).reshape((ny, nx))
        self.xr.coords["lon"] = (("y", "x"), lon)
        self.xr.coords["lat"] = (("y", "x"), lat)

        
    #download radolan recent only
    def radorecent(self,time_res='hourly',to_harddisk=True,rado_data_dir='.//' + 'radolan' + '//'):
        """
        The radolan downloader using wradlib
        """
        def radobyte_to_array(indat,attrs,NODATA=-9999):
            """ 
            A partial copy from a script from wradlib to process the bytesarray to 
            np_array
            """
            mask = 0xFFF # max value integer
            attrs["nodataflag"] = NODATA
            if attrs['producttype'] in ['RX', 'EX', 'WX']:
                # convert to 8bit integer
                arr = np.frombuffer(indat, np.uint8).astype(np.uint8)
                arr = np.where(arr == 250, NODATA, arr)
                attrs['cluttermask'] = np.where(arr == 249)[0]
            elif attrs['producttype'] in ['PG', 'PC']:
                arr = wrl.io.decode_radolan_runlength_array(indat, attrs)
            else:
                # convert to 16-bit integers
                arr = np.frombuffer(indat, np.uint16).astype(np.uint16)
                # evaluate bits 13, 14, 15 and 16
                attrs['secondary'] = np.where(arr & 0x1000)[0]
                nodata = np.where(arr & 0x2000)[0]
                negative = np.where(arr & 0x4000)[0]
                attrs['cluttermask'] = np.where(arr & 0x8000)[0]
                # mask out the last 4 bits
                arr &= mask
                # consider negative flag if product is RD (differences from adjustment)
                if attrs['producttype'] == 'RD':
                    # NOT TESTED, YET
                    arr[negative] = -arr[negative]
                # apply precision factor
                # this promotes arr to float if precision is float
                arr = arr * attrs['precision']
                # set nodata value
                arr[nodata] = NODATA
        
            # anyway, bring it into right shape
            arr = arr.reshape((attrs['nrow'], attrs['ncol']))
            return arr, attrs   
        #create the radolan_output subdirectory
        if to_harddisk:
            if os.path.exists(rado_data_dir) ==False:
                os.mkdir(rado_data_dir)
        
        #create the daterule depending on the mode, one hour substract due to accum
        if time_res=='hourly':
            dts = [dt.strftime('%y%m%d%H') for dt in daterange(self.start_time,self.end_time, relativedelta(hours=1))]
        
        elif time_res=='daily':
            dts = [dt.strftime('%y%m%d') +'2350' for dt in daterange(self.start_time, self.end_time, relativedelta(days=1))]
        else:
            raise ValueError ('Unknown Timerule')
                
        dts_historical = [dt.strftime('%Y%m')for dt in daterange(self.start_time, self.end_time, relativedelta(days=1))]
        dts_historical=list(set(dts_historical))
        years = list(range(self.start_time.year, self.end_time.year + 1))
        
        #define the radolan_projection
        rado_proj_string='+proj=stere +lat_0=90 +lat_ts=90 +lon_0=10 +k=0.93301270189 + x_0=0 +y_0=0 +a=6370040 +b=6370040 +to_meter=1000 +no_defs'
        rado_proj = CRS(rado_proj_string)
        #define a transformer to deal with the differenct CRS
        transformer = Transformer.from_crs(self.roi.crs, rado_proj, always_xy=True)
        #convert the bounds str to radolan projection
        roi_bounds_proj=np.array([transformer.transform(*xy) for xy in self.roi_bounds]) 
        # Connect to the Server
        server='opendata.dwd.de'
        ftp=connect_ftp(server = server,connected = False)        

        initDf=True
        for year in years:
            #check whether data is recent
            if year == _datetime.now().year:
                ftp.cwd('/climate_environment/CDC/grids_germany/' + time_res +'/radolan/recent/bin/')
        
                files = ftp.nlst()
                for dt, file in product(dts, files):
                    if dt in file:
                        print('Retrieving {}...'.format(file))
                        retrieved = False
                        archive = BytesIO()
                        # try to retrieve file
                        while not retrieved:
                            try:
                                ftp.retrbinary("RETR " + file, archive.write)
                                retrieved = True
                            except:
                                ftp=connect_ftp(server = server,connected = False)
                                ftp.cwd(
                                    '/climate_environment/CDC/grids_germany/'+ time_res +'/radolan/recent/bin/'
                                )   
                        archive.seek(0)
                        rado_binary = gzip.open(archive)
                        # rewind file
                        rado_binary.seek(0, 0)
                        header = wrl.io.radolan.read_radolan_header(rado_binary)
                        attrs=wrl.io.parse_dwd_composite_header(header)
                        indat = wrl.io.radolan.read_radolan_binary_array(rado_binary, attrs['datasize'])
                        arr, attrs=radobyte_to_array(indat,attrs)
                        # as we are in initial conditions we further create a XArray in WGS84
                        #get_radolan_grid(nrows=None, ncols=None, trig=False, wgs84=False):
                        rado_xr_raw=wrl.io.radolan.radolan_to_xarray(arr,attrs)
                        # use the great selection tool to get the relevant slice
                        rado_xr_select=rado_xr_raw.sel(x=slice(roi_bounds_proj[0,0],roi_bounds_proj[1,0]),
                                                                  y=slice(roi_bounds_proj[0,1],roi_bounds_proj[1,1]))
                        #if initial conditon we create final xarray, otherwise we append
                        if initDf:
                            rado_xr=rado_xr_select
                            initDf=False
                        else:
                            rado_xr=xr.concat([rado_xr,rado_xr_select], dim='time')
                        print('Processing {}...finished'.format(file))
            else:               
                ftp.cwd('/climate_environment/CDC/grids_germany/'+ time_res +'/radolan/historical/bin/'+str(year)+'/')
                files = ftp.nlst()
                for dt, file in product(dts_historical, files):
                    if dt in file:
                        # if not downloaded yed we download it if required
                        if to_harddisk:
                            if file not in os.listdir(rado_data_dir):
                                
                                print('download{}...'.format(file))
                                retrieved = False
                                # try to retrieve file
                                while not retrieved:
                                    try:
                                        with open(rado_data_dir+file, 'wb') as f:
                                            ftp.retrbinary("RETR " + file, f.write)
                                            retrieved = True
                                    except:
                                        print('reconnect to ftp')
                                        ftp=connect_ftp(server = server,connected = False)
                                        ftp.cwd('/climate_environment/CDC/grids_germany/'+ time_res +'/radolan/historical/bin/'+str(year)+'/')
                                print('download{}...done'.format(file))
                            archive_monthly = tarfile.open(rado_data_dir+file,"r:gz")
                        #otherwise we do everything from memory
                        else:
                            print('Retrieving {}...'.format(file))
                            retrieved = False
                            archive_month = BytesIO()
                            # try to retrieve file
                            while not retrieved:
                                try:
                                    ftp.retrbinary("RETR " + file, archive_month.write)
                                    retrieved = True
                                except:
                                    print('reconnect to ftp')
                                    ftp=connect_ftp(server = server,connected = False)
                                    ftp.cwd('/climate_environment/CDC/grids_germany/'+ time_res +'/radolan/historical/bin/'+str(year)+'/')
                            archive_month.seek(0)
                            archive_monthly = tarfile.open(fileobj=archive_month)
                        #sometimes the tar in tar.gz causes problems, which means we have to unzip again
                        if len(archive_monthly.getnames())<2:
                            archive_monthly=archive_monthly.extractfile(archive_monthly.getnames()[0])
                            archive_monthly.seek(0, 0)
                            archive_monthly = tarfile.open(fileobj=archive_monthly)
                        for dt, file in product(dts, archive_monthly.getnames()):
                            if dt in file:
                                try:
                                    rado_binary = gzip.open(archive_monthly.extractfile(file))
                                    rado_binary.seek(0, 0)
                                    header = wrl.io.radolan.read_radolan_header(rado_binary)
                                    attrs=wrl.io.parse_dwd_composite_header(header)
                                    indat = wrl.io.radolan.read_radolan_binary_array(rado_binary, attrs['datasize'])
                                    arr, attrs=radobyte_to_array(indat,attrs)
                                except Exception as e:
                                    print(e)
                                    print('try to open another directly without gzip')
                                    rado_binary = archive_monthly.extractfile(file)
                                    header = wrl.io.radolan.read_radolan_header(rado_binary)
                                    attrs=wrl.io.parse_dwd_composite_header(header)
                                    indat = wrl.io.radolan.read_radolan_binary_array(rado_binary, attrs['datasize'])
                                    arr, attrs=radobyte_to_array(indat,attrs)
                                # as we are in initial conditions we further create a XArray in WGS84
                                #get_radolan_grid(nrows=None, ncols=None, trig=False, wgs84=False):
                                rado_xr_raw=wrl.io.radolan.radolan_to_xarray(arr,attrs)
                                # use the great selection tool to get the relevant slice
                                rado_xr_select=rado_xr_raw.sel(x=slice(roi_bounds_proj[0,0],roi_bounds_proj[1,0]),
                                                                          y=slice(roi_bounds_proj[0,1],roi_bounds_proj[1,1]))
                                #if initial conditon we create final xarray, otherwise we append
                                if initDf:
                                    rado_xr=rado_xr_select
                                    initDf=False
                                else:
                                    rado_xr=xr.concat([rado_xr,rado_xr_select], dim='time')
                                print('Processing {}...finished'.format(file))
        #close ftp connection
        try:
            ftp.quit()
        except Exception as e:
            print(e)        
        # add projection string
        rado_xr.attrs['crs']=rado_proj_string

        #attach to class itself
        self.xr=rado_xr
        
        #return x_array
        return rado_xr
    
    def weatherprediction_parameters(self,nwp_model='cosmo-d2'):
            """
            Gives you a list of available parameters from the model 
            """
            # connect to ftp
            ftp=connect_ftp(server = 'opendata.dwd.de',connected = False)
            # go to dir
            ftp.cwd('weather/nwp/' + nwp_model + '/grib/00/')
            nwp_parameters=ftp.nlst()
            return nwp_parameters
    
    
    def weatherprediction(self,nwp_model='cosmo-d2',bufferhours=4,parameter='tot_prec'):
        """
        Numerical Weather Prediction Reader and Processing for ICON Model and COSMO Model
        
        ICON BASIC INFORMATION
        
        Die Vorhersagen des Regionalmodell ICON-EU werden aus den vier Modellläufen um 00, 06, 12, und 18 UTC 
        bis +120 Stunden bereitgestellt und aus den Modelläufen um 03, 09, 15 und 21 UTC bis +30 Stunden. 
        Für den Vorhersagezeitraum bis + 78 Stunden sind einstündige Zeitschritte verfügbar, 
        von +81 Stunden bis +120 Stunden dreistündige Zeitschritte
        
        COSMO BASIC INFORMATION:
    
            Aufgrund der Zielstellung des COSMO-D2 (und des kleinen Modellgebiets)
            sind auch nur relativ kurze Vorhersagezeiten sinnvoll. 
            Vom COSMO-D2 werden alle 3 Stunden, ausgehend von 00, 06, 09, 12, 15, 18 und 21 UTC, 
            neue +27-Stunden-Vorhersagen bereitgestellt. Der Modelllauf von 03 UTC liefert 
            sogar einen Vorhersagezeitraum von +45 Stunden
        
        """
            
        print('Start retrieving NWP data for modell',nwp_model)
        initDf = True
        # Get the current time in order to choose appropriate forecast directory
        local_utctime=_datetime.utcnow()
        #available runs 
        if nwp_model=='icon-eu':
            runtms=np.array((0,6,12,18))
        else:
            if nwp_model=='cosmo-d2':
               runtms=np.array((0,6,9,12,15,18,21))
            else:
               sys.exit('Unknown climate model have been chosen, only icon-eu and cosmo-d2 are integrated')
        #calculate nearest run with a buffer for simulation time of model
        delta_hours=list(runtms-(local_utctime.hour-bufferhours))
        smallest_hour=max(i for i in delta_hours if i < 0)
        runtm=runtms[delta_hours.index(smallest_hour)]
        
        # Connect to ftp
        ftp=connect_ftp(server = 'opendata.dwd.de',connected = False)
        if len(str(runtm))<2:
            ftp.cwd('weather/nwp/' + nwp_model + '/grib/'+'0'+str(runtm)+'/'+parameter+'/')
        else:
            ftp.cwd('weather/nwp/' + nwp_model + '/grib/'+str(runtm)+'/'+parameter+'/')   
        files_raw = ftp.nlst()
        #get the grid data only
        files=[i for i in files_raw if 'regular' in i]
        # if cosmo model the last time step is always crap
        if nwp_model=='cosmo-d2':
            files.remove(files[-1])
        
        # loop through the files
        for file in files:
            print('Retrieving {}...'.format(file))
            retrieved=False
            archive = BytesIO()
            # try to retrieve file
            while not retrieved:
                try:
                    ftp.retrbinary("RETR " + file, archive.write)
                    retrieved=True
                except:
                    ftp=connect_ftp(server = 'opendata.dwd.de',connected = False)
                    if len(str(runtm))<2:
                        ftp.cwd('weather/nwp/' + nwp_model + '/grib/'+'0'+str(runtm)+'/tot_prec/')
                    else:
                        ftp.cwd('weather/nwp/' + nwp_model + '/grib/'+str(runtm)+'/tot_prec/')  
                    
            archive.seek(0)
            grib_data = bz2.decompress(archive.read())
            with MemoryFile(grib_data) as memfile:
                dataset=memfile.open()
            # open the xarray
            nwp_xr_raw=xr.open_rasterio(dataset)
            # accumulate over bands in order to get hourly values
            nwp_xr_merge=nwp_xr_raw.sum(dim='band')
            # add the attributes again
            nwp_xr_merge.attrs=nwp_xr_raw.attrs
            # fix multiband to singleband attr
            nwp_xr_merge.attrs['scales']=nwp_xr_merge.attrs['scales'][0]
            nwp_xr_merge.attrs['nodatavals']= nwp_xr_merge.attrs['nodatavals'][0]*len(nwp_xr_raw)
            nwp_xr_merge.attrs['offsets']=nwp_xr_merge.attrs['offsets'][0]
            nwp_xr_merge.attrs['descriptions']=parameter
            # replace na with actual numpy na
            nwp_xr_merge = nwp_xr_merge.where(nwp_xr_merge != nwp_xr_merge.attrs['nodatavals'], np.nan)
            # add a time stamp as coordinate
            #get the correct date attached to the simulation
            # get the numbers from filename
            filenm_nmbrs=re.findall(r'\d+', file)
            # if cosmo-2, we remove first number
            if nwp_model=='cosmo-d2':
                del(filenm_nmbrs[0])
            sim_starttime=_datetime.strptime(filenm_nmbrs[0], '%Y%m%d%H')
            sim_date=sim_starttime+ _timedelta(hours=int(filenm_nmbrs[1]))
            nwp_xr_merge['time']=sim_date
            #if initial conditon we define bounds and final xarray, otherwise we append
            if initDf:
            # get the bounds in nwp coordinate system
                roi_bounds_proj=np.array([transform(Proj(self.roi.crs),Proj(nwp_xr_merge.crs),*xy) for xy in self.roi_bounds])
            # get slice
            nwp_xr_sliced=nwp_xr_merge.sel(x=slice(roi_bounds_proj[0,0],roi_bounds_proj[1,0]),
                                                              y=slice(roi_bounds_proj[1,1],roi_bounds_proj[0,1]))
            #if initial conditon we create final xarray, otherwise we append
            if initDf:
                nwp_start=nwp_xr_sliced
                nwp_xr=nwp_start
                initDf=False
            else:
                nwp_xr=xr.concat([nwp_xr,nwp_xr_sliced], dim='time')
                print('Processing {}...finished'.format(file))
                # repair the nodatavals
                nwp_xr.attrs['nodatavals'] = nwp_xr_merge.attrs['nodatavals']
    # if total precipitation is chosen we substract current from previous array
        if parameter == 'tot_prec':
            print('accumulated rainfaill is recalculated to rainfall per timestep')
            nwp_xr_diff=nwp_xr.diff(dim='time',n=1,label='upper')
            nwp_xr_diff.attrs=nwp_xr.attrs
            # merge with the first initial one
            nwp_xr=xr.concat([nwp_start,nwp_xr_diff], dim='time')
            nwp_xr.attrs['descriptions']='Precipitation per time step'
        
        #attach to class itself
        self.xr=nwp_xr
        
        #return x_array
        return nwp_xr
    
    
    def mhmsoilmoist(self,outer_shape='.\\dbases\\DEU_adm0.shp', fname='nFK_0_25_daily_n14.gif', crs='epsg:4326'):
        """
        a tool which extracts NRT Soil Moisture Maps from the mhm model as GIF image,
        reprojection string is hard coded  and a map of Germany for image clipping
        has to be provided
        Projection system is defined as EPSG 4326
        """
        
        def ffill(arr, mask, mode='nd_max', footprint_size=3):
            """
            different functions to fill holes in 2D array, position of holes is identified by ~mask values
            """
            if mode == 'acc':  # row or column based max value is used for filling holes
                idx = np.where(~mask, np.arange(mask.shape[1]), 0)
                np.maximum.accumulate(idx, axis=1, out=idx)
                out = arr[np.arange(idx.shape[0])[:, None], idx]
            if mode == 'nn':  #interpolate 1D by next neighbor by https://bit.ly/2QGvbQo
                arr[mask] = np.interp(
                    np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
                out = arr
            if mode == 'nd_max':  # inspired by @ https://bit.ly/31q9Miz
                arr_mask = arr.copy()
                arr_mask = arr_mask * mask
                arr_mask = arr_mask.astype(float)
                arr_mask[arr_mask == 0] = np.nan
                mask_int = np.ones((footprint_size, footprint_size))
                mask_int[1, 1] = 0
                arr_int = ndimage.generic_filter(
                    arr_mask,
                    np.nanmax,
                    footprint=mask_int,
                    mode='constant',
                    cval=np.nan)
                #replace only the indices where arr is nan by the interpolated ones
                arr[~mask] = arr_int[~mask]
                out = arr.astype(int)
                out[out < 0] = 0
            return out
        
        #%% Main functionality starts        
        inpt_proj=Proj(self.roi.crs)
        #convert the bounds str to radolan projection
        roi_bounds_proj=np.array([transform(inpt_proj,Proj(crs),*xy) for xy in self.roi_bounds])
        
        
        #%% First some standard defintions of the SM Files
        # Define the colorcode_sm
        color_code_RGB = np.array(
            ([230, 0, 0], [255, 170, 0], [252, 211, 127], [242, 242, 242],
             [230, 230, 230], [217, 217, 217], [189, 235, 191], [90, 204, 95],
             [8, 168, 30], [5, 101, 120], [0, 0, 255]))
        color_code_grey = np.array((106, 0, 157, 253, 248, 240, 210, 126, 118, 75,
                                    165))
        #define corresponding bandmean of soil moisture
        sm_code = np.array((5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 97.5))
        #define the image_transform. If dont know it, do georeferencing first
        # In case of north up images, the GT(2) and GT(4) coefficients are zero,
        # and the GT(1) is pixel width, and GT(5) is pixel height. The (GT(0),GT(3))
        #  position is the top left corner of the top left pixel of raster.
        #for WGS84
        image_transform = ((4.756996182800601, 0.016483987942747476, 0.0,
                            55.10523733895226, 0.0, -0.010363875929350508))
        #the Resolution of the mhm modell
        mhm_res = (175, 225)
        #%% Download the file from `url` and save it locally under `file_name`:
        #connect to server
        con = urllib.request.urlopen('https://files.ufz.de/~drought/' + fname)
        #get last modification date and add 6 hours to be sure to be on same date
        time_struct = _datetime.strptime(
        con.headers['last-modified'],
        '%a, %d %b %Y %H:%M:%S %Z') + _timedelta(hours=6)
                                                 
        obs_enddate = time_struct - _timedelta(days=1)
        obs_startdate = time_struct - _timedelta(days=14)
        
        # open the image directly from URL
        im = Image.open(
            urllib.request.urlopen('https://files.ufz.de/~drought/' + fname))
        
        #small correction of image transform from visual inspection
        image_transform = np.array(image_transform)
        image_transform[3] = image_transform[3] + 0.1
        image_transform = tuple(image_transform)
        
        #%% clip the data to the content of Germany as image has some left and right whitespace
        pix_clip_data, pix_clip_transform, cols, rows = buffered_raster_clipping(
            np.array(im),
            outer_shape,
            raster_transfrm=image_transform,
            raster_proj=crs,
            buffrcllsz=0)
        
        #%% Create our x and y coordinates
        cellwidth = pix_clip_transform[1] * pix_clip_data.shape[1] / mhm_res[0]
        cellheight = pix_clip_transform[5] * pix_clip_data.shape[0] / mhm_res[1]
        #create coordinates
        x=np.linspace(pix_clip_transform[0],pix_clip_transform[0]+cellwidth*(mhm_res[0]-1),mhm_res[0])
        y=np.linspace(pix_clip_transform[3],pix_clip_transform[3]+cellheight*(mhm_res[1]-1),mhm_res[1])
        
        # start our main loop trough the gif file
        nframes = 0
        InitDf=True
        for single_date in daterange(obs_startdate, obs_enddate,relativedelta(days=1)):
            print('Extract mhm nrt soil moisture at date', single_date.strftime("%Y-%m-%d"))
            im_rgb = im.convert('RGB')
            pix_rgb = np.array(im_rgb)
            pix_rgb_clip = pix_rgb[rows[1]:rows[0], cols[0]:cols[1], :]
            im_clip = Image.fromarray(pix_rgb_clip)
            #%% do the initial processing
            # reduce size of image to mhm size
            im_resized = im_clip.resize(mhm_res)
            # find relevant colors in RGB mode
            im_rgb = im_resized.convert('RGB')
            pix_rgb = np.array(im_rgb)
            #we create the mask by combine logical ands
            mask = np.full(pix_rgb.shape[:2], False)
            for rgb_color in color_code_RGB:
                color_mask = np.logical_and(
                    np.logical_and(pix_rgb[:, :, 0] == rgb_color[0],
                                   pix_rgb[:, :, 1] == rgb_color[1]),
                    pix_rgb[:, :, 2] == rgb_color[2])
                mask = np.logical_or(mask, color_mask)
            # Now we do the part which needs to be done for all files
            # get pixels from grey_scale picture and fill nans
            pix_filled = pix_rgb.copy()
            #for i in range(0,3):
            #    pix_filled[:,:,i]=ffill(pix_filled[:,:,i],mask,mode='nd_max',footprint_size=3)
            # an array which is used to calculated the distance to each color from the map
            pix_dist = np.ones((pix_filled.shape[0], pix_filled.shape[1],
                                len(color_code_RGB))) * 256
            for i in range(0, len(color_code_RGB)):
                pix_dist[:, :, i] = np.abs(
                    pix_filled[:, :, 0] - color_code_RGB[i, 0]) + np.abs(
                        pix_filled[:, :, 1] - color_code_RGB[i, 1]) + np.abs(
                            pix_filled[:, :, 2] - color_code_RGB[i, 2])
            #get the color which is closest by
            nearest_colors = np.argmin(pix_dist, axis=2, out=None)
            #assign the sm values
            pix_sm = nearest_colors.copy()
            for i in range(0, len(color_code_grey)):
                pix_sm[pix_sm == i] = sm_code[i]
            #for i in range(0,3):
            pix_sm = ffill(pix_sm, mask, mode='nd_max', footprint_size=3)
            
            day=_datetime(single_date.year,single_date.month,single_date.day)
            #create the data array
            sm_xr_raw=xr.DataArray(pix_sm, coords=[y,x], dims=['y','x'])
            #attach time
            sm_xr_raw.coords['time']=day
            #extend by time
            
            # choose parameter based on selection
            sm_xr_select=sm_xr_raw.sel(y=slice(roi_bounds_proj[1,0],roi_bounds_proj[0,0]),
                                               x=slice(roi_bounds_proj[0,1],roi_bounds_proj[1,1]))
            if InitDf:
                sm_xr=sm_xr_select
                InitDf=False
            else:
                sm_xr=xr.concat([sm_xr,sm_xr_select], dim='time')
        
            nframes += 1
            try:
                im.seek(nframes)
            except EOFError:
                break
        #attach crs
        sm_xr.attrs['crs']=Proj(crs).to_latlong_def()
        
        #attach to class itself
        self.xr=sm_xr
        
        #return x_array
        return sm_xr
    
    
    def modis_products(self,
                   server='https://n5eil01u.ecs.nsidc.org',
                   modis_product="MOD14A1.006",
                   product_parameters=['FireMask', 'MaxFRP'],
                   modis_user=None,
                   modis_pw=None,
                   delete_files=True,
                   modis_data_dir='.\\' + 'temp'):
        """
        A Function to download specific modis products from the http server
        for each product, the script downmodis is used in a mod version (https://bit.ly/2Ot79JF)
        
        WARNING: Date detection is hard coded with file.find('.A2')        
        
        WARNING: NODE FOR MYSELF, after several trys I am still not able to get a hdf4 file 
        from stream, hence, unfortunately things have still to be downloaded to temp dir
        # some suggestions for geotiff export:
        https://github.com/robintw/XArrayAndRasterio
        
        Currently no parameter check involved
        
        Server could be either 'https://n5eil01u.ecs.nsidc.org' or 
        https://e4ftl01.cr.usgs.gov
        """
        # fix some library things
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        #import downmodis_mod as _downmodis
        
        # create temp folder for the downloaded files
        dest_dir = modis_data_dir
        #Create temp directory to store data
        if(not os.path.exists(dest_dir)):
            os.makedirs(dest_dir)
        #tile file
        tile_file=self.path + '\\dbases\\sn_bound_10deg.txt'
        #import datalist of modis products
        modis_products=pd.read_csv(self.path + '\\dbases\\modis_products.csv')
        
        
        def find_tile(lat,lon,tile_file):
            data = np.genfromtxt(tile_file, 
                             skip_header = 7, 
                             skip_footer = 3)
            in_tile = False
            i = 0
            while(not in_tile):
                in_tile = lat >= data[i, 4] and lat <= data[i, 5] and lon >= data[i, 2] and lon <= data[i, 3]
                i += 1
            
            vert = int(data[i-1, 0])
            horiz = int(data[i-1, 1])
            #make a string depending on the 0 issue
            if vert <10:
                if horiz<10:
                   tile_str='h0'+str(horiz)+'v0'+str(vert)
                else:
                   tile_str='h'+str(horiz)+'v0'+str(vert)
            else:
                if horiz<10:
                   tile_str='h0'+str(horiz)+'v'+str(vert)
                else:
                   tile_str='h'+str(horiz)+'v'+str(vert)
                
            return tile_str
        
        
        def append_hdf_to_xarray(file_dir,hdf_filename,hdf_parameters,x_array_ds,initialize=True):
            """
            A tool which appends data from a hdf file to an xarray dataset
            """
            filepath=file_dir+'\\'+hdf_filename
            #first we extract the date which is hard coded
            retrieve_date=_datetime.strptime(hdf_filename[hdf_filename.find('.A2')+2:hdf_filename.find('.A2')+9],'%Y%j')                        
            with _rasterio.open(filepath) as hdf_reader:
                #get subdatasets
                subsets=hdf_reader.subdatasets
                #check subsets whether parameter is inside and add to xarray
                for subset in subsets:                
                    # check whether the dataset matches with requested parameters
                    if any(parameter in subset for parameter in hdf_parameters):
                        #shorten the name
                        subset_name=subset.split(':')[-1]                    
                    # load the correct parameter dataset as xarray
                        xr_array=xr.open_rasterio(subset)
                        # if initializing we need to convert our rectangular data to modis format for slicing
                        if initialize:
                            bnds_x, bnds_y = _warp.transform(self.roi.crs,xr_array.crs, self.roi_bounds_rect[:-1,0], self.roi_bounds_rect[:-1,1])
                            initialize=False
                        #take selection only                    
                        xr_array=xr_array.sel(x=slice(min(bnds_x),max(bnds_x)),y=slice(max(bnds_y),min(bnds_y)))
                        #drop dimension band
                        xr_array=xr_array.squeeze()
                        # delete band if only one is there
                        if xr_array.band.size==1:
                            xr_array=xr_array.drop('band')
                        #add name
                        xr_array.name=subset_name
                        # add dimension time
                        xr_array['time']=retrieve_date
                        xr_array=xr_array.expand_dims('time')                                    
                        #delete units attribute as causes errors by 
                        #open netcdf may due to addition of dim time
                        try:
                            del xr_array.attrs['units']
                        except:
                            pass
                        # merge to dataset
                        x_array_ds=xr.merge([x_array_ds,xr_array])
            return x_array_ds
        #%% next is the registry information, best is to use today and enddate , enddate is opposite down
        
        
        #%% Start Modis Procedure by doing some checks on input data
        print('Checking MODIS data query setting...')
        tiles=list()
        #Spatial extention by check the files
        #convert to lat lon
        bnds_lon, bnds_lat = _warp.transform(self.roi.crs,'epsg:4326', self.roi_bounds_rect[:,0], self.roi_bounds_rect[:,1])
        #check tiles
        for bnd_id in range(0,len(bnds_lon)):
            tile=find_tile(bnds_lat[bnd_id],bnds_lon[bnd_id],tile_file)
            #add to list if not already there
            if tile not in tiles:
                print('MODIS Tile no.', tile, ' is requested')
                tiles.append(tile)
        
        #convert the time to MODIS format, modis downloads reverse order        
        day = self.end_time.strftime('%Y-%m-%d')
        enddate = self.start_time.strftime('%Y-%m-%d')
        
        #next is that we check whether the specified product exist
        pd_substrs=modis_products['Short_Name'].str.find(modis_product)
        #if found select row_id, otherwise we return with none
        if any(pd_substrs>-1):
            print('MODIS product',  modis_product, 'is requested')
            product_id=int(pd_substrs.idxmax())
            product_path=modis_products['Path'][product_id]
        else:
            print('No available MODIS product found')
            #return None
        
        #next we check whether the selected parameters exist--> DONE IN FUTURE
        print('CURRENTLY NO PARAMETER CHECK IMPLEMENTED')
        
        #check whether a user and password are defined
        if modis_user is None or modis_pw is None:
            print('No attribute ''modis user'' or ''modis_pw'' found, add manually')
            print('Registration can be done on https://earthdata.nasa.gov/')
            modis_user=input('username : ')
            modis_pw=input('password : ')
        
        print('Checking MODIS data query setting...done')
        #%%Now we can can create a downmodis object using these given parameters.
        if server=='https://e4ftl01.cr.usgs.gov':
            print('Connect to MODIS Server... https://e4ftl01.cr.usgs.gov')
            modis = _downmodis.downModis(destinationFolder=dest_dir, product=modis_product,today=day,enddate=enddate, tiles=tiles, path=product_path, user=modis_user, password=modis_pw)
            #modis = _downmodis.downModis(destinationFolder=dest_dir, product=modis_product,today=day, tiles=tiles, path=product_path, user=modis_user, password=modis_pw)
            modis.getFilesList()
            modis.connect(ncon=10)
            print('Connect to MODIS Server...done')
            initDf=True
            print('Downloading MODIS files')
            # create an empty dataset for spatial join and for parameter join
            xr_ds=xr.Dataset()
            # get the list of days with data
            modis_days=modis.getListDays()
            for modis_day in modis_days:
                # obtain list of all files
                listAllFiles = modis.getFilesList(modis_day)
                #clean to hdf only
                hdf_files=[i for i in listAllFiles if ".xml" not in i]
                # download each tile individually if it is not in dataset already
                for file in hdf_files:
                    # check whether file is downloaded already, otherwise download it
                    if file not in os.listdir(dest_dir):
                        tile=file.split('.')[2]
                        print('download file', file,'...')
                        modis.downloadFile(file,dest_dir+'\\'+file, modis_day, bytestream=False)
                        print('download file', file,'...finished ')
                    else:
                        print(file,' already in path ',dest_dir)
                    # convert to xarray
                    if initDf:
                        xr_ds=append_hdf_to_xarray(dest_dir+'\\',file,product_parameters,xr_ds,initialize=initDf)
                    else:
                        initDf=False
                        xr_ds=append_hdf_to_xarray(dest_dir+'\\',file,product_parameters,xr_ds,initialize=initDf)
                    #Try to delete all files in the temporary folder
                    if delete_files==True:
                        for file in os.listdir(dest_dir):
                            try:
                                os.remove(dest_dir+'\\'+file)
                            except:
                                pass
            print('Downloading MODIS files...done')
            
            
        elif server=='https://n5eil01u.ecs.nsidc.org':
            #url path has to be adapted
            server_paths={'MOLT': 'MOST','MOLA':'MOSA','MOTA':None}
            #change to new server structure
            if product_path=='MOTA':
                print('required dataset not in this server, us other server!')
            else:
                product_path=server_paths[product_path]
            print('Connect to MODIS Server...https://n5eil01u.ecs.nsidc.org')
            initDf=True            
            # create an empty dataset for spatial join and for parameter join
            xr_ds=xr.Dataset()
            
            
            class EarthdataLogin(requests.Session):
                """
                Prompt user for Earthdata credentials repeatedly until auth success. Source:
                https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+Python
                """
                AUTH_HOST = "urs.earthdata.nasa.gov"              # urs login url    
                ERROR = "Login failed ({0}). Retry or register."  # failure message   
                TEST = ("https://daac.ornl.gov/daacdata/")        # test authentication;    
                REGISTER = HTML(                                  # registration prompt
                    "<p style='font-weight:bold'><a href=https://urs.earth"
                    "data.nasa.gov/users/new target='_blank'>Click here to"
                    " register a NASA Earthdata account.</a></p>")
                                          
                
                def __init__(self):
                    fails = 0     
                    while True:
                        display(self.REGISTER)                     # register prompt
                        username =modis_user
                        password = modis_pw      
                        if sys.version_info.major==2:              # init requests session
                            super(EarthdataLogin, self).__init__() # for Python 2
                        else:
                            super().__init__()                     # for Python 3
                        self.auth = (username, password)           # add username,password         
                        try:                                     
                            response = self.get(self.TEST)         # try to grab TEST
                            response.raise_for_status()            # raise for status>400
                            clear_output()                         # clear output
                            display("Login successful. Download with: session.get(url)")
                            break
                        except requests.exceptions.HTTPError as e:
                            print(e)
                            clear_output()                         # clear cell output
                            fails += 1                             # +1 fail counter
                            display(self.ERROR.format(str(fails))) # print failure msg

    
                def rebuild_auth(self, prepared_request, response):
                    """
                    Overrides from the library to keep headers when redirected to or 
                    from the NASA auth host.
                    """
                    
                    headers = prepared_request.headers
                    url = prepared_request.url
                    if 'Authorization' in headers:
                        original_parsed = requests.utils.urlparse(response.request.url)
                        redirect_parsed = requests.utils.urlparse(url)
                        
                        if (original_parsed.hostname != redirect_parsed.hostname) and \
                                redirect_parsed.hostname != self.AUTH_HOST and \
                                original_parsed.hostname != self.AUTH_HOST:
                            del headers['Authorization']                   
                    self.auth = None   # drop username/password attributes
                    return             # return requests.Session
        
            session = EarthdataLogin()
            dates = [_datetime.strftime(self.start_time + _timedelta(days=x),'%Y.%m.%d') for x in range((self.end_time-self.start_time).days + 1)]
            for date in dates:
                url = 'https://n5eil01u.ecs.nsidc.org/'+ product_path + '/' + modis_product + '/' + date + '/'
                try:
                    source = session.get(url)
                except Exception as e:
                    print(e)
                    print('No data for specified date')
                soup = BeautifulSoup(source.text, "html.parser")
                
                for link in soup.findAll('a'):
                    link_txt = link.text.strip()
                    if tile in link_txt and link_txt.endswith('.hdf') :
                        print (link_txt)
                        download_link = url + link_txt
                        r= session.get(download_link,stream=True)
                        if r.status_code == 200:
                            print('Downloading MODIS files')
                            dest_dir+'\\'+link_txt
                            # write file out
                            if os.path.exists(dest_dir+'\\') ==False:
                                os.mkdir(dest_dir+'\\')
                            with open(dest_dir+'\\'+link_txt, 'wb') as f:
                                for chunk in r:
                                    f.write(chunk)
                            #create the xarray
                            if initDf:
                                xr_ds=append_hdf_to_xarray(dest_dir+'\\',link_txt,product_parameters,xr_ds,initialize=initDf)
                            else:
                                initDf=False
                                xr_ds=append_hdf_to_xarray(dest_dir+'\\',link_txt,product_parameters,xr_ds,initialize=initDf)                                
                            #delete the file
                            os.remove(dest_dir+'\\'+link_txt)
                            
                        else:
                            print('Download error')                                
                            break
        
        
        
        else:
            print('Please choose either https://e4ftl01.cr.usgs.gov or https://n5eil01u.ecs.nsidc.org as server')
            sys.exit()
        #attach to class itself
        self.xr=xr_ds
        
        return xr_ds 


    def satellite_coverage(self,satellite_database='.\dbases\satellite_database.csv',satellite_name='Sentinel-2A',
                     satellite_instrument='-'):
        # -*- coding: utf-8 -*-
        """
        Created on Tue Oct  1 17:51:25 2019
        A tool which uses data from heaven-above and some geometrical processing to
        assign at which overfly of a satellite the region of interest (roi) provided by
        the user 
        Satellite to choose
        Aqua, Aura, CALIPSO, CBERS-4, CloudSAT, GPM, Jason-3, Landsat-7, Landsat-8, NOAA-20
        NOAA-18, NOAA-19, Proba-V, Sentinel-1A, Sentinel-1B, Sentinel-2A, Sentinel-2B
        Sentinel-3A, Sentinel-3B, Suomi NPP, Terra 
        
        [Potential bug  : Swath angle too low that the satellite passes the vertex twice within 2 hours]
        One example happens in NOAA-18, HIRS-4 (High Resolution Infrared Radiation Sounder), with swath angle = 38.85°
        URL: https://www.heavens-above.com/PassSummary.aspx?satid=28654&lat=53.059706232696044&lng=13.8689581095596&loc=Unspecified&alt=0&tz=UCT&showall=t
        Nov 14 18:17:04 (39°), 19:57:42 (44°)
        One temporary solution was added to remove repeated points in line 237 : v = list(dict.fromkeys(v)) 
        But correct solution is to convert 30 min interval instead of 2 hour interval as key by using datetime library
        
        __author__ = "Gloria Kwok, Erik Nixdorf"
        __propertyof__ = "Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ. "
        __email__ = "erik.nixdorf@ufz.de"
        __version__ = "0.1"

        """


        #%% FUNCTION DEFINITION: IF ROI is covered by sat swath only partially,
        # we need two formulaes to update out-of-swath vertices to new boundary vertices
        # If number of outlying vertices is more than 1, which is a more common case, get_new_coordinate1() would be used
        # If number of outlying vertices is just 1, get_new_coordinate2() would be used
    
        def get_new_coordinate1(miss_pos):  # out-of-swath vertices >1
            # get vertex order and altitude of the just out-of-swath vertex
            print(miss_pos)
            miss_vertex = vt[miss_pos]
            miss_altitude = altitude_dict[k + '_' + str(miss_pos)]
    
            # check if vertex order is 0 or max to prevent 'list out of range'
            check_pos = miss_pos - 1
            if check_pos == -1:
                check_pos = len(vt) - 1
            check_pos2 = miss_pos + 1
            if check_pos2 == len(vt):
                check_pos2 = 0
    
            # for direction purpose so to make sure nearby vertex for comparison is the one in swath, not not the one that out of swath
            if altitude_dict[k + '_' + str(check_pos)] > float(
                    maximum_angle):  # make sure minus the correct order
                nearby_vertex = vt[check_pos]
                nearby_altitude = altitude_dict[k + '_' + str(check_pos)]
            else:
                nearby_vertex = vt[check_pos2]
                nearby_altitude = altitude_dict[k + '_' + str(check_pos2)]
    
            # update the new vertex by 'altitude proportion formulae'
            new_vertex = ['', '']
            proportion = (float(maximum_angle) - miss_altitude) / (
                nearby_altitude - miss_altitude)
            for i in range(2):
                new_vertex[i] = miss_vertex[i] + (
                    nearby_vertex[i] - miss_vertex[i]) * proportion
            return new_vertex
    
        def get_new_coordinate2(miss_pos):  # only one out-of-swath vertex
            # get vertex order and altitude of the just out-of-swath vertex
            miss_vertex = vt[miss_pos]
            miss_altitude = altitude_dict[k + '_' + str(miss_pos)]
    
            ## first vertex
            # for direction purpose so to make sure nearby vertex for comparison is the one in swath, not not the one that out of swath
            nearby_vertex1 = vt[miss_pos - 1]
            nearby_altitude1 = altitude_dict[k + '_' + str(miss_pos - 1)]
    
            # get the new vertex by 'altitude proportion formulae'
            new_vertex1 = ['', '']
            proportion = (float(maximum_angle) - miss_altitude) / (
                nearby_altitude1 - miss_altitude)
            for i in range(2):
                new_vertex1[i] = miss_vertex[i] + (
                    nearby_vertex1[i] - miss_vertex[i]) * proportion
    
            ## second vertex
            # for direction purpose so to make sure nearby vertex for comparison is the one in swath, not not the one that out of swath
            if miss_pos == len(vt) - 1:  # if order of outlying index is the last, make nearby index as 0
                nearby_vertex2 = vt[0]
                nearby_altitude2 = altitude_dict[k + '_0']
            else:
                nearby_vertex2 = vt[miss_pos + 1]
                nearby_altitude2 = altitude_dict[k + '_' + str(miss_pos + 1)]
    
            # get the new vertex by 'altitude proportion formulae'
            new_vertex2 = ['', '']
            proportion = (float(maximum_angle) - miss_altitude) / (
                nearby_altitude2 - miss_altitude)
            for i in range(2):
                new_vertex2[i] = miss_vertex[i] + (
                    nearby_vertex2[i] - miss_vertex[i]) * proportion
            return new_vertex2, new_vertex1

        # Read our saved satellite csv file
        sat_metadata = pd.read_csv(satellite_database)
        sat_metadata = sat_metadata.set_index(['Satellite'])
        satellite_id = sat_metadata['Satellite ID'][satellite_name]
        # get the name of all available instruments and the angle on the , at max 5
        instruments_boundangle = {
            sat_metadata['Instrument ' + str(i)][satellite_name]:
            sat_metadata['Converted Angle ' + str(i)][satellite_name]
            for i in range(1, 5)
            if isinstance(sat_metadata['Instrument ' + str(i)][satellite_name], str)
        }
    
        # Read boundary box from bounds
        # get the bounds in nwp coordinate system
        roi_bounds_rect_proj=np.array([transform(Proj(self.roi.crs),Proj({'init':'epsg:4326'}),*xy) for xy in self.roi_bounds_rect])
        vertices=roi_bounds_rect_proj
        number_of_vertices = len(vertices) - 1
        vertices_list = vertices[:-1].tolist()    
    
        #%% To store info we got from heavensabove.com into dictionaries for later manipulation
        if satellite_instrument in instruments_boundangle:
            instruments_boundangle = {
                satellite_instrument: instruments_boundangle[satellite_instrument]
            }
            print('Request Instrument was found in dataset')
        else:
            print(
                'requested Instrument was not found (typing error?), we provide results for all instruments'
            )
    
        # for each instrument in satellite we loop
        df_passes_full = pd.DataFrame(columns=['satellite', 'Sensor'])
        gdf_passes_partial = gpd.GeoDataFrame(columns=['satellite', 'Sensor', 'pass_time', 'geometry'])
        for instrument_name, maximum_angle in instruments_boundangle.items():
            print('We are getting information for ' + satellite_name + ' ' +
                  instrument_name)
            # Make three blank dictionaries and one blank list
            pass_satellite_dict = {
            }  # for storing vertex order (value) with passing date as (key)
            altitude_dict = {
            }  # for storing altitude (value) with passing date and vertex order as (key)
            time_dict = {
            }  # for storing satellite passing time (value) with passing date as (key)
            # test each vertex
            for vertex in vertices_list:
                lat = vertex[1]
                lng = vertex[0]
                vertex_order = vertices_list.index(
                    [lng, lat])  # get the order of the particular index
                # scrap
                try:
                    req = urllib.request.urlopen(
                        'https://www.heavens-above.com/PassSummary.aspx?satid=' +
                        str(satellite_id) + '&lat=' + str(lat) + '&lng=' +
                        str(lng) + '&loc=Unspecified&alt=0&tz=UCT&showall=t')
                except Exception as e:
                    print(e)
                    print('... update certificate and retry')
                    import ssl
                    context = ssl._create_unverified_context()
                    req = urllib.request.urlopen(
                        'https://www.heavens-above.com/PassSummary.aspx?satid=' +
                        str(satellite_id) + '&lat=' + str(lat) + '&lng=' +
                        str(lng) + '&loc=Unspecified&alt=0&tz=UCT&showall=t',
                        context=context)
                    print('... update certificate and retry...sucessful')
                # example url: https://www.heavens-above.com/PassSummary.aspx?satid=28654&lat=53.059706232696044&lng=13.8689581095596&loc=Unspecified&alt=0&tz=UCT&showall=t
                article = req.read().decode('utf-8')
                soup = BeautifulSoup(article, 'html.parser')
                table = soup.find("table", {"class": "standardTable"})
                trs = table.find_all("tr", {"class": "clickableRow"})
                for tr in trs:
                    tds = tr.find_all("td")
                    satellite_info = []
                    for td in tds:
                        satellite_info.append(td.getText())
                    if satellite_info[11] == 'visible':
                        satellite_info[11] = 'night'
                    passdate = satellite_info[0]
                    passtime = satellite_info[5]
                    altitude = satellite_info[6][:-1]
                    # if satellite altitude in heavensabove.com is greater than the boundary angle of the swath, then vertex is inside swath
                    if float(altitude) > float(maximum_angle):
                        if passdate + ' ' + passtime[:2] not in pass_satellite_dict:  # surrounding time
                            if passdate + ' ' + str(int(passtime[:2]) + 1) in pass_satellite_dict:
                                passtime = str(int(passtime[:2]) + 1)
                            elif passdate + ' ' + str(int(passtime[:2]) - 1) in pass_satellite_dict:
                                passtime = str(int(passtime[:2]) - 1)
                            else:  # if it is a new pass date
                                pass_satellite_dict[passdate + ' ' + passtime[:2]] = []  # construct a new date as key
                        # fill out all the three dictionaries
                        pass_satellite_dict.setdefault(passdate + ' ' + passtime[:2], []).append(vertex_order)
                        altitude_dict[passdate + ' ' + passtime[:2] + '_' +str(vertex_order)] = float(altitude)
                        time_dict[passdate + ' ' + passtime[:2]] = satellite_info[5]
    
                    # include also the marginal satellite altitudes which are just slightly out of swath, so that we can measure the proportion and update the vertex later
                    elif float(altitude) <= float(maximum_angle) and float(
                            altitude) > float(float(maximum_angle) * 0.8):
                        if passdate + ' ' + passtime[:2] not in pass_satellite_dict:  # surrounding time #%%
                            if passdate + ' ' + str(int(passtime[:2])+1) in pass_satellite_dict:  #%%
                                passtime = str(int(passtime[:2])+1)
                            elif passdate + ' ' + str(int(passtime[:2])-1) in pass_satellite_dict:
                                passtime = str(int(passtime[:2])-1)
                            else:
                                passtime = passtime             
                        altitude_dict[passdate + ' ' + passtime[:2] + '_' +
                                      str(vertex_order)] = float(altitude)
                    else:
                        continue
    
        #%% Check which points are inside satellite dependent threshold
            # 1: no passes
            if pass_satellite_dict == {}:
                print('No overfly of satellite ' + satellite_name + ' within next 10 days')
                #!! return shp file as None
    
            else:
                for k, v in pass_satellite_dict.items():
                    # remove duplicate
                    v = list(dict.fromkeys(v)) 
                    #bring the pass_time into datetime format
                    pass_datestr = k[:-2] + str(_datetime.utcnow().year) + time_dict[k]
                    pass_datetime = _datetime.strptime(pass_datestr, '%d %b %Y%H:%M:%S')
                    pass_date = pass_datetime.strftime("%Y-%m-%d %H:%M")
    
                    # 2: all passes: create a new ROI Shape with same extention as original
                    if len(v) == number_of_vertices:
                        print(
                            'The whole area is covered under ' + satellite_name + ' ' +
                            instrument_name + ' on ' + k[:-3] + '' + time_dict[k])
                        df_pass_full = pd.DataFrame([[satellite_name, instrument_name]], columns=['satellite', 'Sensor'])
                        df_pass_full['pass_time'] = pass_date
                        df_passes_full = df_passes_full.append(df_pass_full, ignore_index=True, sort=False)
    
                    # 3: partial overfly: create new geometry
                    else:
                        print('partial coverage under ' + satellite_name + ' ' + instrument_name + ' on ' + k[:-2] + '' + time_dict[k])    
                        # get the missing vertices that just out of the swath
                        missing_vertices_order = []
                        missing_vertices = []
                        for i in range(v[0], number_of_vertices):
                            if i not in v:
                                missing_vertices_order.append(i)
                                missing_vertices.append(vertices_list[i])
                        for i in range(0, v[0]):
                            if i not in v:
                                missing_vertices_order.append(i)
                                missing_vertices.append(vertices_list[i])
    
                        mp1 = missing_vertices_order[0]
                        mp2 = missing_vertices_order[-1]
                            # We put the out-of-swath vertices into the above formulae, and then turn the updated vertice list into numpy array
        
                        vt = vertices_list.copy()
                        if mp1 == mp2:  # if number of outlying vertices is equal to 1
                            get = get_new_coordinate2(mp1)  # new vertices
                            vt[mp1] = get[0]
                            
                            # add the extra vertex to correct position
                            check_pt =mp1+1
                            if len(vt) == check_pt:
                                check_pt = 0
                            if vt[mp1][0] > vt[check_pt][0] and vt[check_pt][0] > get[1][0] is True:
                                vt.insert(check_pt, get[1]) # insert one more item
                            elif  vt[mp1][0] < vt[check_pt][0] and vt[check_pt][0] < get[1][0] is True:
                                vt.insert(check_pt, get[1]) 
                            else:
                                vt.insert(mp1, get[1])
    
                            # make sure the last item is equal to the first item
                            first_vertex = vt[0]
                            vt.append(first_vertex)
    
                        else:  # if number of outlying vertices is more than 1
                            get1 = get_new_coordinate1(mp1)  # new vertex 1
                            get2 = get_new_coordinate1(mp2)  # new vertex 2
                            vt[mp1] = get1
                            vt[mp2] = get2
                            # to remove the remaining out of swath vertices
                            for mv in missing_vertices:
                                if mv in vt is True:
                                    vt = vt.remove(mv)
                            # make sure the last item is equal to the first item
                            first_vertex = vt[0]
                            vt.append(first_vertex)
                        # convert the geometry to polygon and write the geopandas dataseries
                                              
                        gdf_pass_partial = gpd.GeoDataFrame(
                            [[satellite_name, instrument_name, pass_date]],
                            columns=['satellite', 'Sensor', 'pass_time'])
                        gdf_pass_partial['geometry'] = Polygon(vt)
                        gdf_pass_partial.crs = 'epsg:4326'
                        #gdf_pass_partial.to_file ('./' + satellite + '_' + instrument_name +'_'+ str(k[:-2]))
                        gdf_passes_partial = gdf_passes_partial.append(gdf_pass_partial, ignore_index=True)
                        gdf_passes_partial.crs = 'epsg:4326'
            
        #add the passes            
        self.satellite_passes={'full_passes' : df_passes_full, 'partial_passes' : gdf_passes_partial}
        #return function        
        return self.satellite_passes
    
def test_nrt_downloader():
    print('This is a test run to test the main functionality of the nrt downloader module')
    test=downloader()
    test.modis_products(modis_product = "MOD14A1.006",server='https://n5eil01u.ecs.nsidc.org',product_parameters=['FireMask','MaxFRP'],modis_user = 'Nixdorf88', modis_pw = 'Dummling88')
    test.modis_products(modis_product = "MOD14A1.006",server='https://e4ftl01.cr.usgs.gov',product_parameters=['FireMask','MaxFRP'],modis_user = 'Nixdorf88', modis_pw = 'Dummling88')
    
    test.satellite_coverage()
    test.weatherprediction_parameters(nwp_model='cosmo-d2')
    test.radorecent(time_res='daily',to_harddisk=True)
    test.weatherprediction(nwp_model='cosmo-d2')
    test.mhmsoilmoist()
    
    #test.reproject(init_crs= test.xr.crs)
    print('All tests sucessful')
    return test
if __name__ == "__main__":
     test_nrt_downloader()              
