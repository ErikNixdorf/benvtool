"""
This Module creates a homogenized netcdf file with data from various possible sources
NOTICE: Currently, BGR DATA is only available as WMS only, which is 

"""
import sys
from roverweb import *
import roverweb as rw
import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from nrt_io import downloader as nrt_dw
from shapely.geometry import box,Polygon
import xarray as xr
import rasterio
import os
import fiona
from urllib.request import urlopen
from rasterstats import point_query,zonal_stats
from io import BytesIO
import xml.etree.ElementTree
dataset_version='0.25'

#%% Some functionality
def make_dict_from_tree(element_tree):
    """Traverse the given XML element tree to convert it into a dictionary.
    from https://ericscrivner.me/2015/07/python-tip-convert-xml-tree-to-a-dictionary/
    but modified and improved by Erik Nixdorf
    :param element_tree: An XML element tree
    :type element_tree: xml.etree.ElementTree
    :rtype: dict
    """

    def internal_iter(tree, accum):
        """Recursively iterate through the elements of the tree accumulating
        a dictionary result.
 
        :param tree: The XML element tree
        :type tree: xml.etree.ElementTree
        :param accum: Dictionary into which data is accumulated
        :type accum: dict
        :rtype: dict
        """
        if tree is None:
            return accum

        if len(tree) >= 1:
            accum[tree.tag] = {}
            for each in tree:
                #check whether a list is inside
                try:
                    accum[tree.tag][each.tag]=internal_iter(each, {})[each.tag]['item']
                except:                        
                    result = internal_iter(each, {})
                    #if dictionary we have to change key value pair
                    try:
                       accum[tree.tag].update({each.attrib['name']: each.text})
                    except:
                        if each.tag in accum[tree.tag]:
                            if not isinstance(accum[tree.tag][each.tag], list):
                                accum[tree.tag][each.tag] = [accum[tree.tag][each.tag]]
                            accum[tree.tag][each.tag].append(result[each.tag])
                        else:
                            accum[tree.tag].update(result)
        else:
            accum[tree.tag] = tree.text

        return accum

    return internal_iter(element_tree, {})

def rasterstats_on_geometry(geoframe,raster_path=None,raster_dtype='int',
                            output_colm='mean_value',stats_type='point_query',
                            metrics=['mean'], interpolate='bilinear',
                            nodata=0,all_touched=False):
    """
    A simple function for the standard procedure to map values from a raster on
    a list of geometries in a geodataframe
    #currently, caterogical data is not integrated    
    """
    #correct metrics if input is str
    if isinstance(metrics,str):
        metrics=[metrics]
    #read rasterfile
    with rasterio.open(raster_path) as src:
        affine = src.transform
        if raster_dtype=='int':
            data = np.int32(src.read(1))
        else:
            data=np.float64(src.read(1))
    #depending on the chosen stats type we apply different methods from raster
    #stats library
    if stats_type=='point_query':
        geoframe[output_colm]=point_query(geoframe.centroid,data,affine=affine,nodata=nodata,interpolate=interpolate)
    if stats_type=='zonal_stats':
        statistic_dict=zonal_stats(geoframe,data,stats=metrics,affine=affine,nodata=nodata,all_touched=all_touched)
        #we loop through the keys
        for key in statistic_dict[0].keys():
            geoframe[output_colm+'_'+key]=[statistic_dict_entry[key] for statistic_dict_entry in statistic_dict]
    #return geotdataframe
    return geoframe
        

def append_bgr_data(gdf_in,bgr_datasets):
    """
    A function which calls download from BGR dataset and maps polygon data 
    on centroid based sjoin method            
    """
    #create an copy of the input data using centroid as geometry
    gdf_left=gdf_in.copy()
    gdf_left.geometry=gdf_left.centroid
    #loop trough each dataset
    for bgr_dataset in bgr_datasets:
        #get server response
        resp = urlopen(bgr_datasets[bgr_dataset]['data_url'])
        #create fiona zip memory file
        memfile=fiona.io.ZipMemoryFile(BytesIO(resp.read()))
        #open the shape file as fiona collection
        fiona_collection=memfile.open(path=bgr_datasets[bgr_dataset]['dataset_path'])
        gdf_bgr = gpd.GeoDataFrame.from_features([feature for feature in fiona_collection], crs=bgr_datasets[bgr_dataset]['crs'])
        #convert the input dataset to the destination crs
        gdf_left=gdf_left.to_crs(gdf_bgr.crs)
        #Add data from Polygons using spatial join
        gdf_joined=gpd.sjoin(gdf_left,gdf_bgr,how='left',op='within')
        #add required data columns to the input gdf
        for key,value in bgr_datasets[bgr_dataset]['layers'].items():
            gdf_in[value]=gdf_joined[key]
        print(bgr_dataset, ' added to dataframe')
        
    return gdf_in
    
    
def precipitation_statistics(row,ds_rado=xr.Dataset(), geometry_col='geometry_radolan',temporal_resolution='daily',timesteps_backward=7,current_date=datetime(2010,12,12)):
    """
    A usefull function to get rainfaill statistics on radolan datasets row_wise
    example: gdf_timestep=gdf_timestep.apply(precipitation_statistics,axis=1,ds_rado=ds_rado_slice, geometry_col='geometry_radolan',temporal_resolution=temporal_resolution,timesteps_backward=rado_time_step,current_date=date)
    """
    #write valid radolan precipitation parameter, either daily or hourly
    if temporal_resolution=='daily':
        precip_parameter='SF'
    else:
        precip_parameter='RW'
    #create output dataset
    precip_stats=xr.Dataset()
    #select the closest location
    precip_position=ds_rado.sel(x=row[geometry_col].x,y=row[geometry_col].y,method='nearest')           
    
    #add accumulation
    precip_stats=xr.merge([precip_stats,precip_position[precip_parameter].sum(dim='time')])
    precip_stats=precip_stats.rename_vars({precip_parameter:'p_' + str(timesteps_backward) + '_acc'})
    #add mean
    precip_stats=xr.merge([precip_stats,precip_position[precip_parameter].mean(dim='time')])
    precip_stats=precip_stats.rename_vars({precip_parameter:'p_' + str(timesteps_backward) + '_mean'})    
    #add deviation
    precip_stats=xr.merge([precip_stats,precip_position[precip_parameter].std(dim='time')])
    precip_stats=precip_stats.rename_vars({precip_parameter:'p_'+str(timesteps_backward)+'_std'})
    #add the amount of hours with rainfall within the dataset     
    event_counts=xr.apply_ufunc(np.count_nonzero, precip_position,input_core_dims=[['time']],vectorize=True, output_dtypes=[int])
    precip_stats=xr.merge([precip_stats,event_counts])
    precip_stats=precip_stats.rename_vars({precip_parameter:'p_'+str(timesteps_backward)+'_counts'})
    #the actual rain during measurement
    precip_mon=ds_rado.sel(x=row[geometry_col].x,y=row[geometry_col].y,time=current_date,method='nearest')
    precip_stats=xr.merge([precip_stats,precip_mon])
    precip_stats=precip_stats.rename_vars({precip_parameter:'p_'+str(timesteps_backward)+'_mon'})
    #create a dataframe
    df_precip_stats=pd.Series(precip_stats.to_array(),index=precip_stats.keys())
    row=pd.concat([row,df_precip_stats],axis=0)

    return row


def get_nearest_indices(row,geometry_column=['geometry_radolan_x'],coordinate_array=[np.zeros((5,5))]):
    """
    A Tool which finds closest column/row indices in a matrix by minimum distance
    Parameters based on data from rows in a dataframe
    ----------
    row : TYPE
        DESCRIPTION.
    geometry_column : TYPE, optional
        DESCRIPTION. The default is ['geometry_radolan_x'].
    coordinate_array : TYPE, optional
        DESCRIPTION. The default is [rado_y].

    Returns
    -------
    index : TYPE
        DESCRIPTION.

    """
    index=list()
    for i in range(0,len(geometry_column)):
        index.append(np.argmin(np.abs(coordinate_array[i]-row[geometry_column[i]])))
    return index



#%% load input data from config file
filename = 'configuration.xml'
with open(filename, 'r') as config_file:
    config = make_dict_from_tree(xml.etree.ElementTree.fromstring(config_file.read()))
#ignore the root level of the xml
for key in config.keys():
    config=config[key]
#%% do some preprocessing     

d1 = datetime.strptime(config['basics']['start_time'], "%Y-%m-%d-%H")
d2 = datetime.strptime(config['basics']['end_time'], "%Y-%m-%d-%H")
#create dates depending on the temporal resolution
if config['basics']['temporal_resolution']=='daily':
    days=abs((d2 - d1).days)
    date_list=[d1 + timedelta(days=x) for x in range(0,days)]
elif config['basics']['temporal_resolution']=='hourly':
    dif = int((d2-d1).total_seconds()/3600) ## time difference in hours
    date_list = [d1 + timedelta(hours=x) for x in range(dif+1)]
else:
    sys.exit('Unnkowm temporal resolution, either daily or hourly is supported')
#create output directory if not exist
if not os.path.exists(os.getcwd()+config['basics']['output_location']):
    os.makedirs(os.getcwd()+config['basics']['output_location'])
#create dabase directory if not exists
if not os.path.exists(os.getcwd()+config['basics']['dbase_location']):
    os.makedirs(os.getcwd()+config['basics']['dbase_location'])
    
#some more renaming and datatype things
new_column_names={"LST_Day_1km": "LST_Day_1km_weekly",
                  "LST_Night_1km": "LST_Night_1km_weekly",
                  'Emis_31':'Emis_31_weekly',
                  'Emis_32':'Emis_32_weekly',
                  '2m_air_temperature':'air_temperature',
                  '2m_relative_humidity':'relative_humidity'
                }


#load the attributes dataset
df_data_attributes=pd.read_csv(os.getcwd()+config['basics']['attributesfile_location'],delimiter=',',encoding = "ISO-8859-1")
df_data_attributes.set_index('NAME',inplace=True)
data_attributes=df_data_attributes.transpose().to_dict()

# set some standard parameters
gdfmueglitz_output = gpd.GeoDataFrame()
failure_dates=list() # counts the failed dates
initdf=True
new_month=True
delete_files=False

config['basics']['grid_crs']
#%% load data
if config['basics']['grid_location'][-3:]=='shp':    
    gdf_in = gpd.GeoDataFrame.from_file(os.getcwd()+config['basics']['grid_location'])
else:
    df_in=pd.read_csv(os.getcwd()+config['basics']['grid_location'])
    #create geometry
    def polygonise_2Dcells(df_row): 
        return Polygon([(df_row.left, df_row.bottom ), (df_row.right, df_row.bottom),(df_row.right, df_row.top), (df_row.left, df_row.top),(df_row.left, df_row.bottom )])
    gdf_in=gpd.GeoDataFrame(df_in,geometry=df_in.apply(polygonise_2Dcells, axis=1),crs=config['basics']['grid_crs'])
#convert to epsg4326

gdf_in=gdf_in.to_crs(crs="epsg:4326")
        
#delete useless first column

for col in ['left','bottom','right','top','Area']:
    try:
        gdf_in.drop(columns=col,inplace=True)
    except:
        pass


#%%We start to loop
for date in date_list:
    #get current month
    current_month=date.month
    
    #%% if an stationary_dataset does not exist we have to create it first
    if os.path.exists(os.getcwd()+config['basics']['output_location']+'\\'+config['basics']['output_name']+'_stationary.nc') is False:
        gdf_static=gdf_in.copy()
        
        #%% get data from soilgrid
        if 'soilgrid' in config:
            print('Appending datasets from Soilgrids database')   
            # try wcs service if not working use rest API
            try:
                gdf_static = rw.soilgrid.apnd_from_wcs(gdf_static,soilgridlrs=config['soilgrid']['wcs_layers'], raster_res=(int(config['basics']['spatial_resolution']), int(config['basics']['spatial_resolution'])), statistical_metric=['mean'], all_touched=True, output=None)
            
            except Exception as e:
                print(e)
                print('trouble with soilgrid WCS, try RESTAPI')
                #repair this annoying crs issue again
                gdf_static.crs={'init':gdf_static.crs}
                gdf_static = rw.soilgrid.apnd_from_restapi(gdf_static,
                                                               soil_attributes=config['soilgrid']['restapi_layers'],
                                                               soil_layerdepths=['sl1'], output=None)
                gdf_static = rw.soilgrid.apnd_from_restapi(gdf_static,
                                                               soil_attributes=config['soilgrid']['restapi_layers'],
                                                               soil_layerdepths=['sl2'], output=None)
                gdf_static = rw.soilgrid.apnd_from_restapi(gdf_static,
                                                               soil_attributes=config['soilgrid']['restapi_layers'],
                                                               soil_layerdepths=['sl3'], output=None)
            #do some dataset corrections
            #correct some value ranges in the stationary dataset             
            data_column_nms=list(gdf_static.columns)
            for data_column_nm in data_column_nms:    
                #data_column=pd.to_numeric(data_column, errors ='coerce').astype('float64')
                # correct for soil layers
                if 'PPT' in data_column_nm:
                    gdf_static[data_column_nm]=gdf_static[data_column_nm].mask(gdf_static[data_column_nm] > 100)
                if 'HOX' in data_column_nm:
                    gdf_static[data_column_nm]=gdf_static[data_column_nm].mask(gdf_static[data_column_nm] > 100)
                if 'CRFVOL' in data_column_nm:
                    gdf_static[data_column_nm]=gdf_static[data_column_nm].mask(gdf_static[data_column_nm] > 100) 
            print('Appending datasets from Soilgrids database done')  
        #%% get data fram BGR
        if 'hydrogeology' in config: 
            print('Appending datasets from BGR')        
            gdf_static=append_bgr_data(gdf_static,config['hydrogeology'])
        
            print('Appending datasets from BGR done')
        
        #%%next is to get data from the overpass API
        if 'openstreetmap' in config:
            # the number of clusters will be dynamically decided based on data len
            clusters=int(np.ceil(np.log(len(gdf_in))))
            if clusters<1:
                clusters=1
            gdf_static = rw.geometry.clustering(
                gdf_static, clusters=clusters, cluster_fld_name='ClusterID')
            # %%having our clustered dataset we can run our osm to get desired secondary
            # datasets
            # group geodataframe
            gdf_static_clusters = gdf_static.groupby(by='ClusterID')
            # create an empty result geodataframe            
            gdf_static_osm = gpd.GeoDataFrame()
            
            # inititate the loop
            for cluster_name, rover_cluster in gdf_static_clusters:
                print('add information for cluster No',cluster_name)
                # create the queryboundstr
                querygeobound = rw.osm.querybound_generation(rover_cluster)
                #loop over all features which want to be extracted from OSM
                for osm_feature in config['openstreetmap'].keys():
                    rover_cluster = rw.osm.apnd_from_overpass(
                    rover_cluster, querygeobound, queryfeatures=config['openstreetmap'][osm_feature]['queryfeatures'],
                    values_out=config['openstreetmap'][osm_feature]['values_out'],
                    values_in=config['openstreetmap'][osm_feature]['values_in'],
                    CountValues=eval(config['openstreetmap'][osm_feature]['CountValues']))
                # append subset back to entire dataset
                gdf_static_osm = gdf_static_osm.append(rover_cluster,sort=True)            
            #sort by index    
            gdf_static=gdf_static_osm.sort_index()
            #try rename some columns
            osm_col_names={'w_highway': 'road_type','w_landuse':'osm_landuse'}
            for osm_col_name in osm_col_names:
                try:
                    gdf_static=gdf_static.rename(columns={osm_col_name: osm_col_names[osm_col_name]})
                except:
                    pass
            #delete some columns
            gdf_static.drop(columns=['ID','ClusterID'],inplace=True)
            #change some data type issues
            # a few corrections on the string data no tyoe
            try:
                gdf_static['road_type']=gdf_static['road_type'].replace(to_replace=np.nan,value='no_road')
            #correctin for certain road type
            except:
                pass
            try:
                gdf_static['road_type']=gdf_static.road_type.replace({'no':'no_road'})
            except:
                pass            
        #%% Next we load the topograhphic datasets
        if 'topography' in config:
            print('Appending static topographical datasets')   
            for topo_dataset in config['topography']:
            #corine data
                gdf_static=rasterstats_on_geometry(gdf_static,
                                                   raster_path=os.getcwd()+config['topography'][topo_dataset]['file_location'],
                                                   raster_dtype=config['topography'][topo_dataset]['raster_dtype'],
                                                   output_colm=topo_dataset,
                                                   stats_type=config['topography'][topo_dataset]['stats_type'],
                                                   interpolate=config['topography'][topo_dataset]['interpolate'], 
                                                   nodata=int(config['topography'][topo_dataset]['no_data']),
                                                   metrics=config['topography'][topo_dataset]['metrics'])
            # same for the slope
            #try rename some columns
            topo_col_names={'waterbodies_max': 'waterbody','waterways_max':'waterway'}
            for topo_col_name in topo_col_names:
                try:
                    gdf_static=gdf_static.rename(columns={topo_col_name: topo_col_names[topo_col_name]})
                except:
                    pass
        print('Appending static topographical datasets done')  
        #%% Final preparations of static dataset
        #change coordinate system and add x and y information
                #get tp geographic coordinates
        gdf_static=gdf_static.to_crs(config['basics']['output_crs'])
        #fix rounding issues, meter precision is enough
        gdf_static['y']=np.round(gdf_static.centroid.y).astype(int)
        gdf_static['x']=np.round(gdf_static.centroid.x).astype(int)
        #replace infinity values
        gdf_static.replace([np.inf, -np.inf], np.nan,inplace=True)       
        #define a new index
        gdf_static.set_index(['x','y'],inplace=True)
        #delete the geometry column        
        gdf_static=gdf_static.drop(columns=['geometry'])
        
        #%% convert dataframe to xarray and save
        ds_out_stat=gdf_static.to_xarray().squeeze()
        #add attributes to the indivdual layers
        for col in ds_out_stat:
            try:
                ds_out_stat[col]=ds_out_stat[col].assign_attrs(data_attributes[col])
            except:
                print('Attributes for parameter',col,' not defined')
                pass
        #write to file
        ds_out_stat.to_netcdf(os.getcwd()+config['basics']['output_location']+'\\'+config['basics']['output_name']+'_stationary.nc',format='NETCDF4_CLASSIC') 
        
    #if the path exist we just upload the file back to xarray
    else:
        ds_stationary=xr.load_dataset(os.getcwd()+config['basics']['output_location']+'\\'+config['basics']['output_name']+'_stationary.nc')
    
    
    #%% Now we go to the dynamic data
    print('Start retrieving dynamic datasets for timestep',date.date())
    #if initial we need to create a boundary shapefile
    if initdf==True:
        bounds=gdf_in.geometry.total_bounds
        b=box(bounds[0],bounds[1],bounds[2],bounds[3])
        bounds_polg = gpd.GeoDataFrame({'data':pd.DataFrame(bounds).iloc[0]})
        bounds_polg['geometry']=b
        bounds_polg.crs=gdf_in.crs
        bounds_polg.to_file('roi.shp')
        #manipulate geometry
        gdf_in.geometry=gdf_in.geometry.centroid
        #do some radolan dependent manipulations
        if 'radolan' in config:
            gdf_in['geometry_radolan_x']=gdf_in.to_crs(rasterio.crs.CRS.from_proj4('+proj=stere +lat_0=90 +lat_ts=90 +lon_0=10 +k=0.93301270189 + x_0=0 +y_0=0 +a=6370040 +b=6370040 +to_meter=1000 +no_defs')).geometry.x        
            gdf_in['geometry_radolan_y']=gdf_in.to_crs(rasterio.crs.CRS.from_proj4('+proj=stere +lat_0=90 +lat_ts=90 +lon_0=10 +k=0.93301270189 + x_0=0 +y_0=0 +a=6370040 +b=6370040 +to_meter=1000 +no_defs')).geometry.y 
        if 'modis' in config:
            gdf_in['geometry_modis_x']=gdf_in.to_crs(rasterio.crs.CRS.from_proj4('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs')).geometry.x
            gdf_in['geometry_modis_y']=gdf_in.to_crs(rasterio.crs.CRS.from_proj4('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs')).geometry.y
            #final projections for output netcdf
        #fix rounding issues, meter precision is enough
        gdf_in['y']=np.round(gdf_in.to_crs(config['basics']['output_crs']).centroid.y).astype(int)
        gdf_in['x']=np.round(gdf_in.to_crs(config['basics']['output_crs']).centroid.x).astype(int)
        
    #%%get copy of original dataframe for the desired time step and change geometry to centroid
    gdf_timestep=gdf_in.copy()
    gdf_timestep['time']=date
    """
    Some Ideas to merge houly data to entire days which allows faster access to dwd database
    In config I suggest in [basics] fo add: <daily_data_processing type="bool">True</daily_data_processing>
           print('append hourly data together to daily datasets')
        #create a new list for the hour of each date
        date_list=[repeat(datetime(date.year,date.month,date.day,x,0),len(gdf_timestep)) for x in range(0,24)]
        #extent the dataframe
        gdf_timestep=pd.concat([gdf_timestep]*24)
        #attach to timestep time
        date_list_extent=list()
        for date_l in date_list:
            date_list_extent.extend(date_l)
        gdf_timestep['time']=date_list_extent
        gdf_timestep=gdf_timestep.reset_index(drop=True)
    """
    #add DOY 
    gdf_timestep['DOY'] = date.timetuple().tm_yday
    #%% we start with Precipitation Data from Raster
    if 'radolan' in config:
        print('get radolan precipitation data for campaign') 
        #create the downloader class with time span depending on statistics
        if config['basics']['temporal_resolution']=='daily':
            time_increment=timedelta(days=int(config['radolan']['statistic_time_steps'][-1]))
        elif config['basics']['temporal_resolution']=='hourly':
            time_increment=timedelta(hours=int(config['radolan']['statistic_time_steps'][-1]))
        
        datemin=(date-time_increment).strftime("%Y-%m-%d:%H%M")
        datemax=date.strftime("%Y-%m-%d:%H%M")
        #%%create the downloader class
        rado_downloader=nrt_dw.downloader(start_time=datemin, end_time=datemax, roi='roi.shp', roi_buffer=0.02)
        #get the radolan data
        rado_downloader.radorecent(time_res=config['basics']['temporal_resolution'],to_harddisk=eval(config['radolan']['to_harddisk']),rado_data_dir=os.getcwd()+config['basics']['dbase_location']+'\\radolan\\')
        ds_rado=rado_downloader.xr
        # if initial run we try to get cordinates and precip parameter
        # a new numpyic approach
        if initdf==True:
            if config['basics']['temporal_resolution']=='daily':
                precip_parameter='SF'
                col_name='precip_daily'
            else:
                precip_parameter='RW'
                col_name='precip_hourly'
            rado_x=ds_rado.coords['x'].values
            rado_y=ds_rado.coords['y'].values    
            position_indices=np.array(gdf_timestep.apply(get_nearest_indices,axis=1,geometry_column=['geometry_radolan_x','geometry_radolan_y'],coordinate_array=[rado_x,rado_y]).to_list())
        #after we got the indices, we need to apply on array for each time step
        #the actual rain at the required timestep
        col_name='precip_'+config['basics']['temporal_resolution']
        if ds_rado[precip_parameter].time.size>1:
            rado_step=np.array(ds_rado[precip_parameter].sortby('time').sel(time=date,method='nearest'))
        else:
            rado_step=np.array(ds_rado[precip_parameter])
        gdf_timestep[col_name]=[rado_step[pos_index[1],pos_index[0]] for pos_index in position_indices]
        #the statisics
        if ds_rado[precip_parameter].time.size>1:
            if isinstance(config['radolan']['statistic_time_steps'],str):
                config['radolan']['statistic_time_steps']=[config['radolan']['statistic_time_steps']]
            for rado_time_step in config['radolan']['statistic_time_steps']:
                
                rado_time_step=int(rado_time_step)
                ds_rado_slice=ds_rado.sortby('time').sel(time=slice(date-timedelta(days=rado_time_step),date+timedelta(days=1)))
                #first the accumulatio(sum)
                col_name='precip_' + str(rado_time_step) + '_acc'
                rado_sum=np.array(ds_rado_slice[precip_parameter].sum(dim='time'))
                #select from rado_sum using approach from questions/35607818/        
                gdf_timestep[col_name]=[rado_sum[pos_index[1],pos_index[0]] for pos_index in position_indices]
                #add mean
                col_name='precip_' + str(rado_time_step) + '_mean'
                rado_mean=np.array(ds_rado_slice[precip_parameter].mean(dim='time'))
                gdf_timestep[col_name]=[rado_mean[pos_index[1],pos_index[0]] for pos_index in position_indices]           
                #add deviation
                col_name='precip_' + str(rado_time_step) + '_std'
                rado_std=np.array(ds_rado_slice[precip_parameter].std(dim='time'))
                gdf_timestep[col_name]=[rado_std[pos_index[1],pos_index[0]] for pos_index in position_indices]
                #add the amount of hours with rainfall within the dataset 
                col_name='precip_'+str(rado_time_step)+'_counts'
                rado_count=np.count_nonzero(np.array(ds_rado_slice[precip_parameter]),axis=0)
                gdf_timestep[col_name]=[rado_count[pos_index[1],pos_index[0]] for pos_index in position_indices]
        #delete geometry columns
        gdf_timestep.drop(columns=['geometry_radolan_x','geometry_radolan_y'],inplace=True)
        print('Done including dwd radolan precipitation data statistics')
    #%% Next is that we get the data from the dwd weather stations
    if 'dwd_weather_stations' in config:
        print('Start to Download DWD Station Data')
        time_str_col = 'Date Time(UTC)'
        no_of_nearest_stations=int(config['dwd_weather_stations']['no_of_nearest_stations'])
        #first the things we only have to do for initial conditions
        if initdf==True:
            #create temporary columns from dwd data extraction
            dwd_temporary_columns=[dwd_category+'_station_'+str(i) for dwd_category in config['dwd_weather_stations']['categories'] for i in range(0,no_of_nearest_stations)]
            dwd_temporary_columns.extend([dwd_category+'_distance_'+str(i) for dwd_category in config['dwd_weather_stations']['categories'] for i in range(0,no_of_nearest_stations)])
            dwd_dbases=dict()
            #create a database for each category
            for dwd_category in config['dwd_weather_stations']['categories']:
                print('Update DWD Database for ',dwd_category)
                gdf_timestep,dwd_dbase=rw.weather.Find_nearest_dwd_stations(gdf_timestep,
                             date_start=d1.strftime('%Y%m%d'),
                             date_end=d2.strftime('%Y%m%d'),
                             dwd_time_format='%Y%m%d%H',
                             data_category=dwd_category,
                             temp_resolution=config['basics']['temporal_resolution'],
                             no_of_nearest_stations=no_of_nearest_stations,
                             memory_save=True,
                             Output=True)
                #add database to dicrionary
                dwd_dbases.update({dwd_category:dwd_dbase})
                print('Update DWD Database for ',dwd_category,' done')
    
            
        # now we add the data for each time_step and each category
        #add the time in specific string format at special column
        if config['basics']['temporal_resolution']=='daily':    
            gdf_timestep[time_str_col]=(date-timedelta(hours=12)).strftime('%Y-%m-%d:%H:%M')
        else:
            gdf_timestep[time_str_col]=date.strftime('%Y-%m-%d:%H:%M')
        # now we add the data for each time_step and each category
        print('Getting dynamic dwd station data for date ',date)
        for dwd_category in config['dwd_weather_stations']['categories']:
            gdf_timestep=rw.weather.Apnd_dwd_data(gdf_timestep,dwd_dbases[dwd_category],
                                 time_col=time_str_col,
                                 data_time_format='%Y-%m-%d:%H:%M',
                                 data_category=dwd_category,
                                 parameters=config['dwd_weather_stations']['categories'][dwd_category],
                                 no_of_nearest_stations=no_of_nearest_stations,
                                 idw_exponent=int(config['dwd_weather_stations']['idw_exponent'])
                                 )
        #delete the temporary dwd columns and transfer to gdf_in
        if initdf==True:
            gdf_in[dwd_temporary_columns]=gdf_timestep[dwd_temporary_columns]
            gdf_in[time_str_col]=gdf_timestep[time_str_col]
        gdf_timestep.drop(columns=dwd_temporary_columns,inplace=True)
        gdf_timestep.drop(columns=time_str_col,inplace=True)
    print('Getting dynamic dwd station data for date ',date, 'done')
    #%% Finally comes the MODIS Data
    if 'modis' in config:
        print('Start to Download modis sensor Data')
        #due to memory problems we clean every month the downloaded file 
        if initdf==False and new_month==True:
            delete_files=True
        for modis_product in config['modis']['products']:
            # we create the new downloader depending on whether we want to interpolate forward or backward
            if config['modis']['products'][modis_product]['interp'] =='backfill':        
                end_time=(date-timedelta(days=int(config['modis']['products'][modis_product]['delta_days_min']))).strftime("%Y-%m-%d:%H%M")
                start_time=(date+timedelta(days=int(config['modis']['products'][modis_product]['delta_days_max']))).strftime("%Y-%m-%d:%H%M")
                downloader=nrt_dw.downloader(start_time=start_time , end_time=end_time, roi='roi.shp', roi_buffer=0.02)
            # fi not we download datasets prior to date of investigation
            else:
                start_time=(date-timedelta(days=int(config['modis']['products'][modis_product]['delta_days_min']))).strftime("%Y-%m-%d:%H%M")
                end_time=(date+timedelta(days=int(config['modis']['products'][modis_product]['delta_days_max']))).strftime("%Y-%m-%d:%H%M")
                downloader=nrt_dw.downloader(start_time=start_time , end_time=end_time, roi='roi.shp', roi_buffer=0.02)        
            #we download the files
            downloader.modis_products(server='https://e4ftl01.cr.usgs.gov',
                                      modis_product = modis_product,
                                      product_parameters=config['modis']['products'][modis_product]['product_parameters'],
                                      modis_user = config['modis']['username'], 
                                      modis_pw = config['modis']['password'],
                                      delete_files=delete_files,
                                      modis_data_dir=os.getcwd()+config['basics']['dbase_location']+'\\modis\\',
                                      )
            config['modis']['products'][modis_product].update({'dataset':downloader.xr})
            #if initial download we get the indices        
            if initdf==True:
                modis_x=config['modis']['products'][modis_product]['dataset'].coords['x'].values
                modis_y=config['modis']['products'][modis_product]['dataset'].coords['y'].values 
                config['modis']['products'][modis_product].update({'position_indices':np.array(gdf_timestep.apply(get_nearest_indices,axis=1,geometry_column=['geometry_modis_x','geometry_modis_y'],coordinate_array=[modis_x,modis_y]).to_list())})    
                    
            # now we map the data from MODIS to our dataset
            
            # if the dataset has only one entry this is very easy
            if len(config['modis']['products'][modis_product]['dataset'].coords['time'])>1:
                #we first first depending on interpolation scheme the nearest position in time
                config['modis']['products'][modis_product]['dataset']=config['modis']['products'][modis_product]['dataset'].sel(time=date,method=config['modis']['products'][modis_product]['interp'])
            #loop through all columns and add to geometry file
            for modis_parameter in config['modis']['products'][modis_product]['dataset']:
                modis_array=np.array(config['modis']['products'][modis_product]['dataset'][modis_parameter]).squeeze()
                #add data to dataframe
                gdf_timestep[modis_parameter]=[modis_array[pos_index[1],pos_index[0]] for pos_index in config['modis']['products'][modis_product]['position_indices']]
                #replace individual no data with np.nan
                gdf_timestep[modis_parameter].replace(to_replace =int(config['modis']['products'][modis_product]['no_data'][modis_parameter]), value =np.nan)
            print('Added parameters from ', modis_product, 'done')
        #delete modis geometry
        gdf_timestep.drop(columns=['geometry_modis_x','geometry_modis_y'],inplace=True)
        #reset delete files
        delete_files=False
        print('finished to Download modis sensor Data')
    #%% Now comes the conversion and data cleaning
    print('Finished download of products for date', date)
    #rename some columns and delete some useless columns
    gdf_timestep.rename(columns=new_column_names,inplace=True)
    #delete some not needed columns
    gdf_timestep=gdf_timestep.drop(columns=['geometry'])
    #delete the nan
    gdf_timestep=gdf_timestep.replace([np.inf, -np.inf], np.nan)
    gdf_timestep.set_index(['x','y','time'],inplace=True)
    #create the x_array
    ds_timestep=gdf_timestep.to_xarray().squeeze()
    
    #%% Write out the netcdf data
    if new_month==True:
        ds_out=ds_timestep
        new_month=False
    else:
        ds_out=xr.concat([ds_out,ds_timestep], dim='time')
    #if we are at the end of a month we write out the data
    if config['basics']['temporal_resolution']=='daily':
        time_increment=timedelta(days=1)
    elif config['basics']['temporal_resolution']=='hourly':
        time_increment=timedelta(hours=1)
    if int((date+time_increment).month) != int(current_month):
        #merge with stationary_data
        ds_out=xr.merge([ds_out,ds_stationary])
        #add attributes to individual layers
        for col in ds_out:
            try:
                ds_out[col]=ds_out[col].assign_attrs(data_attributes[col])
            except:
                print('Attributes for parameter',col,' not defined')
                pass
        #adds attributes to entire dataset
        ds_out=ds_out.assign_attrs({'Description': 'Comprehensive dataset of dynamic and stationary datasets for '+config['basics']['output_name']+' Area for '+date.strftime('%m%Y'),
                                    'Author': 'Erik Nixdorf', 'Version' : dataset_version, 'Version_Date' : datetime.now().date().strftime('%d-%m-%Y'),
                                    'CRS' : config['basics']['output_crs']})
        ds_out.to_netcdf(os.getcwd()+config['basics']['output_location']+'\\'+config['basics']['output_name']+'_'+date.strftime('%m%Y')+'.nc',format='NETCDF4_CLASSIC')        
        #renew loop
        del ds_out
        new_month=True
        print('nc file written for year {}...finished'.format(date.strftime('%m%Y')))
        current_month=date.month
    #set initial conditions to False
    initdf=False