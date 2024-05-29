"""Util functions """

import numpy as np
import xarray as xr
import pandas as pd
import dask
import dask.array as da
from datetime import datetime, timedelta

def era5_preprocess(ds):    
    # Convert the longitude coordinates from [0, 360] to [-180, 180]
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    return ds


def co2_preprocess(ds, start_time, end_time):    
    ds = ds.sel(time=slice(start_time, end_time)) 
    return ds


def fix_coords(ds):
    if 'band' in ds.dims:
        ds = ds.rename_dims({'band': 'time'})
        ds = ds.rename_vars({'band': 'time'})

    if 'x' in ds.dims and 'y' in ds.dims:
        ds = ds.rename_dims({'x': 'longitude', 'y': 'latitude'})
        ds = ds.rename_vars({'x': 'longitude', 'y': 'latitude'})
        
    elif 'lon' in ds.dims and 'lat' in ds.dims:
        ds = ds.rename_dims({'lon': 'longitude', 'lat': 'latitude'})
        ds = ds.rename_vars({'lon': 'longitude', 'lat': 'latitude'})
    return ds


def fix_time(ds, start_time):
    year = datetime.strptime(start_time, "%Y-%m-%d").year
    # convert day of year
    if ds.time.size == 1:
        ds['time'] = [datetime.strptime(start_time, "%Y-%m-%d")]
    elif ds.time.dtype == 'int64':
        # Convert day of year to datetime
        ds['time'] = [datetime(year, 1, 1) + timedelta(int(day) - 1) for day in ds.time.values]
    return ds


def remove_encoding(ds):
    # Remove global encoding
    for var in ds.variables:
        ds[var].encoding = {}
    return ds


def interpolation(ds, other):
    # in time
    ds_interpolated = ds.interp(coords={"time": other["time"]}, method='nearest', kwargs={"fill_value": "extrapolate"})
    
    # in space
    ds_interpolated = ds_interpolated.interp(coords={"longitude": other["longitude"], "latitude": other["latitude"]}, method='linear')
    
    return ds_interpolated


def era5land_accumulated_vars(ds, input_name, output_name, scale_factor):    
    input_da = ds[input_name] / scale_factor
    output_da = input_da.diff("time")
    output_da[0::24] = input_da[1::24]  # accumulation starts at t01 instead of t00
    
    t00 = xr.DataArray(np.nan, coords=input_da.isel(time=0).coords) # assign first t00 to none
    output_da = xr.concat([output_da, t00], dim='time')    
    ds[output_name] = output_da
    
    return ds

def map_landcover_to_igbp(landcover_block, lookup_table):
    # Create a new DataArray with "no data" to hold the mapped values 
    mapped_block = da.full_like(landcover_block, fill_value="No data", dtype="U7")

    # For each key-value pair in the lookup table
    for key, value in lookup_table.items():
        # Where the landcover_block equals the current key, assign the corresponding value
        mapped_block = da.where(landcover_block == key, value, mapped_block)
    
    return mapped_block
        

def landcover_to_igbp(ds, landcover_var_name, encoder, lookup_table, igbp_class):
    landcover = ds[landcover_var_name]
    
    # Replace NaN values with "No data" or 255 in the table
    landcover = da.where(da.isnan(landcover), 255, landcover)
    
    igbp = map_landcover_to_igbp(landcover, lookup_table)
    igbp_reshaped = igbp.reshape(-1, 1)

    transformed = encoder.transform(igbp_reshaped)
    
    # Select the columns that correspond to the categories in igbp_class
    indices = [np.where(encoder.categories_[0] == category)[0][0] for category in igbp_class]    
    transformed = transformed[:, indices]

    # Add each column of the transformed array as a new variable in the dataset
    for i in range(transformed.shape[1]):
        ds[f"IGBP_veg_long{i+1}"] = (("time", "latitude", "longitude"), transformed[:, i].reshape(igbp.shape))

    return ds


def training_testing_preprocess(df):
    #filter the outliers
    df = df[(df['LEtot'] < 750) & (df['LEtot'] > -10)]
    df = df[(df['Htot'] < 750) & (df['Htot'] > -500)]
    df = df[df['Actot']>-10]

    # remove nan
    df = df.dropna()
    return df


def igbp_to_landcover(df, encoder, igbp_class):
    
    # Unsorted categories are not yet supported by dask-ml
    igbp_stemmus_scope = np.sort(igbp_class.reshape(-1,1))
    encoder = encoder.fit(igbp_stemmus_scope)
    
    if isinstance(df, pd.DataFrame):
        igbp = df['IGBP_veg_long'].to_numpy().reshape(-1, 1)
    elif isinstance(df, dask.dataframe.DataFrame):
        igbp = df['IGBP_veg_long'].to_dask_array(lengths=True).reshape(-1, 1)

    transformed = encoder.transform(igbp)

    for i in range(transformed.shape[1]):
        df[f"IGBP_veg_long{i+1}"] = transformed[:, i]
    
    df = df.drop('IGBP_veg_long', axis=1)
    return df


def arr_to_ds(arr, input_ds, output_vars):
    
    output_ds = xr.Dataset(coords=input_ds.coords)
    ds_shape = (output_ds.sizes['time'], output_ds.sizes['latitude'], output_ds.sizes['longitude'])

    for i, name in enumerate(output_vars):
        if arr.ndim == 1:
            output_ds[name] = (("time", "latitude", "longitude"), arr.reshape(ds_shape))
        else:
            output_ds[name] = (("time", "latitude", "longitude"), arr[:, i].reshape(ds_shape))

    # mask nan values
    ds_masked = output_ds.where(input_ds["Rin"].notnull())
    return ds_masked