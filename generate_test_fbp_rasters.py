# -*- coding: utf-8 -*-
"""
Created on Thur Aug 8 17:30:00 2024

@author: Gregory A. Greene
"""

import os
import ProcessRasters as pr
from numpy import float64 as f64
from typing import Union, Optional

# Get input folder
input_folder = os.path.join(os.path.dirname(__file__), 'Test_Data', 'Inputs')

# Get input dataset paths
fuel_type_path = os.path.join(input_folder, 'FuelType.tif')

# Get input data
fuel_type_ras = pr.getRaster(fuel_type_path)
fuel_type_profile = fuel_type_ras.profile
fuel_type_array = fuel_type_ras.read()


def gen_test_data(lat: Union[float, int],
                  long: Union[float, int],
                  elevation: Union[float, int],
                  slope: Union[float, int],
                  aspect: Union[float, int],
                  ws: Union[float, int],
                  wd: Union[float, int],
                  ffmc: Union[float, int],
                  bui: Union[float, int],
                  dj: int,
                  pc: Optional[Union[float, int]] = 50,
                  pdf: Optional[Union[float, int]] = 35,
                  gfl: Optional[Union[float, int]] = 0.35,
                  gcf: Optional[Union[float, int]] = 80):
    # Generate output dataset dictionary
    data_dict = {
        'LAT': lat,
        'LONG': long,
        'ELV': elevation,
        'GS': slope,
        'Aspect': aspect,
        'WS': ws,
        'WD': wd,
        'FFMC': ffmc,
        'BUI': bui,
        'Dj': dj,
        'PC': pc,
        'PDF': pdf,
        'GFL': gfl,
        'cc': gcf
    }

    for dset in list(data_dict.keys()):
        dset_array = fuel_type_array * 0 + data_dict.get(dset)
        # Save output datasets
        pr.arrayToRaster(array=dset_array,
                         out_file=os.path.join(input_folder, f'{dset}.tif'),
                         ras_profile=fuel_type_profile,
                         data_type=f64)


if __name__ == '__main__':
    _lat = 62.245544
    _long = -133.839203
    _elevation = 1176
    _slope = 7
    _aspect = 53
    _ws = 24
    _wd = 266
    _ffmc = 92
    _bui = 31
    _pc = 50
    _pdf = 50
    _gfl = 0.35
    _gcf = 80

    # Generate FBP rasters
    gen_test_data(lat=_lat, long=_long,
                  elevation=_elevation, slope=_slope, aspect=_aspect,
                  ws=_ws, wd=_wd, ffmc=_ffmc, bui=_bui,
                  pc=_pc, pdf=_pdf, gfl=_gfl, gcf=_gcf)
